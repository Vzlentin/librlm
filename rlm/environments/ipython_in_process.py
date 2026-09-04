"""In-process IPython execution and recursive query coordination."""

from __future__ import annotations

import atexit
import copy
import io
import os
import signal
import sys
import threading
import time
import traceback
import types
import uuid
from collections.abc import Callable
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any, cast

from rlm.core.child_execution import ChildExecution, ChildOutcome, ChildRequest
from rlm.core.types import REPLResult, RLMChatCompletion
from rlm.environments.base_env import FinalAnswerDict, extract_tool_value
from rlm.environments.ipython_kernel import install_scaffold
from rlm.environments.ipython_protocol import QueryKind, query_result_text
from rlm.environments.ipython_queries import QueryCoordinator
from rlm.environments.ipython_sessions import KernelSession


class IPythonCellTimeoutError(BaseException):
    """Terminal sentinel propagated through every in-process scaffold path."""

    def __init__(self, message: str, owner: object) -> None:
        super().__init__(message)
        self.owner = owner


_IPYTHON_INTERNAL_NAMES: frozenset[str] = frozenset(
    {
        "In",
        "Out",
        "exit",
        "quit",
        "get_ipython",
        "open",
        "_oh",
        "_dh",
        "_ih",
        "_i",
        "_ii",
        "_iii",
        "_",
        "__",
        "___",
    }
)

_IN_PROCESS_RECURSION = threading.local()
_WORKER_RECURSION_ERROR = (
    "a recursive in-process child cannot start from a worker thread; "
    "run recursive helpers on the cell thread or use subprocess mode"
)


@contextmanager
def _in_process_recursive_call():
    depth = getattr(_IN_PROCESS_RECURSION, "depth", 0)
    _IN_PROCESS_RECURSION.depth = depth + 1
    try:
        yield
    finally:
        _IN_PROCESS_RECURSION.depth = depth


class _InProcessStateOwner:
    """Serialize process globals and reject unsafe cross-thread nested cells."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._owner_thread: int | None = None
        self._depth = 0

    def recursive_worker_conflict(self) -> bool:
        current = threading.get_ident()
        nested = getattr(_IN_PROCESS_RECURSION, "depth", 0) > 0
        with self._condition:
            return nested and self._owner_thread not in (None, current)

    @contextmanager
    def hold(self):
        current = threading.get_ident()
        nested = getattr(_IN_PROCESS_RECURSION, "depth", 0) > 0
        with self._condition:
            if self._owner_thread == current:
                self._depth += 1
            else:
                if nested and self._owner_thread is not None:
                    raise RuntimeError(_WORKER_RECURSION_ERROR)
                self._condition.wait_for(lambda: self._owner_thread is None)
                self._owner_thread = current
                self._depth = 1
        try:
            yield
        finally:
            with self._condition:
                self._depth -= 1
                if self._depth == 0:
                    self._owner_thread = None
                    self._condition.notify_all()


_IN_PROCESS_STATE_OWNER = _InProcessStateOwner()


@dataclass(slots=True)
class _InProcessQueryExecution:
    cancel: threading.Event
    timeout_owner: object
    completions: list[RLMChatCompletion] = field(default_factory=list)
    in_flight: int = 0
    accepting: bool = True


class _InProcessCellQueries:
    """Execution-scoped scaffold capability with no kernel lifecycle access."""

    def __init__(
        self,
        owner: _InProcessQueryChannel,
        execution: _InProcessQueryExecution,
    ) -> None:
        self._owner = owner
        self._execution = execution

    def llm_query(self, prompt: str, model: str | None = None) -> str:
        return self._owner.query(self._execution, "llm", [prompt], model)[0]

    def llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        return self._owner.query(self._execution, "llm", prompts, model)

    def rlm_query(self, prompt: str, model: str | None = None) -> str:
        return self._owner.query(self._execution, "rlm", [prompt], model)[0]

    def rlm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        return self._owner.query(self._execution, "rlm", prompts, model)


class _InProcessQueryChannel:
    """Own execution-scoped query admission, completion capture, and finalization."""

    def __init__(self, coordinator: QueryCoordinator) -> None:
        self._coordinator = coordinator
        self._condition = threading.Condition()
        self._completion_executions: list[_InProcessQueryExecution] = []
        self._closed = False

    def inactive_queries(self) -> _InProcessCellQueries:
        return _InProcessCellQueries(
            self,
            _InProcessQueryExecution(threading.Event(), object(), accepting=False),
        )

    def begin_cell(
        self,
        cancel: threading.Event,
        timeout_owner: object,
    ) -> _InProcessCellQueries:
        with self._condition:
            if self._closed:
                raise RuntimeError("in-process query channel is closed")
            execution = _InProcessQueryExecution(cancel, timeout_owner)
            self._completion_executions.append(execution)
            return _InProcessCellQueries(self, execution)

    def end_cell(self, queries: _InProcessCellQueries) -> list[RLMChatCompletion]:
        execution = queries._execution
        with self._condition:
            execution.accepting = False
            return list(execution.completions)

    def query(
        self,
        execution: _InProcessQueryExecution,
        kind: QueryKind,
        prompts: list[str],
        model: str | None,
    ) -> list[str]:
        with self._condition:
            if self._closed or not execution.accepting:
                raise RuntimeError("query belongs to an inactive IPython execution")
            execution.in_flight += 1
        try:
            if execution.cancel.is_set():
                raise IPythonCellTimeoutError(
                    "cell execution timed out",
                    execution.timeout_owner,
                )
            outcomes = self._coordinator.run(kind, prompts, model, execution.cancel)
            if execution.cancel.is_set():
                raise IPythonCellTimeoutError(
                    "cell execution timed out",
                    execution.timeout_owner,
                )
            with self._condition:
                execution.completions.extend(
                    outcome.completion for outcome in outcomes if outcome.completion is not None
                )
            return [query_result_text(kind, outcome.result) for outcome in outcomes]
        finally:
            with self._condition:
                execution.in_flight -= 1
                self._condition.notify_all()

    def finalize_completion(self) -> None:
        with self._condition:
            executions = self._completion_executions
            self._completion_executions = []
            for execution in executions:
                execution.accepting = False
            self._condition.wait_for(
                lambda: all(execution.in_flight == 0 for execution in executions)
            )

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            executions = self._completion_executions
            self._completion_executions = []
            for execution in executions:
                execution.accepting = False
                execution.cancel.set()
            self._condition.wait_for(
                lambda: all(execution.in_flight == 0 for execution in executions)
            )


class InProcessKernelSession(KernelSession):
    """InteractiveShell adapter with process-local signal and cwd semantics."""

    def __init__(
        self,
        *,
        working_dir: str,
        original_cwd: str,
        lm_handler_address: Callable[[], tuple[str, int] | None],
        depth: int,
        child_execution: ChildExecution | None,
        custom_tools: dict[str, Any],
    ) -> None:
        try:
            from IPython.core.interactiveshell import InteractiveShell
            from traitlets.config import Config
        except ImportError as error:
            raise ImportError(
                "IPython is required for IPythonREPL. Install with: "
                "pip install 'rlms[ipython]' or pip install ipython"
            ) from error

        if _IN_PROCESS_STATE_OWNER.recursive_worker_conflict():
            raise RuntimeError(_WORKER_RECURSION_ERROR)

        self.working_dir = working_dir
        self.original_cwd = original_cwd
        self.custom_tools = custom_tools
        self._query_channel = _InProcessQueryChannel(
            QueryCoordinator(
                lm_handler_address=lm_handler_address,
                depth=depth,
                child_execution=(
                    self._recursive_child(child_execution) if child_execution is not None else None
                ),
            )
        )
        self._last_final_answer: str | None = None
        self._lifecycle = threading.Condition(threading.RLock())
        self._lifecycle_state: str = "open"

        config = Config()
        config.HistoryAccessor.enabled = False
        self.user_module = types.ModuleType(
            f"_rlm_ipython_main_{uuid.uuid4().hex[:8]}",
            doc="Per-instance namespace for an in-process IPythonREPL.",
        )
        try:
            self.shell = InteractiveShell(config=config, user_module=self.user_module)
            self._restore_scaffold(self._query_channel.inactive_queries())
        except BaseException as initialization_error:
            try:
                if hasattr(self, "shell"):
                    self.close()
                else:
                    self._query_channel.close()
                    sys.modules.pop(self.user_module.__name__, None)
            except BaseException as cleanup_error:
                initialization_error.add_note(
                    f"in-process kernel cleanup also failed: {cleanup_error!r}"
                )
            raise

    @staticmethod
    def _recursive_child(
        child_execution: ChildExecution,
    ) -> Callable[[ChildRequest, threading.Event], ChildOutcome]:
        def run(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
            with _in_process_recursive_call():
                return child_execution.run(request, cancel)

        return run

    def _capture_answer(self, content: Any) -> None:
        self._last_final_answer = str(content)

    def _restore_scaffold(self, queries: _InProcessCellQueries) -> None:
        namespace = self.shell.user_ns
        current = namespace.get("answer")
        if not isinstance(current, FinalAnswerDict):
            replacement = FinalAnswerDict(on_ready=self._capture_answer)
            if isinstance(current, dict):
                for key, value in current.items():
                    dict.__setitem__(replacement, key, value)
                if current.get("ready") and self._last_final_answer is None:
                    self._last_final_answer = str(current.get("content", ""))
            current = replacement
        install_scaffold(
            self.shell,
            queries,
            answer=current,
            custom_tools={
                name: extract_tool_value(entry) for name, entry in self.custom_tools.items()
            },
        )

    def execute(self, code: str, *, timeout: float | None) -> REPLResult:
        if timeout is not None and (
            sys.platform == "win32"
            or threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "SIGALRM")
        ):
            raise RuntimeError(
                "in-process cell_timeout requires SIGALRM on the main thread; "
                "use subprocess mode for timeout enforcement in this caller"
            )
        with self._lifecycle:
            self._require_open()
            with _IN_PROCESS_STATE_OWNER.hold():
                return self._execute_owned(code, timeout=timeout)

    def _execute_owned(self, code: str, *, timeout: float | None) -> REPLResult:
        use_alarm = timeout is not None
        previous_handler: Any = None
        previous_deadline: float | None = None
        if use_alarm:
            previous_handler = signal.getsignal(signal.SIGALRM)
            previous_remaining = signal.getitimer(signal.ITIMER_REAL)[0]
            if previous_remaining > 0:
                previous_deadline = getattr(
                    previous_handler,
                    "_rlm_ipython_deadline",
                    None,
                )
                if previous_deadline is None:
                    raise RuntimeError(
                        "in-process cell_timeout cannot replace an active external ITIMER_REAL"
                    )

        start_time = time.perf_counter()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        self._last_final_answer = None
        cell_cancel = threading.Event()
        cell_owner = object()
        queries = self._query_channel.begin_cell(cell_cancel, cell_owner)
        self._restore_scaffold(queries)
        timed_out = False

        def alarm_handler(_signum: int, _frame: Any) -> None:
            cell_cancel.set()
            raise IPythonCellTimeoutError(
                f"cell execution exceeded {timeout}s and was interrupted",
                cell_owner,
            )

        own_deadline = time.monotonic() + timeout if timeout is not None else None
        cast(Any, alarm_handler)._rlm_ipython_deadline = own_deadline
        result: Any = None
        try:
            with self._temporary_cwd():
                if use_alarm:
                    assert own_deadline is not None
                    effective_handler = alarm_handler
                    effective_deadline = own_deadline
                    if previous_deadline is not None and previous_deadline <= own_deadline:
                        effective_handler = previous_handler
                        effective_deadline = previous_deadline
                    signal.signal(signal.SIGALRM, effective_handler)
                    signal.setitimer(
                        signal.ITIMER_REAL,
                        max(0.000001, effective_deadline - time.monotonic()),
                    )
                try:
                    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                        try:
                            result = self.shell.run_cell(
                                code,
                                store_history=False,
                                silent=False,
                            )
                        except IPythonCellTimeoutError as error:
                            if error.owner is not cell_owner:
                                raise
                            timed_out = True
                            stderr_buffer.write(f"\nTimeoutError: {error}")
                        except Exception as error:
                            stderr_buffer.write(f"\n{type(error).__name__}: {error}")
                finally:
                    if use_alarm:
                        signal.signal(signal.SIGALRM, signal.SIG_IGN)
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        signal.signal(
                            signal.SIGALRM,
                            previous_handler if previous_handler is not None else signal.SIG_DFL,
                        )
                        if previous_deadline is not None:
                            remaining = previous_deadline - time.monotonic()
                            if remaining > 0:
                                signal.setitimer(signal.ITIMER_REAL, remaining)
                            elif callable(previous_handler):
                                previous_handler(signal.SIGALRM, None)

            if result is not None:
                for error in (result.error_before_exec, result.error_in_exec):
                    if isinstance(error, IPythonCellTimeoutError):
                        if error.owner is not cell_owner:
                            raise error
                        timed_out = True
                        cell_cancel.set()
                        stderr_buffer.write(f"\nTimeoutError: {error}")
                    elif error is not None:
                        stderr_buffer.write(
                            "\n"
                            + "".join(
                                traceback.format_exception(
                                    type(error),
                                    error,
                                    error.__traceback__,
                                )
                            )
                        )
            if cell_cancel.is_set():
                timed_out = True
            if timed_out:
                cell_cancel.set()
                if "TimeoutError" not in stderr_buffer.getvalue():
                    stderr_buffer.write(
                        f"\nTimeoutError: cell execution exceeded {timeout}s and was interrupted"
                    )
        finally:
            rlm_calls = self._query_channel.end_cell(queries)
            self._restore_scaffold(self._query_channel.inactive_queries())

        result_locals = self._locals_unlocked()
        final_answer = self._last_final_answer
        self._last_final_answer = None
        return REPLResult(
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
            locals=result_locals,
            execution_time=time.perf_counter() - start_time,
            rlm_calls=rlm_calls,
            final_answer=final_answer,
        )

    def assign(
        self,
        name: str,
        value: Any,
        *,
        alias: str | None = None,
        timeout: float | None,
    ) -> None:
        with self._lifecycle:
            self._require_open()
            copied = copy.deepcopy(value)
            self.shell.user_ns[name] = copied
            if alias is not None:
                self.shell.user_ns[alias] = copied

    def namespace_snapshot(self) -> dict[str, Any]:
        """Return the filtered live IPython user namespace."""
        with self._lifecycle:
            self._require_open()
            return self._locals_unlocked()

    def _locals_unlocked(self) -> dict[str, Any]:
        return {
            name: value
            for name, value in self.shell.user_ns.items()
            if not name.startswith("_") and name not in _IPYTHON_INTERNAL_NAMES
        }

    @contextmanager
    def _temporary_cwd(self):
        old = os.getcwd()
        try:
            os.chdir(self.working_dir)
            yield
        finally:
            try:
                os.chdir(old)
            except FileNotFoundError:
                os.chdir(self.original_cwd)

    def _require_open(self) -> None:
        if self._lifecycle_state != "open":
            raise RuntimeError("in-process IPython kernel is closed")

    def finalize_completion(self) -> None:
        with self._lifecycle:
            self._require_open()
            self._query_channel.finalize_completion()

    def close(self) -> None:
        with self._lifecycle:
            while self._lifecycle_state == "closing":
                self._lifecycle.wait()
            if self._lifecycle_state == "closed":
                return
            self._lifecycle_state = "closing"

        errors: list[BaseException] = []

        def attempt(operation: Callable[[], Any]) -> None:
            try:
                operation()
            except BaseException as error:
                errors.append(error)

        try:
            attempt(self._query_channel.close)
            attempt(lambda: atexit.unregister(self.shell.atexit_operations))
            if not _IN_PROCESS_STATE_OWNER.recursive_worker_conflict():
                attempt(self._reset_shell)
            sys.modules.pop(self.user_module.__name__, None)
        finally:
            with self._lifecycle:
                self._lifecycle_state = "closed"
                self._lifecycle.notify_all()

        if len(errors) == 1:
            error = errors[0]
            raise error.with_traceback(error.__traceback__)
        if errors:
            raise BaseExceptionGroup("in-process IPython cleanup failed", errors)

    def _reset_shell(self) -> None:
        with _IN_PROCESS_STATE_OWNER.hold():
            self.shell.reset(new_session=False)
