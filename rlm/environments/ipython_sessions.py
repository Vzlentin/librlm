"""Kernel-session adapters used by :class:`IPythonREPL`."""

from __future__ import annotations

import copy
import importlib
import json
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rlm.core.child_execution import ChildExecution
from rlm.core.types import FinalValue, REPLResult
from rlm.environments.base_env import extract_tool_value
from rlm.environments.ipython_async import AsyncRLMHost, QueryBatchRunner
from rlm.environments.ipython_kernel import (
    _CUSTOM_TOOLS_KEY,
    IPythonOutputStream,
    kernel_bootstrap_code,
)

_ANSI_RE = re.compile(
    r"\x1b(?:"
    r"\[[0-9;?]*[ -/]*[@-~]"
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|[@-Z\\-_]"
    r")"
)


class KernelSession(ABC):
    """Small mode-independent interface for one persistent IPython kernel."""

    @abstractmethod
    def execute(self, code: str, *, timeout: float | None) -> REPLResult:
        """Execute one user cell and return its complete observable result."""

    @abstractmethod
    def assign(
        self,
        name: str,
        value: Any,
        *,
        alias: str | None = None,
        timeout: float | None,
    ) -> None:
        """Persist one host value in the kernel namespace."""

    @abstractmethod
    def finalize_completion(self) -> None:
        """Join work owned by the ending RLM completion."""

    @abstractmethod
    def close(self) -> None:
        """Release every resource owned by this kernel mode."""


class SubprocessKernelSession(KernelSession):
    """KernelManager adapter with bridge execution and hard interruption semantics."""

    def __init__(
        self,
        *,
        working_dir: str,
        startup_timeout: float,
        subcall_timeout: float | None,
        custom_tools: dict[str, Any],
        output_callback: Callable[[dict[str, Any]], None] | None,
        child_execution: ChildExecution | None,
        query_runner: QueryBatchRunner,
        recursive_queries: bool,
        max_concurrent_subcalls: int,
    ) -> None:
        try:
            from jupyter_client.manager import KernelManager
        except ImportError as error:
            raise ImportError(
                "jupyter_client and ipykernel are required for IPythonREPL "
                "in subprocess mode. Install with: pip install 'rlms[ipython]' "
                "or pip install jupyter_client ipykernel"
            ) from error

        self.working_dir = working_dir
        self.startup_timeout = startup_timeout
        self.subcall_timeout = subcall_timeout
        self.custom_tools = custom_tools
        self.output_callback = output_callback
        self._restart_state: dict[str, Any] = {}
        self._execution_lock = threading.RLock()
        self._lifecycle = threading.Condition()
        self._lifecycle_state: str = "open"
        self._active_operations = 0
        self.host = AsyncRLMHost(
            child_execution,
            query_runner=query_runner,
            recursive_queries=recursive_queries,
            max_concurrent=max_concurrent_subcalls,
        )
        self.transport_dir = Path(tempfile.mkdtemp(prefix="ipython_transport_"))
        self.manager: Any = None
        self.client: Any = None
        try:
            self.host.start()
            self.manager = KernelManager(kernel_name="python3")
            kernel_spec = self.manager.kernel_spec
            if kernel_spec is None:
                raise RuntimeError("Jupyter did not provide a kernel specification")
            kernel_spec.argv = [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ]
            self.manager.start_kernel(cwd=self.working_dir)
            self.client = self.manager.client()
            self.client.start_channels()
            self.client.wait_for_ready(timeout=self.startup_timeout)
            result = self._execute_control(
                kernel_bootstrap_code(
                    self.host.address,
                    self.host.auth_token,
                    self.subcall_timeout,
                ),
                timeout=self.startup_timeout,
            )
            if result.stderr:
                raise RuntimeError(f"Kernel bootstrap failed:\n{result.stderr}")
            self._inject_custom_tools()
        except BaseException as startup_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                startup_error.add_note(f"subprocess kernel cleanup also failed: {cleanup_error!r}")
            raise

    def _inject_custom_tools(self) -> None:
        try:
            dill = importlib.import_module("dill")
        except ImportError:
            dill = None

        values = {name: extract_tool_value(entry) for name, entry in self.custom_tools.items()}
        if dill is not None:
            expressions: list[str] = []
            for name, value in values.items():
                try:
                    payload = dill.dumps(value, recurse=True).hex()
                except Exception as error:
                    raise RuntimeError(
                        f"Custom tool {name!r} could not be pickled with dill: {error}"
                    ) from error
                expressions.append(f"{name!r}: _rlm_dill.loads(bytes.fromhex({payload!r}))")
            assignment = (
                f"import dill as _rlm_dill\n{_CUSTOM_TOOLS_KEY} = {{{', '.join(expressions)}}}"
            )
        else:
            try:
                payload = json.dumps(values)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "Custom tools are not JSON-serializable and dill is not installed. "
                    "Install dill (pip install dill) to inject arbitrary callables/objects. "
                    f"({error})"
                ) from error
            assignment = (
                f"import json as _rlm_json\n{_CUSTOM_TOOLS_KEY} = _rlm_json.loads({payload!r})"
            )

        code = (
            assignment
            + "\n"
            + kernel_bootstrap_code(
                self.host.address,
                self.host.auth_token,
                self.subcall_timeout,
            )
        )
        result = self._execute_control(code, timeout=None)
        if result.stderr:
            raise RuntimeError(f"Failed to inject custom tools into kernel: {result.stderr}")

    @contextmanager
    def _operation(self) -> Iterator[None]:
        with self._lifecycle:
            self._require_open()
            self._active_operations += 1
        try:
            yield
        finally:
            with self._lifecycle:
                self._active_operations -= 1
                self._lifecycle.notify_all()

    def execute(self, code: str, *, timeout: float | None) -> REPLResult:
        with self._execution_lock:
            with self._operation():
                return self._execute(code, timeout=timeout, track_execution=True)

    def _execute_control(self, code: str, *, timeout: float | None) -> REPLResult:
        """Run internal bootstrap or assignment code without a user execution."""
        return self._execute(code, timeout=timeout, track_execution=False)

    @staticmethod
    def _require_successful_reply(reply: Any, *, operation: str) -> None:
        if not isinstance(reply, dict):
            raise RuntimeError(f"Kernel {operation} returned no execute reply")
        content = reply.get("content", {})
        if content.get("status") == "ok":
            return
        name = str(content.get("ename", "KernelError"))
        value = str(content.get("evalue", "unknown kernel failure"))
        raise RuntimeError(f"Kernel {operation} failed: {name}: {value}")

    def _execute(
        self,
        code: str,
        *,
        timeout: float | None,
        track_execution: bool,
    ) -> REPLResult:
        if self.client is None or self.manager is None:
            raise RuntimeError("IPython subprocess kernel is closed")
        start_time = time.perf_counter()
        cell_id = uuid.uuid4().hex if track_execution else None
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        error_info: dict[str, Any] | None = None
        summary = None
        successful = False
        execution_ended = False
        observer_error: Exception | None = None

        def accept_output(event: dict[str, Any]) -> None:
            nonlocal observer_error
            if event["type"] == "clear":
                stdout_parts.clear()
                stderr_parts.clear()
            else:
                kind = event["kind"]
                text = event["text"]
                if kind == "stderr":
                    stderr_parts.append(text)
                elif kind == "error":
                    stderr_parts.append("\n" + _ANSI_RE.sub("", text))
                else:
                    stdout_parts.append(text)
                    if kind in ("result", "display") and not text.endswith("\n"):
                        stdout_parts.append("\n")
            if self.output_callback is not None:
                try:
                    self.output_callback(copy.deepcopy(event))
                except Exception as error:
                    observer_error = error
                    self.output_callback = None
                except BaseException:
                    self.output_callback = None
                    raise

        output_stream = IPythonOutputStream(accept_output)

        def output_hook(message: dict[str, Any]) -> None:
            nonlocal error_info
            msg_type = message.get("header", {}).get("msg_type")
            content = message.get("content", {})
            if msg_type == "error":
                error_info = content
            output_stream.feed(msg_type, content)

        if cell_id is not None:
            self.host.begin_execution(cell_id)
        try:
            if cell_id is not None:
                try:
                    activation_reply = self.client.execute_interactive(
                        kernel_bootstrap_code(
                            self.host.address,
                            self.host.auth_token,
                            self.subcall_timeout,
                            execution_id=cell_id,
                            working_dir=self.working_dir,
                        ),
                        timeout=self.startup_timeout,
                        output_hook=lambda _message: None,
                        store_history=False,
                        stop_on_error=False,
                        allow_stdin=False,
                    )
                    self._require_successful_reply(
                        activation_reply,
                        operation="activation",
                    )
                except Exception as error:
                    raise RuntimeError(f"Failed to activate IPython execution: {error}") from error

            timed_out = False
            try:
                self.client.execute_interactive(
                    code,
                    timeout=timeout,
                    output_hook=output_hook,
                    store_history=False,
                    stop_on_error=False,
                    allow_stdin=False,
                )
            except TimeoutError:
                timed_out = True
                if self._is_open() and not self._interrupt_and_drain(output_hook):
                    if cell_id is not None:
                        summary = self.host.end_execution(cell_id, successful=False)
                        execution_ended = True
                    self._restart_kernel()
                stderr_parts.append(
                    f"\nTimeoutError: cell execution exceeded {timeout}s and was interrupted"
                )
            except BaseException:
                if self._is_open() and not self._interrupt_and_drain(output_hook):
                    if cell_id is not None:
                        summary = self.host.end_execution(cell_id, successful=False)
                        execution_ended = True
                    self._restart_kernel()
                raise
            successful = not timed_out and error_info is None
        finally:
            if cell_id is not None and not execution_ended:
                summary = self.host.end_execution(cell_id, successful=successful)

        if observer_error is not None:
            warnings.warn(
                f"IPython output callback failed and was disabled: {observer_error}",
                RuntimeWarning,
                stacklevel=2,
            )
        return REPLResult(
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            locals={},
            execution_time=time.perf_counter() - start_time,
            rlm_calls=[] if summary is None else summary.completions,
            final=FinalValue.absent() if summary is None else summary.final,
        )

    def _interrupt_and_drain(
        self,
        output_hook: Callable[[dict[str, Any]], None],
        grace: float = 0.75,
    ) -> bool:
        """Interrupt once and consume IOPub until idle within a bounded grace."""
        assert self.manager is not None and self.client is not None
        try:
            self.manager.interrupt_kernel()
        except Exception:
            return False
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            timeout = max(0.01, min(0.05, deadline - time.monotonic()))
            try:
                message = self.client.get_iopub_msg(timeout=timeout)
            except queue.Empty:
                if not self.manager.is_alive():
                    return False
                continue
            msg_type = message.get("header", {}).get("msg_type")
            content = message.get("content", {})
            if msg_type == "status" and content.get("execution_state") == "idle":
                return True
            output_hook(message)
        return False

    def _restart_kernel(self) -> None:
        """Replace a kernel that ignored interruption and restore host-owned state."""
        if self.manager is None or self.client is None:
            raise RuntimeError("IPython subprocess kernel is closed")
        restart_state = dict(self._restart_state)
        self.client.stop_channels()
        self.manager.restart_kernel(now=True)
        self.client = self.manager.client()
        self.client.start_channels()
        self.client.wait_for_ready(timeout=self.startup_timeout)
        bootstrap = self._execute_control(
            kernel_bootstrap_code(
                self.host.address,
                self.host.auth_token,
                self.subcall_timeout,
            ),
            timeout=self.startup_timeout,
        )
        if bootstrap.stderr:
            raise RuntimeError(f"Kernel bootstrap after timeout failed:\n{bootstrap.stderr}")
        self._inject_custom_tools()
        self._restart_state.clear()
        for name, value in restart_state.items():
            self.assign(name, value, timeout=self.startup_timeout)

    def assign(
        self,
        name: str,
        value: Any,
        *,
        alias: str | None = None,
        timeout: float | None,
    ) -> None:
        with self._execution_lock:
            with self._operation():
                if isinstance(value, str):
                    path = self.transport_dir / f"{name}.txt"
                    path.write_text(value)
                    code = f"with open({str(path)!r}, 'r') as _rlm_f:\n    {name} = _rlm_f.read()"
                else:
                    path = self.transport_dir / f"{name}.json"
                    path.write_text(json.dumps(value))
                    code = (
                        "import json as _rlm_json\n"
                        f"with open({str(path)!r}, 'r') as _rlm_f:\n"
                        f"    {name} = _rlm_json.load(_rlm_f)"
                    )
                if alias is not None:
                    code += f"\n{alias} = {name}"
                result = self._execute_control(code, timeout=timeout)
                if result.stderr:
                    raise RuntimeError(f"Failed to assign {name}: {result.stderr}")
                copied = copy.deepcopy(value)
                self._restart_state[name] = copied
                if alias is not None:
                    self._restart_state[alias] = copied

    def restart_state_snapshot(self) -> dict[str, Any]:
        """Return host-assigned values retained solely for kernel restart."""
        with self._operation():
            return dict(self._restart_state)

    def _require_open(self) -> None:
        if self._lifecycle_state != "open":
            raise RuntimeError("IPython subprocess kernel is closed")

    def _is_open(self) -> bool:
        with self._lifecycle:
            return self._lifecycle_state == "open"

    def finalize_completion(self) -> None:
        with self._execution_lock:
            with self._operation():
                self.host.finalize_completion()

    @staticmethod
    def _raise_teardown_errors(errors: list[BaseException]) -> None:
        if not errors:
            return
        if len(errors) == 1:
            error = errors[0]
            raise error.with_traceback(error.__traceback__)
        raise BaseExceptionGroup("IPython subprocess teardown failed", errors)

    def close(self) -> None:
        with self._lifecycle:
            while self._lifecycle_state == "closing":
                self._lifecycle.wait()
            if self._lifecycle_state == "closed":
                return
            self._lifecycle_state = "closing"
            client = self.client
            manager = self.manager

        errors: list[BaseException] = []

        def attempt(operation: Callable[[], Any]) -> None:
            try:
                operation()
            except BaseException as error:
                errors.append(error)

        attempt(self.host.reset)
        if manager is not None:
            attempt(manager.interrupt_kernel)

        with self._lifecycle:
            drained = self._lifecycle.wait_for(
                lambda: self._active_operations == 0,
                timeout=1.0,
            )

        if manager is not None:
            attempt(lambda: manager.shutdown_kernel(now=True))
        if client is not None:
            attempt(client.stop_channels)

        if not drained:
            with self._lifecycle:
                drained = self._lifecycle.wait_for(
                    lambda: self._active_operations == 0,
                    timeout=5.0,
                )
        if not drained:
            errors.append(RuntimeError("active IPython kernel operation did not stop during close"))

        attempt(self.host.stop)
        attempt(lambda: shutil.rmtree(self.transport_dir))

        with self._lifecycle:
            self.client = None
            self.manager = None
            self._lifecycle_state = "closed"
            self._lifecycle.notify_all()

        self._raise_teardown_errors(errors)
