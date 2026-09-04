"""Persistent IPython REPL with in-process and subprocess kernel sessions."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable
from typing import Any, Literal

from rlm.core.child_execution import (
    Cancellation,
    ChildExecution,
    ChildOutcome,
    ChildRequest,
)
from rlm.core.types import REPLResult, RLMChatCompletion, UsageSummary
from rlm.environments.base_env import (
    CompletionFinalization,
    NonIsolatedEnv,
    validate_custom_tools,
)
from rlm.environments.ipython_in_process import InProcessKernelSession
from rlm.environments.ipython_queries import QueryCoordinator
from rlm.environments.ipython_sessions import KernelSession, SubprocessKernelSession

KernelMode = Literal["in_process", "subprocess"]


class IPythonREPL(NonIsolatedEnv):
    """IPython-backed REPL using one kernel-session adapter.

    ``kernel_mode="in_process"`` uses a dedicated ``InteractiveShell`` in the
    parent process. It is fast, but cwd, output redirection, and Unix alarms are
    process-global while a cell runs. ``kernel_mode="subprocess"`` uses an
    isolated ``ipykernel`` plus an authenticated host for queries and async
    handles. Both modes serialize ``execute_code`` on this instance.

    Args:
        lm_handler_address: Address of the LM handler socket server.
        context_payload: Initial value loaded as ``context`` and ``context_0``.
        setup_code: Optional cell executed after initial context loading.
        persistent: Preserve versioned context and history across RLM calls.
        depth: RLM depth used for LM routing.
        subcall_fn: Recursive callback used by ``rlm_query`` in both modes and
            by ``rlm.spawn`` in subprocess mode.
        custom_tools: Values injected into the kernel namespace.
        kernel_mode: ``"in_process"`` or ``"subprocess"``.
        cell_timeout: Optional per-cell timeout. Subprocess mode interrupts the
            kernel. In-process mode uses ``SIGALRM`` when available on the main
            thread and rejects an already-active external ``ITIMER_REAL``.
        startup_timeout: Seconds allowed for subprocess startup.
        subcall_timeout: Optional kernel-to-host request timeout.
        max_concurrent_subcalls: Shared recursive callback concurrency limit.
            In-process recursive batches remain sequential on the cell thread.
        working_dir: Kernel cwd. A temporary directory is owned when omitted.
        output_callback: Optional normalized subprocess output callback.
            Rejected in in-process mode.
        async_child_runner: Optional subprocess ``rlm.spawn`` adapter returning
            one ``RLMChatCompletion``. Wire results are derived by the host.
            Rejected in in-process mode.
    """

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        subcall_fn: (ChildExecution | Callable[[str, str | None], RLMChatCompletion] | None) = None,
        custom_tools: dict[str, Any] | None = None,
        kernel_mode: KernelMode = "in_process",
        cell_timeout: float | None = None,
        startup_timeout: float = 60.0,
        subcall_timeout: float | None = None,
        max_concurrent_subcalls: int = 4,
        working_dir: str | None = None,
        output_callback: Callable[[dict[str, Any]], None] | None = None,
        async_child_runner: (
            Callable[[ChildRequest, Cancellation], RLMChatCompletion] | None
        ) = None,
    ) -> None:
        if kernel_mode not in ("in_process", "subprocess"):
            raise ValueError(
                f"kernel_mode must be 'in_process' or 'subprocess', got {kernel_mode!r}"
            )
        if kernel_mode == "in_process" and async_child_runner is not None:
            raise ValueError("async_child_runner is available only in subprocess mode")
        if kernel_mode == "in_process" and output_callback is not None:
            raise ValueError("output_callback is available only in subprocess mode")
        if startup_timeout <= 0:
            raise ValueError(f"startup_timeout must be positive, got {startup_timeout!r}")
        if subcall_timeout is not None and subcall_timeout <= 0:
            raise ValueError(
                f"subcall_timeout must be positive or None (got {subcall_timeout!r}); "
                "use None to disable the timeout."
            )
        if cell_timeout is not None and cell_timeout <= 0:
            raise ValueError(f"cell_timeout must be positive or None (got {cell_timeout!r})")
        super().__init__(
            persistent=persistent,
            depth=depth,
            max_concurrent_subcalls=max_concurrent_subcalls,
        )

        self._lock = threading.RLock()
        self._lifecycle = threading.Condition()
        self._lifecycle_state: Literal["open", "closing", "closed"] = "open"
        self._finalizing = False

        self.lm_handler_address = lm_handler_address
        self.subcall_fn = subcall_fn
        self.kernel_mode: KernelMode = kernel_mode
        self.cell_timeout = cell_timeout
        self.startup_timeout = startup_timeout
        self.subcall_timeout = subcall_timeout
        self.output_callback = output_callback
        self.async_child_runner = async_child_runner
        self.custom_tools = custom_tools or {}
        validate_custom_tools(self.custom_tools)

        self.original_cwd = os.getcwd()
        if working_dir is None:
            self.working_dir = tempfile.mkdtemp(prefix=f"ipython_env_{uuid.uuid4()}_")
            self._owns_working_dir = True
        else:
            resolved_working_dir = os.path.abspath(working_dir)
            if not os.path.isdir(resolved_working_dir):
                raise ValueError(f"working_dir is not a directory: {working_dir!r}")
            self.working_dir = resolved_working_dir
            self._owns_working_dir = False
        self.temp_dir = self.working_dir

        self._accounting_lock = threading.Lock()
        self._pending_accounting = UsageSummary.empty()
        self._subcall_threads: set[int] = set()
        self._executing_threads: set[int] = set()
        self._subcall_threads_lock = threading.Lock()
        self._context_count = 0
        self._history_count = 0
        self._kernel: KernelSession | None = None
        self._public_locals_snapshot: Callable[[], dict[str, Any]] | None = None
        self._owned_child_executions: list[ChildExecution] = []
        self._recursive_children = self._recursive_child_execution(subcall_fn)
        self._spawn_children = self._spawn_child_execution(async_child_runner)
        self._child_executions = list(
            {
                id(execution): execution
                for execution in (self._recursive_children, self._spawn_children)
                if execution is not None
            }.values()
        )

        try:
            self.setup()
            if context_payload is not None:
                self.load_context(context_payload)
            if setup_code:
                self.execute_code(setup_code)
        except BaseException as startup_error:
            try:
                self.cleanup()
            except BaseException as cleanup_error:
                startup_error.add_note(f"IPythonREPL cleanup also failed: {cleanup_error!r}")
            raise

    def _recursive_child_execution(
        self,
        subcall_fn: ChildExecution | Callable[[str, str | None], RLMChatCompletion] | None,
    ) -> ChildExecution | None:
        if subcall_fn is None:
            return None
        if isinstance(subcall_fn, ChildExecution):
            return subcall_fn

        def run_legacy(
            request: ChildRequest,
            _cancellation: Cancellation,
        ) -> RLMChatCompletion:
            current = threading.get_ident()
            with self._subcall_threads_lock:
                self._subcall_threads.add(current)
            try:
                return subcall_fn(request.prompt, request.model)
            finally:
                with self._subcall_threads_lock:
                    self._subcall_threads.discard(current)

        execution = ChildExecution.from_completion_callback(
            run_legacy,
            settle=lambda _outcome: None,
            max_concurrent=self.max_concurrent_subcalls,
        )
        self._owned_child_executions.append(execution)
        return execution

    def _spawn_child_execution(
        self,
        runner: Callable[[ChildRequest, Cancellation], RLMChatCompletion] | None,
    ) -> ChildExecution | None:
        if runner is None:
            return self._recursive_children

        def run_custom(
            request: ChildRequest,
            cancellation: Cancellation,
        ) -> RLMChatCompletion:
            current = threading.get_ident()
            with self._subcall_threads_lock:
                self._subcall_threads.add(current)
            try:
                return runner(request, cancellation)
            finally:
                with self._subcall_threads_lock:
                    self._subcall_threads.discard(current)

        if self._recursive_children is not None:
            execution = self._recursive_children.with_completion_adapter(
                run_custom,
                settle=self._settle_custom_child,
            )
        else:
            execution = ChildExecution.from_completion_callback(
                run_custom,
                settle=self._settle_custom_child,
                max_concurrent=self.max_concurrent_subcalls,
            )
        self._owned_child_executions.append(execution)
        return execution

    def _settle_custom_child(self, outcome: ChildOutcome) -> None:
        if not isinstance(outcome.usage, UsageSummary):
            raise TypeError("integrated custom child must report a UsageSummary")
        with self._accounting_lock:
            self._pending_accounting = UsageSummary.aggregate(
                [self._pending_accounting, outcome.usage]
            )

    def setup(self) -> None:
        """Create the selected one-shot kernel adapter."""
        with self._lifecycle:
            if self._lifecycle_state != "open":
                raise RuntimeError("IPythonREPL is closed")
            if self._kernel is not None:
                return
            if self.kernel_mode == "in_process":
                session = InProcessKernelSession(
                    working_dir=self.working_dir,
                    original_cwd=self.original_cwd,
                    lm_handler_address=lambda: self.lm_handler_address,
                    depth=self.depth,
                    child_execution=self._recursive_children,
                    custom_tools=self.custom_tools,
                )
                self._kernel = session
                self._public_locals_snapshot = session.namespace_snapshot
                return

            coordinator = QueryCoordinator(
                lm_handler_address=lambda: self.lm_handler_address,
                depth=self.depth,
                child_execution=(
                    self._recursive_children.run if self._recursive_children is not None else None
                ),
            )
            session = SubprocessKernelSession(
                working_dir=self.working_dir,
                startup_timeout=self.startup_timeout,
                subcall_timeout=self.subcall_timeout,
                custom_tools=self.custom_tools,
                output_callback=self.output_callback,
                child_execution=self._spawn_children,
                query_runner=coordinator.run,
                recursive_queries=coordinator.has_recursive_queries,
                max_concurrent_subcalls=self.max_concurrent_subcalls,
            )
            self._kernel = session
            self._public_locals_snapshot = session.restart_state_snapshot

    def load_context(self, context_payload: dict | list | str) -> None:
        self.add_context(context_payload, 0)

    def add_context(
        self,
        context_payload: dict | list | str,
        context_index: int | None = None,
    ) -> int:
        with self._lock:
            if context_index is None:
                context_index = self._context_count
            kernel = self._require_available_kernel()
            try:
                kernel.assign(
                    f"context_{context_index}",
                    context_payload,
                    alias="context" if context_index == 0 else None,
                    timeout=self.cell_timeout,
                )
            except TypeError:
                raise
            except Exception as error:
                raise RuntimeError(f"Failed to load context: {error}") from error
            self._context_count = max(self._context_count, context_index + 1)
            return context_index

    def get_context_count(self) -> int:
        return self._context_count

    def add_history(
        self,
        message_history: list[dict[str, Any]],
        history_index: int | None = None,
    ) -> int:
        with self._lock:
            if history_index is None:
                history_index = self._history_count
            kernel = self._require_available_kernel()
            try:
                kernel.assign(
                    f"history_{history_index}",
                    message_history,
                    alias="history" if history_index == 0 else None,
                    timeout=self.cell_timeout,
                )
            except Exception as error:
                raise RuntimeError(f"Failed to load history: {error}") from error
            self._history_count = max(self._history_count, history_index + 1)
            return history_index

    def get_history_count(self) -> int:
        return self._history_count

    def update_handler_address(self, address: tuple[str, int]) -> None:
        with self._lock:
            self._require_available_kernel()
            self.lm_handler_address = address

    def execute_code(self, code: str) -> REPLResult:
        current = threading.get_ident()
        with self._subcall_threads_lock:
            in_subcall = current in self._subcall_threads
            already_executing = current in self._executing_threads
        if in_subcall:
            raise RuntimeError(
                "Reentrant execute_code on the same instance from inside subcall_fn is not "
                "supported — it would deadlock or clobber the parent cell's bookkeeping. "
                "subcall_fn must spawn a child REPL with its own lock."
            )
        if already_executing:
            raise RuntimeError(
                "Reentrant execute_code on the same instance is not supported. Use a separate "
                "REPL instance for nested execution."
            )
        with self._subcall_threads_lock:
            self._executing_threads.add(current)
        try:
            with self._lock:
                return self._require_available_kernel().execute(code, timeout=self.cell_timeout)
        finally:
            with self._subcall_threads_lock:
                self._executing_threads.discard(current)

    def finalize_completion(self) -> CompletionFinalization:
        """Join delivery and child execution, then drain custom child usage."""
        with self._lock:
            with self._lifecycle:
                kernel = self._require_available_kernel()
                self._finalizing = True
            errors: list[BaseException] = []
            try:
                try:
                    kernel.finalize_completion()
                except BaseException as error:
                    errors.append(error)
                for execution in self._child_executions:
                    try:
                        execution.finalize()
                    except BaseException as error:
                        errors.append(error)
            finally:
                with self._accounting_lock:
                    usage = self._pending_accounting
                    self._pending_accounting = UsageSummary.empty()
                with self._lifecycle:
                    self._finalizing = False
                    self._lifecycle.notify_all()
        if len(errors) == 1:
            error: BaseException | None = errors[0]
        elif errors:
            error = BaseExceptionGroup("IPython completion finalization failed", errors)
        else:
            error = None
        return CompletionFinalization(usage, error)

    @property
    def locals(self) -> dict[str, Any]:
        """Return the legacy mode-specific parent-visible state.

        In-process mode returns a live namespace snapshot. Subprocess mode
        returns only host-assigned restart state, not the kernel namespace.
        """
        with self._lock:
            if self._lifecycle_state != "open" or self._public_locals_snapshot is None:
                return {}
            self._require_available_kernel()
            return self._public_locals_snapshot()

    def _require_open_kernel(self) -> KernelSession:
        kernel = self._kernel
        if self._lifecycle_state != "open" or kernel is None:
            raise RuntimeError("IPythonREPL is closed")
        return kernel

    def _require_available_kernel(self) -> KernelSession:
        kernel = self._require_open_kernel()
        if self._finalizing:
            raise RuntimeError("IPythonREPL completion is being finalized")
        return kernel

    def cleanup(self) -> None:
        current = threading.get_ident()
        thread_lock = getattr(self, "_subcall_threads_lock", None)
        if thread_lock is not None:
            with thread_lock:
                if current in self._executing_threads or current in self._subcall_threads:
                    raise RuntimeError(
                        "cleanup cannot run from an active IPython cell or child callback"
                    )
        lifecycle = getattr(self, "_lifecycle", None)
        if lifecycle is None:
            return
        with lifecycle:
            while self._lifecycle_state == "closing" or self._finalizing:
                lifecycle.wait()
            if self._lifecycle_state == "closed":
                return
            self._lifecycle_state = "closing"
            kernel = self._kernel
            self._kernel = None
            self._public_locals_snapshot = None

        cleanup_errors: list[BaseException] = []
        try:
            if kernel is not None:
                kernel.close()
        except BaseException as error:
            cleanup_errors.append(error)
        for execution in getattr(self, "_owned_child_executions", []):
            try:
                execution.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if getattr(self, "_owns_working_dir", False):
            try:
                shutil.rmtree(self.working_dir)
            except BaseException as error:
                cleanup_errors.append(error)
        with lifecycle:
            self._lifecycle_state = "closed"
            lifecycle.notify_all()

        if len(cleanup_errors) == 1:
            error = cleanup_errors[0]
            raise error.with_traceback(error.__traceback__)
        if cleanup_errors:
            raise BaseExceptionGroup("IPythonREPL cleanup failed", cleanup_errors)

    def __enter__(self) -> IPythonREPL:
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> bool:
        self.cleanup()
        return False

    def __del__(self) -> None:
        try:
            self.cleanup()
        except BaseException:
            pass
