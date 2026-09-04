"""Canonical child execution, lifecycle, admission, and settlement."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from rlm.core.types import JSONValue, RLMChatCompletion, UsageSummary, canonical_json_value

ChildStatus = Literal["ok", "error", "cancelled", "timeout"]


class Cancellation(Protocol):
    """Cooperative cancellation observed by one child execution."""

    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


class ChildCancelledError(RuntimeError):
    """A child was cancelled before it produced a deliverable outcome."""


class ChildContractError(TypeError):
    """A child runtime violated the execution interface."""


@dataclass(frozen=True, slots=True)
class ChildRequest:
    """Execution intent for one fresh child, without handle-delivery state."""

    task: str
    context: str | None = None
    model: str | None = None
    working_dir: str | None = None
    execution_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task:
            raise TypeError("child task must be a non-empty string")
        if self.context is not None and not isinstance(self.context, str):
            raise TypeError("child context must be a string or None")
        if self.model is not None and not isinstance(self.model, str):
            raise TypeError("child model must be a string or None")
        if self.working_dir is not None and not isinstance(self.working_dir, str):
            raise TypeError("child working_dir must be a string or None")
        if self.execution_id is not None and not isinstance(self.execution_id, str):
            raise TypeError("child execution_id must be a string or None")

    @property
    def prompt(self) -> str:
        if self.context is None:
            return self.task
        return f"{self.task}\n\n<context>\n{self.context}\n</context>"


@dataclass(frozen=True, slots=True, init=False)
class ChildOutcome:
    """One authoritative child outcome published after settlement and teardown.

    Integrated RLM children carry a ``UsageSummary`` and completion metadata.
    Standalone runtime adapters may carry another canonical JSON usage object.
    Wire results are projected from this value rather than supplied alongside it.
    """

    status: ChildStatus
    text: str | None
    error: str | None
    usage: UsageSummary | dict[str, JSONValue]
    elapsed_ms: int
    truncated: bool
    completion: RLMChatCompletion | None
    failures: tuple[BaseException, ...]

    def __init__(
        self,
        *,
        status: ChildStatus,
        text: str | None,
        error: str | None,
        usage: UsageSummary | dict[str, JSONValue],
        elapsed_ms: int,
        truncated: bool,
        completion: RLMChatCompletion | None = None,
        failures: tuple[BaseException, ...] = (),
    ) -> None:
        if status not in ("ok", "error", "cancelled", "timeout"):
            raise ValueError(f"invalid child status: {status!r}")
        if text is not None and not isinstance(text, str):
            raise TypeError("child text must be a string or None")
        if error is not None and not isinstance(error, str):
            raise TypeError("child error must be a string or None")
        if isinstance(usage, UsageSummary):
            stored_usage: UsageSummary | dict[str, JSONValue] = usage
        else:
            canonical = canonical_json_value(usage, label="child usage")
            if not isinstance(canonical, dict):
                raise TypeError("child usage must be a string-keyed object")
            stored_usage = json.loads(
                json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
            )
        if type(elapsed_ms) is not int or elapsed_ms < 0:
            raise TypeError("child elapsed_ms must be a non-negative integer")
        if type(truncated) is not bool:
            raise TypeError("child truncated must be a boolean")
        if completion is not None and not isinstance(completion, RLMChatCompletion):
            raise TypeError("child completion must be an RLMChatCompletion or None")
        if any(not isinstance(failure, BaseException) for failure in failures):
            raise TypeError("child failures must contain BaseException values")
        if status == "ok":
            if text is None or error is not None or failures:
                raise ValueError("a successful child must contain only result text")
        elif error is None or not error:
            raise ValueError("an unsuccessful child must carry an error")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "usage", stored_usage)
        object.__setattr__(self, "elapsed_ms", elapsed_ms)
        object.__setattr__(self, "truncated", truncated)
        object.__setattr__(self, "completion", completion)
        object.__setattr__(self, "failures", failures)

    @classmethod
    def from_completion(
        cls,
        completion: RLMChatCompletion,
        usage: UsageSummary,
        *,
        elapsed_ms: int,
    ) -> ChildOutcome:
        if not isinstance(completion, RLMChatCompletion):
            raise TypeError("child runtime must return an RLMChatCompletion")
        if not isinstance(usage, UsageSummary):
            raise TypeError("child runtime paid usage must be a UsageSummary")
        if completion.usage_summary != usage:
            failure = ChildContractError(
                "child runtime paid usage must match its completion usage summary"
            )
            return cls.failed(
                failure,
                usage,
                elapsed_ms=elapsed_ms,
                completion=completion,
            )
        if completion.error:
            return cls(
                status="error",
                text=None,
                error=completion.error,
                usage=usage,
                elapsed_ms=elapsed_ms,
                truncated=False,
                completion=completion,
            )
        return cls(
            status="ok",
            text=completion.response,
            error=None,
            usage=usage,
            elapsed_ms=elapsed_ms,
            truncated=False,
            completion=completion,
        )

    @classmethod
    def failed(
        cls,
        failure: BaseException,
        usage: UsageSummary | dict[str, JSONValue],
        *,
        elapsed_ms: int,
        completion: RLMChatCompletion | None = None,
        cancelled: bool = False,
    ) -> ChildOutcome:
        detail = str(failure)
        message = f"{type(failure).__name__}: {detail}" if detail else type(failure).__name__
        return cls(
            status="cancelled" if cancelled else "error",
            text=None,
            error=message,
            usage=usage,
            elapsed_ms=elapsed_ms,
            truncated=False,
            completion=completion,
            failures=(failure,),
        )

    @classmethod
    def external(
        cls,
        *,
        status: ChildStatus,
        text: str | None,
        error: str | None,
        usage: dict[str, JSONValue],
        elapsed_ms: int,
        truncated: bool,
    ) -> ChildOutcome:
        return cls(
            status=status,
            text=text,
            error=error,
            usage=usage,
            elapsed_ms=elapsed_ms,
            truncated=truncated,
        )

    @property
    def usage_json(self) -> dict[str, JSONValue]:
        if isinstance(self.usage, UsageSummary):
            return self.usage.to_dict()
        return json.loads(json.dumps(self.usage, ensure_ascii=False, separators=(",", ":")))

    def with_failure(
        self,
        failure: BaseException,
        *,
        elapsed_ms: int,
        cancelled: bool = False,
    ) -> ChildOutcome:
        failures = (*self.failures, failure)
        first = failures[0]
        detail = str(first)
        message = f"{type(first).__name__}: {detail}" if detail else type(first).__name__
        return replace(
            self,
            status="cancelled" if cancelled else "error",
            text=None,
            error=message,
            elapsed_ms=elapsed_ms,
            failures=failures,
        )

    def unwrap_completion(self) -> RLMChatCompletion:
        if self.failures:
            failure = self.failures[0]
            raise failure.with_traceback(failure.__traceback__)
        if self.completion is None:
            raise ChildContractError("child outcome has no RLM completion metadata")
        return self.completion


class ChildRuntime(Protocol):
    """Private runtime seam opened for one admitted child request."""

    def execute(self, cancellation: Cancellation) -> RLMChatCompletion | ChildOutcome: ...

    def paid_usage(self) -> UsageSummary: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class PreparedChild:
    """Callable child runtime used by the internal preparation seam."""

    execute_child: Callable[[Cancellation], RLMChatCompletion]
    usage_source: Callable[[], UsageSummary]
    teardown_child: Callable[[], None]

    def execute(self, cancellation: Cancellation) -> RLMChatCompletion:
        return self.execute_child(cancellation)

    def paid_usage(self) -> UsageSummary:
        return self.usage_source()

    def close(self) -> None:
        self.teardown_child()


ChildRuntimeFactory = Callable[[ChildRequest, Cancellation], ChildRuntime]
ChildSettlement = Callable[[ChildOutcome], None]


class _CombinedCancellation:
    def __init__(self, caller: Cancellation | None, lifecycle: threading.Event) -> None:
        self._caller = caller
        self._lifecycle = lifecycle

    def is_set(self) -> bool:
        return self._lifecycle.is_set() or (self._caller is not None and self._caller.is_set())

    def wait(self, timeout: float | None = None) -> bool:
        if self.is_set():
            return True
        if timeout is None:
            while not self.is_set():
                self._lifecycle.wait(0.05)
            return True
        deadline = time.monotonic() + timeout
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._lifecycle.wait(min(0.05, remaining))
        return True


class _CompletionCallbackRuntime:
    def __init__(
        self,
        request: ChildRequest,
        callback: Callable[[ChildRequest, Cancellation], RLMChatCompletion],
    ) -> None:
        self._request = request
        self._callback = callback
        self._completion: RLMChatCompletion | None = None

    def execute(self, cancellation: Cancellation) -> RLMChatCompletion:
        completion = self._callback(self._request, cancellation)
        if not isinstance(completion, RLMChatCompletion):
            raise ChildContractError("child callback must return an RLMChatCompletion")
        self._completion = completion
        return completion

    def paid_usage(self) -> UsageSummary:
        if self._completion is None:
            return UsageSummary.empty()
        return self._completion.usage_summary

    def close(self) -> None:
        return None


class _OutcomeCallbackRuntime:
    def __init__(
        self,
        request: ChildRequest,
        callback: Callable[[ChildRequest, Cancellation], ChildOutcome],
    ) -> None:
        self._request = request
        self._callback = callback

    def execute(self, cancellation: Cancellation) -> ChildOutcome:
        outcome = self._callback(self._request, cancellation)
        if not isinstance(outcome, ChildOutcome):
            raise ChildContractError("child callback must return a ChildOutcome")
        return outcome

    def paid_usage(self) -> UsageSummary:
        return UsageSummary.empty()

    def close(self) -> None:
        return None


class ChildExecution:
    """Own admission, cancellation, lifecycle ordering, and settlement for children."""

    def __init__(
        self,
        runtime_factory: ChildRuntimeFactory,
        *,
        settle: ChildSettlement,
        max_concurrent: int,
        thread_name_prefix: str = "rlm-subcall",
        _admission: threading.BoundedSemaphore | None = None,
    ) -> None:
        if not callable(runtime_factory):
            raise TypeError("child runtime factory must be callable")
        if not callable(settle):
            raise TypeError("child settlement must be callable")
        if max_concurrent <= 0:
            raise ValueError("child concurrency must be positive")
        self._runtime_factory = runtime_factory
        self._settle = settle
        self._max_concurrent = max_concurrent
        self._thread_name_prefix = thread_name_prefix
        self._admission = _admission or threading.BoundedSemaphore(max_concurrent)
        self._condition = threading.Condition()
        self._cancellation = threading.Event()
        self._executor: ThreadPoolExecutor | None = None
        self._futures: set[Future[ChildOutcome]] = set()
        self._failures: list[BaseException] = []
        self._inflight = 0
        self._finalizing = False
        self._closed = False

    @classmethod
    def from_completion_callback(
        cls,
        callback: Callable[[ChildRequest, Cancellation], RLMChatCompletion],
        *,
        settle: ChildSettlement,
        max_concurrent: int,
    ) -> ChildExecution:
        return cls(
            lambda request, _cancellation: _CompletionCallbackRuntime(request, callback),
            settle=settle,
            max_concurrent=max_concurrent,
        )

    @classmethod
    def from_outcome_callback(
        cls,
        callback: Callable[[ChildRequest, Cancellation], ChildOutcome],
        *,
        settle: ChildSettlement | None = None,
        max_concurrent: int,
    ) -> ChildExecution:
        return cls(
            lambda request, _cancellation: _OutcomeCallbackRuntime(request, callback),
            settle=settle if settle is not None else (lambda _outcome: None),
            max_concurrent=max_concurrent,
        )

    def with_completion_adapter(
        self,
        callback: Callable[[ChildRequest, Cancellation], RLMChatCompletion],
        *,
        settle: ChildSettlement,
    ) -> ChildExecution:
        """Create another execution adapter under the same admission limit."""
        return ChildExecution(
            lambda request, _cancellation: _CompletionCallbackRuntime(request, callback),
            settle=settle,
            max_concurrent=self._max_concurrent,
            thread_name_prefix=self._thread_name_prefix,
            _admission=self._admission,
        )

    def __call__(self, prompt: str, model: str | None = None) -> RLMChatCompletion:
        return self.run(ChildRequest(task=prompt, model=model)).unwrap_completion()

    def run(
        self,
        request: ChildRequest,
        cancellation: Cancellation | None = None,
    ) -> ChildOutcome:
        if not isinstance(request, ChildRequest):
            raise TypeError("child execution requires a ChildRequest")
        with self._condition:
            lifecycle = self._accept_locked()
        return self._run_accepted(request, cancellation, lifecycle)

    def submit(
        self,
        request: ChildRequest,
        cancellation: Cancellation | None = None,
    ) -> Future[ChildOutcome]:
        if not isinstance(request, ChildRequest):
            raise TypeError("child execution requires a ChildRequest")
        with self._condition:
            lifecycle = self._accept_locked()
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_concurrent,
                    thread_name_prefix=self._thread_name_prefix,
                )
            try:
                future = self._executor.submit(
                    self._run_accepted,
                    request,
                    cancellation,
                    lifecycle,
                )
            except BaseException:
                self._inflight -= 1
                self._condition.notify_all()
                raise
            self._futures.add(future)
        future.add_done_callback(self._future_finished)
        return future

    def _accept_locked(self) -> threading.Event:
        if self._closed:
            raise RuntimeError("child execution is closed")
        if self._finalizing:
            raise RuntimeError("child execution is being finalized")
        self._inflight += 1
        return self._cancellation

    def _run_accepted(
        self,
        request: ChildRequest,
        cancellation: Cancellation | None,
        lifecycle: threading.Event,
    ) -> ChildOutcome:
        combined = _CombinedCancellation(cancellation, lifecycle)
        admitted = False
        try:
            while not combined.is_set():
                if self._admission.acquire(timeout=0.05):
                    admitted = True
                    break
            if not admitted or combined.is_set():
                if admitted:
                    self._admission.release()
                    admitted = False
                outcome = self._settled_cancelled()
            else:
                outcome = self._run_admitted(request, combined)
            self._record_failures(outcome)
            return outcome
        finally:
            if admitted:
                self._admission.release()
            with self._condition:
                self._inflight -= 1
                self._condition.notify_all()

    def _future_finished(self, future: Future[ChildOutcome]) -> None:
        cancelled = future.cancelled()
        if cancelled:
            outcome = self._settled_cancelled()
            self._record_failures(outcome)
        with self._condition:
            self._futures.discard(future)
            if cancelled:
                self._inflight -= 1
            self._condition.notify_all()

    def _record_failures(self, outcome: ChildOutcome) -> None:
        failures = [
            failure for failure in outcome.failures if not isinstance(failure, ChildCancelledError)
        ]
        if not failures:
            return
        with self._condition:
            self._failures.extend(failures)

    def _settled_cancelled(self) -> ChildOutcome:
        failure = ChildCancelledError("child cancelled before admission")
        outcome = ChildOutcome.failed(
            failure,
            UsageSummary.empty(),
            elapsed_ms=0,
            cancelled=True,
        )
        try:
            self._settle(outcome)
        except BaseException as error:
            outcome = outcome.with_failure(error, elapsed_ms=0, cancelled=True)
        return outcome

    def _run_admitted(
        self,
        request: ChildRequest,
        cancellation: Cancellation,
    ) -> ChildOutcome:
        started = time.monotonic()
        runtime: ChildRuntime | None = None
        try:
            runtime = self._runtime_factory(request, cancellation)
            if not all(
                callable(getattr(runtime, name, None))
                for name in ("execute", "paid_usage", "close")
            ):
                raise ChildContractError(
                    "child runtime must implement execute, paid_usage, and close"
                )
        except BaseException as error:
            outcome = ChildOutcome.failed(
                error,
                UsageSummary.empty(),
                elapsed_ms=round((time.monotonic() - started) * 1000),
                cancelled=cancellation.is_set(),
            )
            try:
                self._settle(outcome)
            except BaseException as settlement_error:
                outcome = outcome.with_failure(
                    settlement_error,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                    cancelled=cancellation.is_set(),
                )
            return outcome

        try:
            try:
                value = runtime.execute(cancellation)
            except BaseException as error:
                cancelled = cancellation.is_set()
                failure = (
                    ChildCancelledError("child cancelled during execution") if cancelled else error
                )
                try:
                    usage = runtime.paid_usage()
                    if not isinstance(usage, UsageSummary):
                        raise ChildContractError(
                            "child runtime paid_usage must return a UsageSummary"
                        )
                except BaseException as usage_error:
                    usage = UsageSummary.empty()
                    outcome = ChildOutcome.failed(
                        failure,
                        usage,
                        elapsed_ms=round((time.monotonic() - started) * 1000),
                        cancelled=cancelled,
                    ).with_failure(
                        usage_error,
                        elapsed_ms=round((time.monotonic() - started) * 1000),
                        cancelled=cancellation.is_set(),
                    )
                else:
                    outcome = ChildOutcome.failed(
                        failure,
                        usage,
                        elapsed_ms=round((time.monotonic() - started) * 1000),
                        cancelled=cancelled,
                    )
            else:
                if isinstance(value, ChildOutcome):
                    outcome = value
                elif isinstance(value, RLMChatCompletion):
                    try:
                        usage = runtime.paid_usage()
                        if not isinstance(usage, UsageSummary):
                            raise ChildContractError(
                                "child runtime paid_usage must return a UsageSummary"
                            )
                    except BaseException as usage_error:
                        outcome = ChildOutcome.failed(
                            usage_error,
                            value.usage_summary,
                            elapsed_ms=round((time.monotonic() - started) * 1000),
                            completion=value,
                        )
                    else:
                        outcome = ChildOutcome.from_completion(
                            value,
                            usage,
                            elapsed_ms=round((time.monotonic() - started) * 1000),
                        )
                else:
                    error = ChildContractError(
                        "child runtime execute must return RLMChatCompletion or ChildOutcome"
                    )
                    try:
                        usage = runtime.paid_usage()
                    except BaseException:
                        usage = UsageSummary.empty()
                    outcome = ChildOutcome.failed(
                        error,
                        usage,
                        elapsed_ms=round((time.monotonic() - started) * 1000),
                    )

            if cancellation.is_set() and outcome.status != "cancelled":
                outcome = outcome.with_failure(
                    ChildCancelledError("child cancelled during execution"),
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                    cancelled=True,
                )

            try:
                runtime.close()
            except BaseException as error:
                outcome = outcome.with_failure(
                    error,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                    cancelled=cancellation.is_set(),
                )

            try:
                self._settle(outcome)
            except BaseException as error:
                outcome = outcome.with_failure(
                    error,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                    cancelled=cancellation.is_set(),
                )
            return outcome
        finally:
            runtime = None

    def finalize(self) -> None:
        """Cancel and join current work, then reopen for a later completion."""
        self._finalize(close=False)

    def close(self) -> None:
        """Cancel and join current work, then reject future child requests."""
        self._finalize(close=True)

    def _finalize(self, *, close: bool) -> None:
        with self._condition:
            if self._closed:
                return
            while self._finalizing:
                self._condition.wait()
                if self._closed:
                    return
            self._finalizing = True
            self._cancellation.set()
            futures = list(self._futures)
            executor = self._executor
        if futures:
            wait(futures)
        if executor is not None:
            executor.shutdown(wait=True)
        with self._condition:
            self._condition.wait_for(lambda: self._inflight == 0)
            failures = self._failures
            self._failures = []
            self._executor = None
            self._futures.clear()
            self._closed = close
            if not close:
                self._cancellation = threading.Event()
            self._finalizing = False
            self._condition.notify_all()
        if len(failures) == 1:
            failure = failures[0]
            raise failure.with_traceback(failure.__traceback__)
        if failures:
            raise BaseExceptionGroup("child execution finalization failed", failures)
