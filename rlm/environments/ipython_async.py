"""Authenticated execution and per-handle delivery state for subprocess IPython."""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import CancelledError as FutureCancelled
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures import wait as wait_for_futures
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from rlm.core.child_execution import ChildExecution, ChildOutcome, ChildRequest
from rlm.core.types import FinalValue, JSONValue, RLMChatCompletion
from rlm.environments.ipython_client import RLMAnswerDict, RLMClient, RLMHandle
from rlm.environments.ipython_kernel import IPythonOutputStream
from rlm.environments.ipython_protocol import (
    UNIT_RESULT,
    ChildResult,
    FailureCode,
    FinalRequest,
    GatherRequest,
    GatherResponse,
    HostRequest,
    HostResponse,
    QueryKind,
    QueryRequest,
    QueryResponse,
    ReleaseRequest,
    RLMHostError,
    SpawnRequest,
    UnitResult,
    ValidateRequest,
)
from rlm.environments.ipython_queries import QueryOutcome
from rlm.environments.ipython_transport import (
    DEFAULT_MAX_MESSAGE_BYTES,
    FramedProtocolServer,
    ProtocolConnection,
)

__all__ = [
    "AsyncRLMHost",
    "ChildExecution",
    "ChildOutcome",
    "ChildRequest",
    "ChildResult",
    "ExecutionSummary",
    "FailureCode",
    "HostExecution",
    "IPythonOutputStream",
    "QueryBatchRunner",
    "QueryOutcome",
    "RLMAnswerDict",
    "RLMClient",
    "RLMHandle",
    "RLMHostError",
]

_DEFAULT_MAX_REQUEST_BYTES = 1024 * 1024
_DEFAULT_MAX_LIVE_HANDLES = 16
_RELEASE_REPLAY_CACHE_SIZE = 1024


QueryBatchRunner = Callable[
    [QueryKind, list[str], str | None, threading.Event],
    list[QueryOutcome],
]
ActivityCallback = Callable[[str, str], None]
ReleaseCallback = Callable[[str], None]
_SubcallResult = TypeVar("_SubcallResult")


class _QueryScheduler:
    """Own bounded plain-query transport work and complete shutdown."""

    def __init__(self, max_concurrent: int, *, thread_name_prefix: str = "rlm-subcall") -> None:
        if max_concurrent <= 0:
            raise ValueError("subcall concurrency must be positive")
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix=thread_name_prefix,
        )
        self._lock = threading.Lock()
        self._futures: set[Future[Any]] = set()
        self._closed = False

    def submit(
        self,
        callback: Callable[..., _SubcallResult],
        *args: Any,
    ) -> Future[_SubcallResult]:
        with self._lock:
            if self._closed:
                raise RuntimeError("query scheduler is closed")
            future = self._executor.submit(callback, *args)
            self._futures.add(future)
        future.add_done_callback(self._finished)
        return future

    def _finished(self, future: Future[Any]) -> None:
        with self._lock:
            self._futures.discard(future)

    def wait(self) -> None:
        """Wait for current query work without closing the scheduler."""
        while True:
            with self._lock:
                futures = list(self._futures)
            if not futures:
                return
            wait_for_futures(futures)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = list(self._futures)
        for future in futures:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)


@dataclass(slots=True)
class ExecutionSummary:
    """State committed by one completed IPython execution."""

    final: FinalValue = field(default_factory=FinalValue.absent)
    completions: list[RLMChatCompletion] = field(default_factory=list)
    usages: list[JSONValue] = field(default_factory=list)
    spawned: int = 0
    gathered: int = 0

    def to_wire(self) -> dict[str, Any]:
        return {
            **self.final.to_dict(),
            "usages": list(self.usages),
            "spawned": self.spawned,
            "gathered": self.gathered,
        }


@dataclass(slots=True)
class _Execution:
    cancel: threading.Event = field(default_factory=threading.Event)
    final: FinalValue = field(default_factory=FinalValue.absent)
    completions: list[RLMChatCompletion] = field(default_factory=list)
    usages: list[JSONValue] = field(default_factory=list)
    spawned: int = 0
    gathered: int = 0
    query_futures: set[Future[Any]] = field(default_factory=set)


_Claim = tuple[str, str]


@dataclass(frozen=True, slots=True)
class _Running:
    claim: _Claim | None = None


@dataclass(frozen=True, slots=True)
class _Ready:
    outcome: ChildOutcome
    claim: _Claim | None = None


@dataclass(frozen=True, slots=True)
class _Delivered:
    outcome: ChildOutcome


@dataclass(frozen=True, slots=True)
class _Discarding:
    pass


_ChildState = _Running | _Ready | _Delivered | _Discarding


@dataclass(slots=True)
class _Child:
    """One stable handle whose tagged state owns every legal transition."""

    handle: str
    request: ChildRequest
    cancel: threading.Event
    future: Future[ChildOutcome]
    state: _ChildState = field(default_factory=_Running)


@dataclass(frozen=True, slots=True)
class _GatherClaim:
    request: GatherRequest
    execution: _Execution
    records: tuple[_Child, ...]
    claimed: tuple[_Child, ...]


class HostExecution:
    """Bind host and client execution ordering behind one public interface."""

    def __init__(self, host: AsyncRLMHost, client: RLMClient, execution_id: str) -> None:
        self.host = host
        self.client = client
        self.execution_id = execution_id
        self.summary: ExecutionSummary | None = None
        self._binding: Any = None
        self._entered = False

    def __enter__(self) -> HostExecution:
        if self._entered:
            raise RuntimeError("execution session cannot be entered twice")
        self.host.begin_execution(self.execution_id)
        try:
            self._binding = self.client.bind_execution(self.execution_id)
        except BaseException:
            self.host.end_execution(self.execution_id, successful=False)
            raise
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, _exc: Any, _tb: Any) -> bool:
        if not self._entered:
            raise RuntimeError("execution session has not started")
        try:
            self.summary = self.host.end_execution(
                self.execution_id,
                successful=exc_type is None,
            )
        finally:
            binding = self._binding
            self._binding = None
            if binding is not None:
                self.client.reset_execution(binding)
        return False


class AsyncRLMHost:
    """Own executions, stable child handles, callbacks, transport, and shutdown."""

    def __init__(
        self,
        child_execution: ChildExecution | None,
        *,
        query_runner: QueryBatchRunner | None = None,
        recursive_queries: bool = False,
        on_activity: ActivityCallback | None = None,
        on_release: ReleaseCallback | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        max_concurrent: int = _DEFAULT_MAX_LIVE_HANDLES,
        max_live_handles: int = _DEFAULT_MAX_LIVE_HANDLES,
        max_request_bytes: int = _DEFAULT_MAX_REQUEST_BYTES,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        if max_concurrent <= 0 or max_live_handles <= 0:
            raise ValueError("concurrency and live-handle limits must be positive")
        if child_execution is not None and not isinstance(child_execution, ChildExecution):
            raise TypeError("child_execution must be a ChildExecution or None")
        self.child_execution = child_execution
        self.query_runner = query_runner
        self.recursive_queries = recursive_queries
        self.on_activity = on_activity
        self.on_release = on_release
        self.max_live_handles = max_live_handles
        self.max_request_bytes = max_request_bytes
        self.auth_token = secrets.token_hex(32)
        self.query_scheduler = _QueryScheduler(
            max_concurrent,
            thread_name_prefix="rlm-query",
        )
        self._lock = threading.RLock()
        self._executions: dict[str, _Execution] = {}
        self._children: dict[str, _Child] = {}
        self._ended_execution_ids: set[str] = set()
        self._released_execution_ids: OrderedDict[str, None] = OrderedDict()
        self._shutting_down = False
        self._transport = FramedProtocolServer(
            self.auth_token,
            self._dispatch,
            host=host,
            port=port,
            max_message_bytes=max_message_bytes,
        )

    def start(self) -> tuple[str, int]:
        with self._lock:
            if self._shutting_down:
                raise RLMHostError("RLM host is stopped")
            return self._transport.start()

    @property
    def address(self) -> tuple[str, int]:
        return self._transport.address

    @property
    def live_handle_count(self) -> int:
        with self._lock:
            return len(self._children)

    def execution(self, client: RLMClient, execution_id: str) -> HostExecution:
        """Return a session that orders host begin and client activation correctly."""
        return HostExecution(self, client, execution_id)

    def begin_execution(self, execution_id: str) -> None:
        if not execution_id:
            raise ValueError("execution id must not be empty")
        with self._lock:
            if self._shutting_down:
                raise RLMHostError("RLM host is shutting down")
            if (
                execution_id in self._executions
                or execution_id in self._ended_execution_ids
                or execution_id in self._released_execution_ids
                or self._has_origin_handles_locked(execution_id)
            ):
                raise RLMHostError("duplicate execution id")
            self._executions[execution_id] = _Execution()

    def end_execution(self, execution_id: str, successful: bool) -> ExecutionSummary:
        query_cancellations: list[Future[Any]] = []
        with self._lock:
            execution = self._executions.pop(execution_id, None)
            if execution is None:
                raise RLMHostError(
                    "the originating IPython execution is no longer active",
                    FailureCode.EXECUTION_INACTIVE,
                )
            execution.cancel.set()
            query_cancellations.extend(execution.query_futures)
            for child in list(self._children.values()):
                claim = self._claim_of(child)
                if claim is not None and claim[1] == execution_id:
                    self._clear_claim(child, claim)
                if not successful and child.request.execution_id == execution_id:
                    self._discard_child_locked(child)
            self._ended_execution_ids.add(execution_id)
            release = self._mark_released_locked(execution_id)
            summary = ExecutionSummary(
                final=execution.final if successful else FinalValue.absent(),
                completions=list(execution.completions),
                usages=list(execution.usages),
                spawned=execution.spawned,
                gathered=execution.gathered,
            )
        for future in query_cancellations:
            future.cancel()
        if release:
            self._notify_release(execution_id)
        return summary

    def reset(self) -> None:
        query_cancellations: list[Future[Any]] = []
        with self._lock:
            execution_ids = set(self._executions)
            execution_ids.update(child.request.execution_id for child in self._children.values())
            for execution in self._executions.values():
                execution.cancel.set()
                query_cancellations.extend(execution.query_futures)
            for child in list(self._children.values()):
                self._discard_child_locked(child)
            self._executions.clear()
            self._ended_execution_ids.update(execution_ids)
            releasable = {
                execution_id
                for execution_id in execution_ids
                if self._mark_released_locked(execution_id)
            }
        for future in query_cancellations:
            future.cancel()
        for execution_id in releasable:
            self._notify_release(execution_id)

    def finalize_completion(self) -> None:
        """Cancel and join work that cannot outlive its owning RLM completion."""
        self.reset()
        self.query_scheduler.wait()
        while self.live_handle_count:
            time.sleep(0.001)

    def stop(self) -> None:
        with self._lock:
            if self._shutting_down:
                return
            self._shutting_down = True

        errors: list[BaseException] = []
        for operation in (
            self.reset,
            self._transport.stop,
            (self.child_execution.finalize if self.child_execution is not None else lambda: None),
            self.query_scheduler.shutdown,
        ):
            try:
                operation()
            except BaseException as error:
                errors.append(error)
        if len(errors) == 1:
            error = errors[0]
            raise error.with_traceback(error.__traceback__)
        if errors:
            raise BaseExceptionGroup("RLM host shutdown failed", errors)

    def _dispatch(self, request: HostRequest, connection: ProtocolConnection) -> HostResponse:
        """Dispatch exhaustively while retaining each operation's concrete type."""
        match request:
            case SpawnRequest():
                return self._spawn(request, connection)
            case GatherRequest():
                return self._gather(request, connection)
            case ReleaseRequest():
                return self._release_handles(request, connection)
            case QueryRequest():
                return self._query(request, connection)
            case FinalRequest():
                return self._final(request, connection)
            case ValidateRequest():
                return self._validate_execution(request, connection)
            case _:
                raise TypeError(f"unsupported host operation: {type(request).__name__}")

    def _validate_execution(
        self,
        request: ValidateRequest,
        _connection: ProtocolConnection,
    ) -> UnitResult:
        with self._lock:
            self._active_execution_locked(request.execution_id)
        return UNIT_RESULT

    def _spawn(self, request: SpawnRequest, _connection: ProtocolConnection) -> UnitResult:
        if self.child_execution is None:
            raise RLMHostError("rlm.spawn is unavailable because no child runner is configured")
        if not os.path.isabs(request.cwd) or not os.path.isdir(request.cwd):
            raise RLMHostError("spawn requires an accessible absolute cwd")
        request_bytes = len(request.task.encode()) + (
            len(request.context.encode()) if request.context else 0
        )
        if request_bytes > self.max_request_bytes:
            raise RLMHostError("child task and context exceed the size limit")
        child_request = ChildRequest(
            task=request.task,
            context=request.context,
            working_dir=request.cwd,
            execution_id=request.execution_id,
        )
        with self._lock:
            existing = self._children.get(request.request_id)
            if existing is not None:
                if existing.request != child_request:
                    raise RLMHostError("spawn idempotency key was reused with different input")
                return UNIT_RESULT
            execution = self._active_execution_locked(request.execution_id)
            if len(self._children) >= self.max_live_handles:
                raise RLMHostError("too many live child handles")
            cancel = threading.Event()
            future = self.child_execution.submit(child_request, cancel)
            child = _Child(request.request_id, child_request, cancel, future)
            self._children[request.request_id] = child
            execution.spawned += 1
            future.add_done_callback(lambda completed: self._child_finished(child, completed))
        return UNIT_RESULT

    def _discard_child_locked(self, child: _Child) -> None:
        """Detach queued delivery immediately and signal admitted work to stop."""
        child.cancel.set()
        child.state = _Discarding()
        if child.future.done() or not child.future.running():
            self._children.pop(child.handle, None)

    def _child_finished(self, child: _Child, future: Future[ChildOutcome]) -> None:
        release = False
        with self._lock:
            current = self._children.get(child.handle)
            if current is not child:
                return
            if isinstance(child.state, _Discarding):
                self._children.pop(child.handle, None)
                release = self._mark_released_locked(child.request.execution_id)
            elif isinstance(child.state, _Running):
                child.state = _Ready(future.result(), child.state.claim)
        if release:
            self._notify_release(child.request.execution_id)

    @staticmethod
    def _claim_of(child: _Child) -> _Claim | None:
        if isinstance(child.state, (_Running, _Ready)):
            return child.state.claim
        return None

    @staticmethod
    def _delivered_outcome(child: _Child) -> ChildOutcome | None:
        if isinstance(child.state, _Delivered):
            return child.state.outcome
        return None

    @staticmethod
    def _set_claim(child: _Child, claim: _Claim) -> None:
        if isinstance(child.state, _Running):
            child.state = _Running(claim)
        elif isinstance(child.state, _Ready):
            child.state = _Ready(child.state.outcome, claim)
        else:
            raise RLMHostError(f"child handle is not gatherable: {child.handle}")

    @staticmethod
    def _clear_claim(child: _Child, claim: _Claim | None = None) -> None:
        if isinstance(child.state, _Running) and (claim is None or child.state.claim == claim):
            child.state = _Running()
        elif isinstance(child.state, _Ready) and (claim is None or child.state.claim == claim):
            child.state = _Ready(child.state.outcome)

    @staticmethod
    def _wire_outcome(outcome: ChildOutcome) -> ChildResult:
        return ChildResult(
            status=outcome.status,
            text=outcome.text,
            error=outcome.error,
            usage=outcome.usage_json,
            elapsed_ms=outcome.elapsed_ms,
            truncated=outcome.truncated,
        )

    def _gather(
        self,
        request: GatherRequest,
        connection: ProtocolConnection,
    ) -> GatherResponse:
        self._validate_handle_count(request.handles)
        claim = self._claim_gather(request)
        try:
            outcomes = self._wait_for_gather(claim, connection)
            response = GatherResponse(
                tuple(self._wire_outcome(outcomes[handle]) for handle in request.handles)
            )
            connection.ensure_response_fits(request, response)
            self._commit_gather(claim, outcomes)
            return response
        finally:
            self._release_gather_claim(claim)

    def _claim_gather(self, request: GatherRequest) -> _GatherClaim:
        with self._lock:
            execution = self._active_execution_locked(request.execution_id)
            records: list[_Child] = []
            for handle in request.handles:
                child = self._children.get(handle)
                if child is None or isinstance(child.state, _Discarding):
                    raise RLMHostError(f"unknown or already released child handle: {handle}")
                if self._delivered_outcome(child) is None and self._claim_of(child) is not None:
                    raise RLMHostError(
                        f"child handle is already being gathered: {handle}",
                        FailureCode.GATHER_PENDING,
                    )
                records.append(child)
            unique = list({child.handle: child for child in records}.values())
            claimed = tuple(child for child in unique if self._delivered_outcome(child) is None)
            claim = (request.request_id, request.execution_id)
            for child in claimed:
                self._set_claim(child, claim)
            return _GatherClaim(request, execution, tuple(records), claimed)

    def _wait_for_gather(
        self,
        claim: _GatherClaim,
        connection: ProtocolConnection,
    ) -> dict[str, ChildOutcome]:
        if claim.claimed:
            suffix = "" if len(claim.claimed) == 1 else "ren"
            self._activity(
                claim.request.execution_id,
                f"Waiting for {len(claim.claimed)} RLM child{suffix}…",
            )
        outcomes: dict[str, ChildOutcome] = {}
        unique_records = {child.handle: child for child in claim.records}.values()
        for child in unique_records:
            delivered = self._delivered_outcome(child)
            if delivered is not None:
                outcomes[child.handle] = delivered
        remaining = list(claim.claimed)
        completed = 0
        while remaining:
            if claim.execution.cancel.is_set():
                raise RLMHostError(
                    "the originating IPython execution is no longer active",
                    FailureCode.EXECUTION_INACTIVE,
                )
            if connection.disconnected():
                raise RLMHostError("gather client disconnected")
            for child in list(remaining):
                ready = child.state.outcome if isinstance(child.state, _Ready) else None
                try:
                    outcome = ready if ready is not None else child.future.result(timeout=0)
                except FutureTimeout:
                    continue
                outcomes[child.handle] = outcome
                remaining.remove(child)
                completed += 1
                self._activity(
                    claim.request.execution_id,
                    f"RLM children completed: {completed}/{len(claim.claimed)}",
                )
            if remaining:
                claim.execution.cancel.wait(0.02)
        return outcomes

    def _commit_gather(
        self,
        claim: _GatherClaim,
        outcomes: dict[str, ChildOutcome],
    ) -> None:
        with self._lock:
            execution = self._active_execution_locked(claim.request.execution_id)
            if execution is not claim.execution:
                raise RLMHostError(
                    "the originating IPython execution is no longer active",
                    FailureCode.EXECUTION_INACTIVE,
                )
            owner = (claim.request.request_id, claim.request.execution_id)
            for child in claim.claimed:
                if self._claim_of(child) != owner:
                    raise RLMHostError("gather claim is no longer active")
            for child in claim.claimed:
                outcome = outcomes[child.handle]
                child.state = _Delivered(outcome)
                execution.gathered += 1
                execution.usages.append(outcome.usage_json)
                if outcome.completion is not None:
                    execution.completions.append(outcome.completion)

    def _release_gather_claim(self, claim: _GatherClaim) -> None:
        owner = (claim.request.request_id, claim.request.execution_id)
        with self._lock:
            for child in claim.claimed:
                self._clear_claim(child, owner)

    def _release_handles(
        self,
        request: ReleaseRequest,
        _connection: ProtocolConnection,
    ) -> UnitResult:
        self._validate_handle_count(request.handles)
        releases: set[str] = set()
        with self._lock:
            self._active_execution_locked(request.execution_id)
            children = [
                child
                for handle in dict.fromkeys(request.handles)
                if (child := self._children.get(handle)) is not None
            ]
            for child in children:
                if self._claim_of(child) is not None:
                    raise RLMHostError(f"child handle is being gathered: {child.handle}")
            for child in children:
                if isinstance(child.state, _Delivered):
                    self._children.pop(child.handle, None)
                else:
                    self._discard_child_locked(child)
                if self._mark_released_locked(child.request.execution_id):
                    releases.add(child.request.execution_id)
        for origin in releases:
            self._notify_release(origin)
        return UNIT_RESULT

    def _validate_handle_count(self, handles: tuple[str, ...]) -> None:
        if len(handles) > self.max_live_handles:
            raise RLMHostError("too many handles")

    def _query(self, request: QueryRequest, _connection: ProtocolConnection) -> QueryResponse:
        prompts = list(request.prompts)
        with self._lock:
            execution = self._active_execution_locked(request.execution_id)
            if not prompts:
                return QueryResponse(())
            if self.query_runner is None:
                raise RLMHostError(f"No {request.kind} query runner configured")
            recursive = request.kind == "rlm" and self.recursive_queries
            batches = [[prompt] for prompt in prompts] if recursive else [prompts]
            futures = [
                self.query_scheduler.submit(
                    self._run_query_batch,
                    request.execution_id,
                    execution,
                    self.query_runner,
                    request.kind,
                    batch,
                    request.model,
                )
                for batch in batches
            ]
            execution.query_futures.update(futures)
            for future in futures:
                future.add_done_callback(
                    lambda completed, owner=execution: self._query_finished(owner, completed)
                )

        try:
            batch_outcomes = [cast(list[QueryOutcome], future.result()) for future in futures]
        except FutureCancelled as error:
            raise RLMHostError(
                "the originating IPython execution is no longer active",
                FailureCode.EXECUTION_INACTIVE,
            ) from error
        outcomes = [outcome for batch in batch_outcomes for outcome in batch]

        with self._lock:
            current = self._active_execution_locked(request.execution_id)
            if current is not execution:
                raise RLMHostError(
                    "the originating IPython execution is no longer active",
                    FailureCode.EXECUTION_INACTIVE,
                )
            execution.completions.extend(
                outcome.completion for outcome in outcomes if outcome.completion is not None
            )
        return QueryResponse(tuple(outcome.result for outcome in outcomes))

    def _run_query_batch(
        self,
        execution_id: str,
        execution: _Execution,
        runner: QueryBatchRunner,
        kind: QueryKind,
        prompts: list[str],
        model: str | None,
    ) -> list[QueryOutcome]:
        with self._lock:
            if self._active_execution_locked(execution_id) is not execution:
                raise RLMHostError(
                    "the originating IPython execution is no longer active",
                    FailureCode.EXECUTION_INACTIVE,
                )
        try:
            outcomes = runner(kind, prompts, model, execution.cancel)
        except BaseException as error:
            detail = str(error)
            message = f"{type(error).__name__}: {detail}" if detail else type(error).__name__
            return [QueryOutcome.failure(message) for _ in prompts]
        if not isinstance(outcomes, list) or len(outcomes) != len(prompts):
            raise TypeError("query runner must return one outcome per prompt")
        if any(not isinstance(outcome, QueryOutcome) for outcome in outcomes):
            raise TypeError("query runner must return QueryOutcome values")
        return outcomes

    def _query_finished(self, execution: _Execution, future: Future[Any]) -> None:
        with self._lock:
            execution.query_futures.discard(future)

    def _final(self, request: FinalRequest, _connection: ProtocolConnection) -> UnitResult:
        final = FinalValue.of(request.value)
        encoded = json.dumps(final.value, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > self.max_request_bytes:
            raise RLMHostError("final value exceeds the size limit")
        with self._lock:
            execution = self._active_execution_locked(request.execution_id)
            if not execution.final.is_present:
                execution.final = final
        return UNIT_RESULT

    def _active_execution_locked(self, execution_id: str) -> _Execution:
        execution = self._executions.get(execution_id)
        if self._shutting_down or execution is None or execution.cancel.is_set():
            raise RLMHostError(
                "the originating IPython execution is no longer active",
                FailureCode.EXECUTION_INACTIVE,
            )
        return execution

    def _has_origin_handles_locked(self, execution_id: str) -> bool:
        return any(child.request.execution_id == execution_id for child in self._children.values())

    def _mark_released_locked(self, execution_id: str) -> bool:
        if (
            execution_id not in self._ended_execution_ids
            or execution_id in self._executions
            or self._has_origin_handles_locked(execution_id)
        ):
            return False
        self._ended_execution_ids.remove(execution_id)
        self._released_execution_ids[execution_id] = None
        if len(self._released_execution_ids) > _RELEASE_REPLAY_CACHE_SIZE:
            self._released_execution_ids.popitem(last=False)
        return True

    def _activity(self, execution_id: str, message: str) -> None:
        if self.on_activity is not None:
            try:
                self.on_activity(execution_id, message)
            except Exception:
                pass

    def _notify_release(self, execution_id: str) -> None:
        if self.on_release is not None:
            try:
                self.on_release(execution_id)
            except Exception:
                pass
