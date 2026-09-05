"""Execution-bound kernel client for async IPython RLM operations."""

from __future__ import annotations

import contextvars
import inspect
import os
import secrets
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from rlm.core.types import JSONValue
from rlm.environments.ipython_protocol import (
    ChildResult,
    FinalRequest,
    GatherRequest,
    QueryKind,
    QueryRequest,
    ReleaseRequest,
    RLMHostError,
    SpawnRequest,
    ValidateRequest,
    query_result_text,
)
from rlm.environments.ipython_transport import (
    DEFAULT_MAX_MESSAGE_BYTES,
    DEFAULT_REQUEST_TIMEOUT,
    HostChannel,
)


class _HandlePhase(Enum):
    AVAILABLE = "available"
    GATHERING = "gathering"
    RELEASING = "releasing"
    CONSUMED = "consumed"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class _GatherOwner:
    generation: int


@dataclass(frozen=True, slots=True)
class _ExecutionBinding:
    owner: object
    execution_id: contextvars.Token[str | None]
    generation: contextvars.Token[int]


class RLMHandle:
    """Opaque local handle with one cached delivery and one terminal disposition."""

    def __init__(self, client: RLMClient, handle_id: str) -> None:
        self._client = client
        self._id = handle_id
        self._phase = _HandlePhase.AVAILABLE
        self._owner: _GatherOwner | None = None
        self._result: ChildResult | None = None

    def __repr__(self) -> str:
        if self._phase is _HandlePhase.AVAILABLE:
            label = "recoverable" if self._result is not None else "pending"
        else:
            label = self._phase.value
        return f"<rlm.Handle {label}>"

    def claim_gather(self, owner: _GatherOwner) -> None:
        if self._phase is _HandlePhase.RELEASED:
            raise RLMHostError("rlm.gather received a released handle")
        if self._phase is _HandlePhase.CONSUMED:
            raise RLMHostError("rlm.gather received a consumed handle")
        if self._phase is _HandlePhase.RELEASING:
            raise RLMHostError("child handle is being released")
        if self._phase is _HandlePhase.GATHERING:
            assert self._owner is not None
            if self._owner.generation >= owner.generation:
                raise RLMHostError("child handle is already being gathered")
        self._phase = _HandlePhase.GATHERING
        self._owner = owner

    def ensure_owner(self, owner: _GatherOwner) -> None:
        if self._phase is not _HandlePhase.GATHERING or self._owner is not owner:
            raise RLMHostError("gather ownership moved to a later IPython execution")

    def cache_result(self, owner: _GatherOwner, result: ChildResult) -> None:
        self.ensure_owner(owner)
        if self._result is not None and self._result != result:
            raise RLMHostError("RLM host returned conflicting results for a child handle")
        self._result = result

    def finish_gather(self, owner: _GatherOwner) -> None:
        self.ensure_owner(owner)
        if self._result is None:
            raise AssertionError("gather completed without a child result")
        self._phase = _HandlePhase.CONSUMED
        self._owner = None
        self._result = None

    def rollback_gather(self, owner: _GatherOwner) -> None:
        if self._phase is _HandlePhase.GATHERING and self._owner is owner:
            self._phase = _HandlePhase.AVAILABLE
            self._owner = None

    def claim_release(self) -> bool:
        if self._phase is _HandlePhase.GATHERING:
            raise RLMHostError("cannot release a handle while it is being gathered")
        if self._phase is _HandlePhase.RELEASING:
            raise RLMHostError("child handle is already being released")
        if self._phase is not _HandlePhase.AVAILABLE:
            return False
        self._phase = _HandlePhase.RELEASING
        return True

    def finish_release(self) -> None:
        if self._phase is not _HandlePhase.RELEASING:
            raise AssertionError("release lost ownership of a child handle")
        self._phase = _HandlePhase.RELEASED
        self._result = None

    def rollback_release(self) -> None:
        if self._phase is _HandlePhase.RELEASING:
            self._phase = _HandlePhase.AVAILABLE

    async def release(self) -> None:
        """Release this handle and cancel unfinished child work."""
        await self._client.release([self])


class _GatherAttempt:
    """Atomic local ownership, delivery caching, cleanup, and single delivery."""

    def __init__(self, items: list[RLMHandle], generation: int) -> None:
        self.items = items
        self.unique_items = list({handle._id: handle for handle in items}.values())
        self.owner = _GatherOwner(generation)
        claimed: list[RLMHandle] = []
        try:
            for handle in self.unique_items:
                handle.claim_gather(self.owner)
                claimed.append(handle)
        except Exception:
            for handle in claimed:
                handle.rollback_gather(self.owner)
            raise

    def ensure_owner(self) -> None:
        for handle in self.unique_items:
            handle.ensure_owner(self.owner)

    async def deliver(self, client: RLMClient, execution_id: str) -> None:
        unresolved = [handle for handle in self.unique_items if handle._result is None]
        if not unresolved:
            return
        request = GatherRequest(
            secrets.token_hex(16),
            execution_id,
            tuple(handle._id for handle in unresolved),
        )
        response = await client.channel.request(request)
        self.ensure_owner()
        if len(response.results) != len(unresolved):
            raise RLMHostError("RLM host returned an invalid gather result")
        for handle, result in zip(unresolved, response.results, strict=True):
            handle.cache_result(self.owner, result)

    async def cleanup(self, client: RLMClient, execution_id: str) -> None:
        await client.channel.request(
            ReleaseRequest(
                secrets.token_hex(16),
                execution_id,
                tuple(handle._id for handle in self.unique_items),
            )
        )
        self.ensure_owner()

    def commit(self) -> list[ChildResult]:
        results = {
            handle._id: handle._result for handle in self.unique_items if handle._result is not None
        }
        if len(results) != len(self.unique_items):
            raise AssertionError("gather committed without every child result")
        delivered = [cast(ChildResult, results[handle._id]) for handle in self.items]
        for handle in self.unique_items:
            handle.finish_gather(self.owner)
        return delivered

    def rollback(self) -> None:
        for handle in self.unique_items:
            handle.rollback_gather(self.owner)


class RLMClient:
    """Execution-bound sync and async interface over one deep host channel."""

    def __init__(
        self,
        address: tuple[str, int],
        auth_token: str,
        *,
        timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        self.channel = HostChannel(
            address,
            auth_token,
            timeout=timeout,
            max_message_bytes=max_message_bytes,
        )
        self._pending_execution_id: str | None = None
        self._pending_execution_generation = 0
        self._pending_activation_owner: int | None = None
        self._binding_lock = threading.Lock()
        self._standalone_execution_generation = 0
        self._execution_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"rlm_execution_id_{id(self)}",
            default=None,
        )
        self._execution_generation: contextvars.ContextVar[int] = contextvars.ContextVar(
            f"rlm_execution_generation_{id(self)}",
            default=0,
        )
        self._hook_installed = False
        self._hook_shell: Any = None
        self._answer = RLMAnswerDict(self)

    @property
    def address(self) -> tuple[str, int]:
        return self.channel.address

    @property
    def auth_token(self) -> str:
        return self.channel.auth_token

    @property
    def timeout(self) -> float | None:
        return self.channel.timeout

    @timeout.setter
    def timeout(self, value: float | None) -> None:
        if value is not None and value <= 0:
            raise ValueError("timeout must be positive or None")
        self.channel.timeout = value

    @property
    def answer(self) -> RLMAnswerDict:
        return self._answer

    @staticmethod
    def _validate_execution_id(execution_id: str) -> None:
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError("execution id must be a non-empty string")

    def prepare_execution(self, execution_id: str) -> None:
        """Prepare one sequential IPython cell for its pre-cell activation hook.

        A single client cannot have overlapping prepare/activate pairs. Standalone
        callers that need nested or concurrent bindings use ``bind_execution``.
        """
        self._validate_execution_id(execution_id)
        current = threading.get_ident()
        with self._binding_lock:
            if self._pending_activation_owner is not None:
                raise RuntimeError("overlapping IPython execution preparation is not supported")
            self._pending_activation_owner = current
            if execution_id != self._pending_execution_id:
                self._pending_execution_id = execution_id
                self._pending_execution_generation += 1
                self._answer = RLMAnswerDict(self)

    def activate_execution(self, *_args: Any, **_kwargs: Any) -> None:
        current = threading.get_ident()
        with self._binding_lock:
            if self._pending_activation_owner not in (None, current):
                raise RuntimeError("IPython execution must be activated by its preparing thread")
            execution_id = self._pending_execution_id
            generation = self._pending_execution_generation
            self._pending_activation_owner = None
        if execution_id is not None:
            self._execution_id.set(execution_id)
            self._execution_generation.set(generation)

    def bind_execution(self, execution_id: str) -> _ExecutionBinding:
        """Bind one standalone execution until its returned tokens are reset."""
        self._validate_execution_id(execution_id)
        with self._binding_lock:
            self._standalone_execution_generation -= 1
            generation = self._standalone_execution_generation
        execution_token = self._execution_id.set(execution_id)
        try:
            generation_token = self._execution_generation.set(generation)
        except BaseException:
            self._execution_id.reset(execution_token)
            raise
        return _ExecutionBinding(self, execution_token, generation_token)

    def reset_execution(self, binding: _ExecutionBinding) -> None:
        """Restore the context that preceded ``bind_execution``."""
        if not isinstance(binding, _ExecutionBinding) or binding.owner is not self:
            raise TypeError("execution binding was not created by this RLM client")
        self._execution_generation.reset(binding.generation)
        self._execution_id.reset(binding.execution_id)

    def install_ipython(self, shell: Any) -> None:
        if self._hook_installed:
            if shell is not self._hook_shell:
                raise RuntimeError("RLMClient is already installed in another IPython shell")
            return
        shell.events.register("pre_run_cell", self.activate_execution)
        self._hook_shell = shell
        self._hook_installed = True

    def uninstall_ipython(self, shell: Any) -> None:
        """Remove this client's execution-activation hook."""
        if not self._hook_installed:
            return
        if shell is not self._hook_shell:
            raise RuntimeError("RLMClient is installed in another IPython shell")
        shell.events.unregister("pre_run_cell", self.activate_execution)
        self._hook_shell = None
        self._hook_installed = False

    async def spawn(self, task: str, *, context: str | None = None) -> RLMHandle:
        """Get or create one child under a client-generated stable handle ID."""
        if not isinstance(task, str):
            raise TypeError("rlm.spawn task must be a string")
        if context is not None and not isinstance(context, str):
            raise TypeError("rlm.spawn context must be a string or None")
        handle_id = secrets.token_hex(16)
        execution_id = self._current_execution()
        await self.channel.request(
            SpawnRequest(handle_id, execution_id, task, context, os.getcwd())
        )
        return RLMHandle(self, handle_id)

    async def gather(self, handles: Iterable[RLMHandle]) -> list[ChildResult]:
        """Deliver each handle once and recover committed values after response loss."""
        items = self._handles(handles, "gather")
        execution_id = self._current_execution()
        await self._validate_active_execution(execution_id)
        if not items:
            return []
        attempt = _GatherAttempt(items, self._execution_generation.get())
        try:
            await attempt.deliver(self, execution_id)
            await attempt.cleanup(self, execution_id)
            return attempt.commit()
        finally:
            attempt.rollback()

    async def release(self, handles: Iterable[RLMHandle]) -> None:
        """Idempotently release handles and discard any cached delivery."""
        items = self._handles(handles, "release")
        execution_id = self._current_execution()
        unique = list({handle._id: handle for handle in items}.values())
        claimed: list[RLMHandle] = []
        try:
            for handle in unique:
                if handle.claim_release():
                    claimed.append(handle)
            await self._validate_active_execution(execution_id)
            if claimed:
                await self.channel.request(
                    ReleaseRequest(
                        secrets.token_hex(16),
                        execution_id,
                        tuple(handle._id for handle in claimed),
                    )
                )
            for handle in claimed:
                handle.finish_release()
        finally:
            for handle in claimed:
                handle.rollback_release()

    def validate_execution(self) -> None:
        """Positively verify that the currently bound execution exists on the host."""
        self.channel.request_sync(ValidateRequest(secrets.token_hex(16), self._current_execution()))

    async def _validate_active_execution(self, execution_id: str) -> None:
        await self.channel.request(ValidateRequest(secrets.token_hex(16), execution_id))

    async def final(self, value: JSONValue | ChildResult) -> None:
        await self.channel.request(
            FinalRequest(
                secrets.token_hex(16),
                self._current_execution(),
                self._final_wire_value(value),
            )
        )

    def final_sync(self, value: JSONValue | ChildResult) -> None:
        self.channel.request_sync(
            FinalRequest(
                secrets.token_hex(16),
                self._current_execution(),
                self._final_wire_value(value),
            )
        )

    @classmethod
    def _final_wire_value(cls, value: Any) -> JSONValue:
        if isinstance(value, ChildResult):
            return value.to_wire()
        if isinstance(value, list):
            return [cls._final_wire_value(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._final_wire_value(item) for key, item in value.items()}
        return cast(JSONValue, value)

    def llm_query(self, prompt: str, model: str | None = None) -> str:
        return self._single_query("llm", prompt, model)

    def llm_query_batched(self, prompts: Iterable[str], model: str | None = None) -> list[str]:
        items = list(prompts)
        try:
            return self._query_sync("llm", items, model)
        except Exception as error:
            return [f"Error: LM query failed - {error}"] * len(items)

    def rlm_query(self, prompt: str, model: str | None = None) -> str:
        return self._single_query("rlm", prompt, model)

    def rlm_query_batched(self, prompts: Iterable[str], model: str | None = None) -> list[str]:
        items = list(prompts)
        try:
            return self._query_sync("rlm", items, model)
        except Exception as error:
            return [f"Error: RLM query failed - {error}"] * len(items)

    def _single_query(self, kind: QueryKind, prompt: str, model: str | None) -> str:
        try:
            return self._query_sync(kind, [prompt], model)[0]
        except Exception as error:
            label = "LM" if kind == "llm" else "RLM"
            return f"Error: {label} query failed - {error}"

    def _query_sync(self, kind: QueryKind, prompts: list[str], model: str | None) -> list[str]:
        response = self.channel.request_sync(
            QueryRequest(
                secrets.token_hex(16),
                self._current_execution(),
                kind,
                tuple(prompts),
                model,
            )
        )
        if len(response.results) != len(prompts):
            raise RLMHostError("RLM host returned malformed query results")
        return [query_result_text(kind, result) for result in response.results]

    def _handles(self, handles: Iterable[RLMHandle], operation: str) -> list[RLMHandle]:
        try:
            items = list(handles)
        except TypeError as error:
            raise TypeError(f"rlm.{operation} requires an iterable of rlm handles") from error
        for handle in items:
            if inspect.iscoroutine(handle):
                raise TypeError(
                    f"rlm.{operation} received a coroutine; use h = await rlm.spawn(...) first"
                )
            if not isinstance(handle, RLMHandle) or handle._client is not self:
                raise TypeError(f"rlm.{operation} received a handle from another RLM client")
        return items

    def _current_execution(self) -> str:
        execution_id = self._execution_id.get()
        generation = self._execution_generation.get()
        if execution_id is None:
            raise RLMHostError(
                "RLM API is not bound to this task; raw background threads do not inherit "
                "an IPython execution. Use the active async task or asyncio.to_thread()."
            )
        if generation > 0 and generation != self._pending_execution_generation:
            raise RLMHostError("the originating IPython execution is no longer active")
        return execution_id


class RLMAnswerDict(dict[str, Any]):
    """Execution-scoped legacy ``answer`` adapter for the canonical client."""

    def __init__(self, client: RLMClient) -> None:
        super().__init__(content="", ready=False)
        self.client = client

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "ready" and value:
            self.client.final_sync(str(self.get("content", "")))
        super().__setitem__(key, value)
