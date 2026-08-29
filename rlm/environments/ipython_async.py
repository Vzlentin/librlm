"""Reusable async RLM primitives for persistent IPython kernels.

The kernel-facing :class:`RLMClient` exposes ``spawn``, ``gather``, and
``final``. :class:`AsyncRLMHost` owns live handles, atomic gather claims,
structured results, final-value gating, and execution cancellation in the
parent process. Model execution is supplied by a callback, so environments can
bind these mechanics to any model runtime without importing a provider SDK.
"""

from __future__ import annotations

import asyncio
import contextvars
import hmac
import json
import os
import secrets
import socket
import socketserver
import struct
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

_PROTOCOL_VERSION = 1
_DEFAULT_MAX_MESSAGE_BYTES = 5 * 1024 * 1024
_DEFAULT_MAX_REQUEST_BYTES = 1024 * 1024
_DEFAULT_MAX_LIVE_HANDLES = 16
_DEFAULT_REQUEST_TIMEOUT = 310.0


class RLMHostError(RuntimeError):
    """A host operation failed without invalidating the IPython kernel."""


class IPythonOutputStream:
    """Normalize incremental Jupyter output messages and clear semantics."""

    def __init__(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.callback = callback
        self._pending_clear = False

    def feed(self, msg_type: str, content: dict[str, Any]) -> None:
        if msg_type == "clear_output":
            if bool(content.get("wait")):
                self._pending_clear = True
            else:
                self._clear()
            return

        event: dict[str, Any] | None = None
        if msg_type == "stream":
            event = {
                "type": "text",
                "kind": str(content.get("name", "stdout")),
                "text": str(content.get("text", "")),
            }
        elif msg_type in ("execute_result", "display_data", "update_display_data"):
            value = content.get("data", {}).get("text/plain")
            if isinstance(value, list):
                value = "".join(str(part) for part in value)
            if isinstance(value, str):
                event = {
                    "type": "text",
                    "kind": "result" if msg_type == "execute_result" else "display",
                    "text": value,
                }
        elif msg_type == "error":
            trace = content.get("traceback")
            if isinstance(trace, list):
                text = "\n".join(str(line) for line in trace)
            else:
                name = str(content.get("ename", "Error"))
                value = str(content.get("evalue", ""))
                text = f"{name}: {value}".rstrip()
            event = {"type": "text", "kind": "error", "text": text}

        if event is not None:
            if self._pending_clear:
                self._clear()
            if event["text"]:
                self.callback(event)

    def _clear(self) -> None:
        self._pending_clear = False
        self.callback({"type": "clear"})


@dataclass(frozen=True, slots=True)
class ChildRequest:
    """One backend-neutral child request admitted by ``rlm.spawn``."""

    handle: str
    execution_id: str
    task: str
    context: str | None
    cwd: str


ChildResult = dict[str, Any]
ChildRunner = Callable[[ChildRequest, threading.Event], ChildResult]
ActivityCallback = Callable[[str, str], None]
ReleaseCallback = Callable[[str], None]


@dataclass(slots=True, eq=False)
class RLMHandle:
    """Opaque, single-consumption handle returned by :meth:`RLMClient.spawn`."""

    _client: RLMClient = field(repr=False)
    _id: str = field(repr=False)
    _result: ChildResult | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        state = "ready" if self._result is not None else "pending"
        return f"<rlm.Handle {state}>"


class RLMClient:
    """Async kernel API backed by an authenticated ``AsyncRLMHost`` socket."""

    def __init__(
        self,
        address: tuple[str, int],
        auth_token: str,
        *,
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,
        max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        if not auth_token:
            raise ValueError("auth_token must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._address = address
        self._auth_token = auth_token
        self._timeout = timeout
        self._max_message_bytes = max_message_bytes
        self._pending_execution_id: str | None = None
        self._execution_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"rlm_execution_id_{id(self)}", default=None
        )
        self._hook_installed = False

    def _set_execution(self, execution_id: str) -> None:
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError("execution id must be a non-empty string")
        self._pending_execution_id = execution_id

    def _activate_execution(self, *_args: Any, **_kwargs: Any) -> None:
        # IPython calls pre_run_cell in the user cell's task context. Tasks
        # created by that cell inherit this value instead of adopting a later
        # cell's identity from mutable global state.
        if self._pending_execution_id is not None:
            self._execution_id.set(self._pending_execution_id)

    def _install_ipython_hook(self, shell: Any) -> None:
        if self._hook_installed:
            return
        shell.events.register("pre_run_cell", self._activate_execution)
        self._hook_installed = True

    async def spawn(self, task: str, *, context: str | None = None) -> RLMHandle:
        """Start one child and return immediately with an opaque handle."""
        if not isinstance(task, str):
            raise TypeError("rlm.spawn task must be a string")
        if context is not None and not isinstance(context, str):
            raise TypeError("rlm.spawn context must be a string or None")
        result = await self._request("spawn", task=task, context=context, cwd=os.getcwd())
        handle = result.get("handle") if isinstance(result, dict) else None
        if not isinstance(handle, str):
            raise RLMHostError("RLM host returned an invalid child handle")
        return RLMHandle(self, handle)

    async def gather(self, handles: Iterable[RLMHandle]) -> list[ChildResult]:
        """Wait for handles and return ordered structured child results."""
        try:
            items = list(handles)
        except TypeError as error:
            raise TypeError("rlm.gather requires an iterable of rlm handles") from error
        for handle in items:
            if not isinstance(handle, RLMHandle) or handle._client is not self:
                raise TypeError("rlm.gather received a handle from another RLM client")

        unresolved = [handle for handle in items if handle._result is None]
        if unresolved:
            result = await self._request("gather", handles=[handle._id for handle in unresolved])
            values = result.get("results") if isinstance(result, dict) else None
            if not isinstance(values, list) or len(values) != len(unresolved):
                raise RLMHostError("RLM host returned an invalid gather result")
            for handle, value in zip(unresolved, values, strict=True):
                if not isinstance(value, dict):
                    raise RLMHostError("RLM host returned a malformed child value")
                handle._result = value

        return [dict(handle._result or {}) for handle in items]

    async def final(self, value: Any) -> None:
        """Freeze the first JSON value accepted for the active execution."""
        result = await self._request("final", value=value)
        if not isinstance(result, dict) or "value" not in result:
            raise RLMHostError("RLM host returned an invalid final acknowledgement")

    async def _request(self, operation: str, **payload: Any) -> Any:
        execution_id = self._execution_id.get()
        if execution_id is None:
            raise RLMHostError("RLM API is not associated with an active IPython cell")
        request_id = secrets.token_hex(16)
        request = {
            "version": _PROTOCOL_VERSION,
            "auth": self._auth_token,
            "id": request_id,
            "execution_id": execution_id,
            "op": operation,
            **payload,
        }
        try:
            encoded = json.dumps(
                request, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise TypeError(f"RLM request is not JSON serializable: {error}") from error
        if len(encoded) > self._max_message_bytes:
            raise RLMHostError("RLM host request exceeded the size limit")

        try:
            async with asyncio.timeout(self._timeout):
                reader, writer = await asyncio.open_connection(self._address[0], self._address[1])
                try:
                    writer.write(struct.pack(">I", len(encoded)) + encoded)
                    await writer.drain()
                    raw_length = await reader.readexactly(4)
                    length = struct.unpack(">I", raw_length)[0]
                    if length > self._max_message_bytes:
                        raise RLMHostError("RLM host response exceeded the size limit")
                    raw = await reader.readexactly(length)
                finally:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except Exception:
                        pass
        except RLMHostError:
            raise
        except TimeoutError as error:
            raise RLMHostError("RLM host operation timed out") from error
        except (ConnectionError, OSError, asyncio.IncompleteReadError) as error:
            raise RLMHostError(f"RLM host connection failed: {error}") from error

        try:
            response = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RLMHostError("RLM host returned invalid JSON") from error
        if not isinstance(response, dict):
            raise RLMHostError("RLM host response must be an object")
        if response.get("version") != _PROTOCOL_VERSION:
            raise RLMHostError("RLM host response has an unsupported protocol version")
        if response.get("id") != request_id:
            raise RLMHostError("RLM host response id did not match the request")
        if response.get("ok") is not True:
            message = response.get("error")
            raise RLMHostError(message if isinstance(message, str) else "RLM host failed")
        return response.get("result")


@dataclass(slots=True)
class _Execution:
    cancel: threading.Event = field(default_factory=threading.Event)
    final_set: bool = False
    final_value: Any = None
    spawned: int = 0
    gathered_handles: list[str] = field(default_factory=list)
    gathered_results: list[ChildResult] = field(default_factory=list)
    usages: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class _Child:
    request: ChildRequest
    cancel: threading.Event = field(default_factory=threading.Event)
    future: Future[ChildResult] | None = None
    gather_claim: tuple[str, str] | None = None


class AsyncRLMHost:
    """Own async handles and execution-scoped state for an IPython kernel.

    ``child_runner`` is called in a worker thread. It must return a JSON object
    and should stop promptly when its cancellation event becomes set.
    """

    def __init__(
        self,
        child_runner: ChildRunner,
        *,
        on_activity: ActivityCallback | None = None,
        on_release: ReleaseCallback | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        max_concurrent: int = _DEFAULT_MAX_LIVE_HANDLES,
        max_live_handles: int = _DEFAULT_MAX_LIVE_HANDLES,
        max_request_bytes: int = _DEFAULT_MAX_REQUEST_BYTES,
        max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        if max_concurrent <= 0 or max_live_handles <= 0:
            raise ValueError("concurrency and live-handle limits must be positive")
        self.child_runner = child_runner
        self.on_activity = on_activity
        self.on_release = on_release
        self.host = host
        self._port = port
        self.max_live_handles = max_live_handles
        self.max_request_bytes = max_request_bytes
        self.max_message_bytes = max_message_bytes
        self.auth_token = secrets.token_hex(32)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="rlm-child"
        )
        self._lock = threading.RLock()
        self._executions: dict[str, _Execution] = {}
        self._children: dict[str, _Child] = {}
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None
        self._shutting_down = False

    def start(self) -> tuple[str, int]:
        if self._server is not None:
            return self.address
        parent = self

        class _Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                try:
                    request = parent._recv(self.request)
                    response = parent._dispatch(request)
                except Exception as error:
                    request_id = ""
                    if isinstance(locals().get("request"), dict):
                        value = request.get("id")
                        request_id = value if isinstance(value, str) else ""
                    response = parent._failure(request_id, f"{type(error).__name__}: {error}")
                try:
                    parent._send(self.request, response)
                except (BrokenPipeError, ConnectionError, OSError):
                    pass

        class _Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = _Server((self.host, self._port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="rlm-async-host", daemon=True
        )
        self._thread.start()
        return self.address

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            return self.host, self._port
        return self.host, int(self._server.server_address[1])

    def begin_execution(self, execution_id: str) -> None:
        if not execution_id:
            raise ValueError("execution id must not be empty")
        with self._lock:
            if self._shutting_down:
                raise RLMHostError("RLM host is shutting down")
            if execution_id in self._executions:
                raise RLMHostError("duplicate execution id")
            self._executions[execution_id] = _Execution()

    def end_execution(self, execution_id: str, successful: bool) -> dict[str, Any]:
        release = False
        with self._lock:
            execution = self._executions.pop(execution_id, None)
            if execution is None:
                return {
                    "has_final": False,
                    "usages": [],
                    "spawned": 0,
                    "gathered": 0,
                    "gathered_handles": [],
                    "gathered_results": [],
                    "discarded_handles": [],
                }
            execution.cancel.set()
            for child in self._children.values():
                if child.gather_claim and child.gather_claim[1] == execution_id:
                    child.gather_claim = None
                if not successful and child.request.execution_id == execution_id:
                    child.cancel.set()
            discarded_handles: list[str] = []
            if not successful:
                for handle, child in list(self._children.items()):
                    if child.request.execution_id == execution_id:
                        self._children.pop(handle, None)
                        discarded_handles.append(handle)
            release = not self._has_origin_handles_locked(execution_id)
            summary = {
                "has_final": successful and execution.final_set,
                "final_value": execution.final_value
                if successful and execution.final_set
                else None,
                "usages": list(execution.usages),
                "spawned": execution.spawned,
                "gathered": len(execution.gathered_handles),
                "gathered_handles": list(execution.gathered_handles),
                "gathered_results": list(execution.gathered_results),
                "discarded_handles": discarded_handles,
            }
        if release:
            self._released(execution_id)
        return summary

    def cancel_execution(self, execution_id: str) -> dict[str, Any]:
        return self.end_execution(execution_id, successful=False)

    def reset(self) -> None:
        with self._lock:
            execution_ids = set(self._executions)
            execution_ids.update(child.request.execution_id for child in self._children.values())
            for execution in self._executions.values():
                execution.cancel.set()
            for child in self._children.values():
                child.cancel.set()
            self._executions.clear()
            self._children.clear()
        for execution_id in execution_ids:
            self._released(execution_id)

    def stop(self) -> None:
        with self._lock:
            if self._shutting_down:
                return
            self._shutting_down = True
        self.reset()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _dispatch(self, data: dict[str, Any]) -> dict[str, Any]:
        request_id = data.get("id")
        if not isinstance(request_id, str):
            request_id = ""
        if data.get("version") != _PROTOCOL_VERSION:
            return self._failure(request_id, "unsupported protocol version")
        supplied_auth = data.get("auth")
        if not isinstance(supplied_auth, str) or not hmac.compare_digest(
            supplied_auth, self.auth_token
        ):
            return self._failure(request_id, "host authentication failed")
        execution_id = data.get("execution_id")
        if not isinstance(execution_id, str):
            return self._failure(request_id, "execution_id must be a string")

        try:
            operation = data.get("op")
            if operation == "spawn":
                result = self._spawn(execution_id, data)
            elif operation == "gather":
                result = self._gather(execution_id, request_id, data)
            elif operation == "final":
                result = self._final(execution_id, data.get("value"))
            else:
                raise RLMHostError("unknown host operation")
            return self._success(request_id, result)
        except Exception as error:
            return self._failure(request_id, str(error))

    def _spawn(self, execution_id: str, data: dict[str, Any]) -> dict[str, Any]:
        task = data.get("task")
        context = data.get("context")
        cwd = data.get("cwd")
        if not isinstance(task, str) or not task.strip():
            raise RLMHostError("spawn requires a non-empty string task")
        if context is not None and not isinstance(context, str):
            raise RLMHostError("spawn context must be a string or None")
        if not isinstance(cwd, str) or not os.path.isabs(cwd) or not os.path.isdir(cwd):
            raise RLMHostError("spawn requires an accessible absolute cwd")
        request_bytes = len(task.encode()) + (len(context.encode()) if context else 0)
        if request_bytes > self.max_request_bytes:
            raise RLMHostError("child task and context exceed the size limit")

        with self._lock:
            execution = self._active_execution_locked(execution_id)
            if len(self._children) >= self.max_live_handles:
                raise RLMHostError("too many live child handles")
            handle = secrets.token_urlsafe(18)
            request = ChildRequest(handle, execution_id, task, context, cwd)
            child = _Child(request=request)
            self._children[handle] = child
            execution.spawned += 1
            child.future = self._executor.submit(self._run_child, child)
        return {"handle": handle}

    def _run_child(self, child: _Child) -> ChildResult:
        started = time.monotonic()
        try:
            result = self.child_runner(child.request, child.cancel)
            if not isinstance(result, dict):
                raise TypeError("child_runner must return a JSON object")
            json.dumps(result, ensure_ascii=False, allow_nan=False)
            return result
        except Exception as error:
            return {
                "status": "cancelled" if child.cancel.is_set() else "error",
                "text": None,
                "error": f"{type(error).__name__}: {error}",
                "usage": {},
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "truncated": False,
            }

    def _gather(self, execution_id: str, request_id: str, data: dict[str, Any]) -> dict[str, Any]:
        handles = data.get("handles")
        if not isinstance(handles, list) or any(not isinstance(item, str) for item in handles):
            raise RLMHostError("gather requires a list of handle ids")
        if len(handles) > self.max_live_handles:
            raise RLMHostError("too many handles in gather")

        with self._lock:
            execution = self._active_execution_locked(execution_id)
            records: list[_Child] = []
            for handle in handles:
                child = self._children.get(handle)
                if child is None:
                    raise RLMHostError(f"unknown or already gathered child handle: {handle}")
                if child.gather_claim and child.gather_claim != (request_id, execution_id):
                    raise RLMHostError(f"child handle is already being gathered: {handle}")
                records.append(child)
            unique_records = list({child.request.handle: child for child in records}.values())
            for child in unique_records:
                child.gather_claim = (request_id, execution_id)

        committed = False
        try:
            if unique_records:
                suffix = "" if len(unique_records) == 1 else "ren"
                self._activity(
                    execution_id, f"Waiting for {len(unique_records)} RLM child{suffix}…"
                )
            results_by_handle: dict[str, ChildResult] = {}
            remaining = list(unique_records)
            completed = 0
            while remaining:
                if execution.cancel.is_set():
                    raise RLMHostError("the originating IPython execution is no longer active")
                for child in list(remaining):
                    assert child.future is not None
                    try:
                        result = child.future.result(timeout=0)
                    except FutureTimeout:
                        continue
                    results_by_handle[child.request.handle] = result
                    remaining.remove(child)
                    completed += 1
                    self._activity(
                        execution_id,
                        f"RLM children completed: {completed}/{len(unique_records)}",
                    )
                if remaining:
                    execution.cancel.wait(0.02)

            results = [results_by_handle[child.request.handle] for child in records]
            encoded = json.dumps(results, ensure_ascii=False, allow_nan=False).encode()
            if len(encoded) > self.max_message_bytes:
                raise RLMHostError("gather response exceeds the size limit")

            releases: set[str] = set()
            with self._lock:
                current = self._active_execution_locked(execution_id)
                if current is not execution or execution.cancel.is_set():
                    raise RLMHostError("the originating IPython execution is no longer active")
                for child in unique_records:
                    result = results_by_handle[child.request.handle]
                    usage = result.get("usage")
                    if usage is not None:
                        execution.usages.append(usage)
                    execution.gathered_handles.append(child.request.handle)
                    execution.gathered_results.append(result)
                    self._children.pop(child.request.handle, None)
                    if not self._has_origin_handles_locked(child.request.execution_id):
                        releases.add(child.request.execution_id)
                committed = True
            for origin in releases:
                self._released(origin)
            return {"results": results}
        finally:
            if not committed:
                with self._lock:
                    for child in unique_records:
                        if child.gather_claim == (request_id, execution_id):
                            child.gather_claim = None

    def _final(self, execution_id: str, value: Any) -> dict[str, Any]:
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        except (TypeError, ValueError) as error:
            raise TypeError(f"final value must be JSON serializable: {error}") from error
        if len(encoded) > self.max_request_bytes:
            raise RLMHostError("final value exceeds the size limit")
        with self._lock:
            execution = self._active_execution_locked(execution_id)
            if not execution.final_set:
                execution.final_set = True
                execution.final_value = value
            return {"accepted": True, "value": execution.final_value}

    def _active_execution_locked(self, execution_id: str) -> _Execution:
        execution = self._executions.get(execution_id)
        if execution is None or execution.cancel.is_set():
            raise RLMHostError("the originating IPython execution is no longer active")
        return execution

    def _has_origin_handles_locked(self, execution_id: str) -> bool:
        return any(child.request.execution_id == execution_id for child in self._children.values())

    def _activity(self, execution_id: str, message: str) -> None:
        if self.on_activity is not None:
            try:
                self.on_activity(execution_id, message)
            except Exception:
                pass

    def _released(self, execution_id: str) -> None:
        if self.on_release is not None:
            try:
                self.on_release(execution_id)
            except Exception:
                pass

    def _recv(self, connection: socket.socket) -> dict[str, Any]:
        raw_length = self._read_exact(connection, 4)
        length = struct.unpack(">I", raw_length)[0]
        if length > self.max_message_bytes:
            raise RLMHostError("request exceeded the size limit")
        raw = self._read_exact(connection, length)
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RLMHostError("request must be a JSON object")
        return value

    def _send(self, connection: socket.socket, value: dict[str, Any]) -> None:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        if len(raw) > self.max_message_bytes:
            raw = json.dumps(self._failure("", "response exceeded the size limit")).encode()
        connection.sendall(struct.pack(">I", len(raw)) + raw)

    @staticmethod
    def _read_exact(connection: socket.socket, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = connection.recv(remaining)
            if not chunk:
                raise ConnectionError("connection closed before message completed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _success(request_id: str, result: Any) -> dict[str, Any]:
        return {"version": _PROTOCOL_VERSION, "id": request_id, "ok": True, "result": result}

    @staticmethod
    def _failure(request_id: str, error: str) -> dict[str, Any]:
        return {"version": _PROTOCOL_VERSION, "id": request_id, "ok": False, "error": error}
