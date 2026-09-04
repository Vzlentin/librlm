"""Authenticated framed host channel for async IPython operations."""

from __future__ import annotations

import asyncio
import hmac
import json
import select
import socket
import socketserver
import struct
import threading
from collections.abc import Callable
from typing import Any, cast, overload

from rlm.environments.ipython_protocol import (
    PROTOCOL_VERSION,
    FailureCode,
    FinalRequest,
    GatherRequest,
    GatherResponse,
    HostRequest,
    HostResponse,
    OperationRequest,
    QueryRequest,
    QueryResponse,
    ReleaseRequest,
    RLMHostError,
    SpawnRequest,
    UnitResult,
    ValidateRequest,
    decode_request_payload,
    request_payload,
    result_from_wire,
    result_to_wire,
    retry_on_disconnect,
)

DEFAULT_MAX_MESSAGE_BYTES = 5 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT = 310.0


def _envelope(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RLMHostError(f"RLM host {label} must be an object")
    return cast(dict[str, object], value)


class _WireCodec:
    """JSON and length framing shared by the client and server adapters."""

    def __init__(self, max_message_bytes: int) -> None:
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")
        self.max_message_bytes = max_message_bytes

    def encode(self, value: object, too_large: str) -> bytes:
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        except (TypeError, ValueError) as error:
            raise TypeError(f"RLM protocol value is not JSON serializable: {error}") from error
        if len(encoded) > self.max_message_bytes:
            raise RLMHostError(too_large, FailureCode.MESSAGE_TOO_LARGE)
        return struct.pack(">I", len(encoded)) + encoded

    def decode(self, raw: bytes) -> object:
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RLMHostError("RLM protocol peer returned invalid JSON") from error

    def read_sync(self, connection: socket.socket) -> object:
        length = struct.unpack(">I", self._read_exact(connection, 4))[0]
        self._check_length(length)
        return self.decode(self._read_exact(connection, length))

    async def read_async(self, reader: asyncio.StreamReader) -> object:
        length = struct.unpack(">I", await reader.readexactly(4))[0]
        self._check_length(length)
        return self.decode(await reader.readexactly(length))

    def _check_length(self, length: int) -> None:
        if length > self.max_message_bytes:
            raise RLMHostError(
                "RLM protocol message exceeded the size limit",
                FailureCode.MESSAGE_TOO_LARGE,
            )

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


class HostChannel:
    """The one client seam for correlation, retries, timeouts, and wire errors."""

    def __init__(
        self,
        address: tuple[str, int],
        auth_token: str,
        *,
        timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        connection_retries: int = 1,
    ) -> None:
        if not auth_token:
            raise ValueError("auth_token must not be empty")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive or None")
        if connection_retries < 0:
            raise ValueError("connection_retries must be non-negative")
        self.address = address
        self.auth_token = auth_token
        self.timeout = timeout
        self.connection_retries = connection_retries
        self.codec = _WireCodec(max_message_bytes)

    @overload
    def request_sync(
        self,
        request: SpawnRequest | ReleaseRequest | FinalRequest | ValidateRequest,
    ) -> UnitResult: ...

    @overload
    def request_sync(self, request: GatherRequest) -> GatherResponse: ...

    @overload
    def request_sync(self, request: QueryRequest) -> QueryResponse: ...

    def request_sync(self, request: OperationRequest) -> HostResponse:
        encoded = self._encode_request(request)
        retries = self._connection_retries(request)
        try:
            for attempt in range(retries + 1):
                try:
                    with socket.create_connection(self.address, timeout=self.timeout) as connection:
                        connection.sendall(encoded)
                        return self._decode_response(self.codec.read_sync(connection), request)
                except RLMHostError:
                    raise
                except (ConnectionError, OSError, TimeoutError):
                    if attempt == retries:
                        raise
        except RLMHostError:
            raise
        except TimeoutError as error:
            raise RLMHostError("RLM host operation timed out") from error
        except (ConnectionError, OSError) as error:
            raise RLMHostError(f"RLM host connection failed: {error}") from error
        raise AssertionError("unreachable host-channel retry state")

    @overload
    async def request(
        self,
        request: SpawnRequest | ReleaseRequest | FinalRequest | ValidateRequest,
    ) -> UnitResult: ...

    @overload
    async def request(self, request: GatherRequest) -> GatherResponse: ...

    @overload
    async def request(self, request: QueryRequest) -> QueryResponse: ...

    async def request(self, request: OperationRequest) -> HostResponse:
        encoded = self._encode_request(request)
        retries = self._connection_retries(request)
        try:
            async with asyncio.timeout(self.timeout):
                connection_failures = 0
                while True:
                    try:
                        return await self._exchange(encoded, request)
                    except RLMHostError as error:
                        if error.code is not FailureCode.GATHER_PENDING:
                            raise
                        await asyncio.sleep(0.01)
                    except (ConnectionError, OSError, asyncio.IncompleteReadError):
                        connection_failures += 1
                        if connection_failures > retries:
                            raise
                        await asyncio.sleep(0)
        except RLMHostError:
            raise
        except TimeoutError as error:
            raise RLMHostError("RLM host operation timed out") from error
        except (ConnectionError, OSError, asyncio.IncompleteReadError) as error:
            raise RLMHostError(f"RLM host connection failed: {error}") from error
        raise AssertionError("unreachable host-channel retry state")

    def _connection_retries(self, request: OperationRequest) -> int:
        return self.connection_retries if retry_on_disconnect(request) else 0

    async def _exchange(self, encoded: bytes, request: OperationRequest) -> HostResponse:
        reader, writer = await asyncio.open_connection(*self.address)
        try:
            writer.write(encoded)
            await writer.drain()
            return self._decode_response(await self.codec.read_async(reader), request)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _encode_request(self, request: OperationRequest) -> bytes:
        return self.codec.encode(
            {
                "version": PROTOCOL_VERSION,
                "auth": self.auth_token,
                "id": request.request_id,
                "execution_id": request.execution_id,
                "op": request.operation,
                **request_payload(request),
            },
            "RLM host request exceeded the size limit",
        )

    def _decode_response(
        self,
        value: object,
        request: OperationRequest,
    ) -> HostResponse:
        data = _envelope(value, "response")
        if data.get("version") != PROTOCOL_VERSION:
            raise RLMHostError("RLM host response has an unsupported protocol version")
        if data.get("id") != request.request_id:
            raise RLMHostError("RLM host response id did not match the request")
        if data.get("ok") is True:
            if set(data) != {"version", "id", "ok", "result"}:
                raise RLMHostError("RLM host success response has invalid fields")
            return result_from_wire(request, data["result"])
        if data.get("ok") is not False:
            raise RLMHostError("RLM host response has an invalid status")
        if set(data) != {"version", "id", "ok", "error", "error_code"}:
            raise RLMHostError("RLM host failure response has invalid fields")
        try:
            code = FailureCode(data["error_code"])
        except (TypeError, ValueError) as error:
            raise RLMHostError("RLM host response has an invalid failure code") from error
        message = data["error"]
        raise RLMHostError(message if isinstance(message, str) else "RLM host failed", code)


class ProtocolConnection:
    """Per-request disconnect and response-size checks for host operations."""

    def __init__(self, connection: socket.socket, codec: _WireCodec) -> None:
        self.connection = connection
        self.codec = codec

    def disconnected(self) -> bool:
        readable, _, _ = select.select([self.connection], [], [], 0)
        if not readable:
            return False
        try:
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except BlockingIOError:
            return False
        except (ConnectionError, OSError):
            return True

    def ensure_response_fits(self, request: OperationRequest, result: HostResponse) -> None:
        self.codec.encode(
            {
                "version": PROTOCOL_VERSION,
                "id": request.request_id,
                "ok": True,
                "result": result_to_wire(request, result),
            },
            "RLM host response exceeded the size limit",
        )


RequestDispatcher = Callable[[HostRequest, ProtocolConnection], HostResponse]


class FramedProtocolServer:
    """Own server sockets, authentication, framing, handlers, and error envelopes."""

    def __init__(
        self,
        auth_token: str,
        dispatcher: RequestDispatcher,
        *,
        host: str,
        port: int,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        self.auth_token = auth_token
        self.dispatcher = dispatcher
        self.host = host
        self.port = port
        self.codec = _WireCodec(max_message_bytes)
        self._lock = threading.Lock()
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None
        self._connections: set[socket.socket] = set()
        self._stopping = False

    @property
    def address(self) -> tuple[str, int]:
        server = self._server
        if server is None:
            return self.host, self.port
        return self.host, int(server.server_address[1])

    def start(self) -> tuple[str, int]:
        with self._lock:
            if self._stopping:
                raise RuntimeError("framed protocol server is stopped")
            if self._server is not None:
                return self.address
        parent = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                connection = cast(socket.socket, self.request)
                if not parent._register(connection):
                    connection.close()
                    return
                try:
                    request_id = ""
                    try:
                        wire = parent.codec.read_sync(connection)
                        request_id, request = parent._decode_request(wire)
                        result = parent.dispatcher(
                            request,
                            ProtocolConnection(connection, parent.codec),
                        )
                        response = {
                            "version": PROTOCOL_VERSION,
                            "id": request.request_id,
                            "ok": True,
                            "result": result_to_wire(request, result),
                        }
                    except BaseException as error:
                        response = parent._failure(request_id, error)
                    parent._send(connection, response)
                finally:
                    parent._unregister(connection)

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = False
            block_on_close = True

        with self._lock:
            if self._stopping:
                raise RuntimeError("framed protocol server is stopped")
            if self._server is not None:
                return self.address
            server = Server((self.host, self.port), Handler)
            thread = threading.Thread(
                target=server.serve_forever,
                name="rlm-async-host",
                daemon=True,
            )
            self._server = server
            self._thread = thread
            thread.start()
            return self.address

    def _decode_request(self, value: object) -> tuple[str, HostRequest]:
        data = _envelope(value, "request")
        request_id = data.get("id") if isinstance(data.get("id"), str) else ""
        if data.get("version") != PROTOCOL_VERSION:
            raise RLMHostError("unsupported protocol version")
        supplied_auth = data.get("auth")
        if not isinstance(supplied_auth, str) or not hmac.compare_digest(
            supplied_auth,
            self.auth_token,
        ):
            raise RLMHostError("host authentication failed")
        execution_id = data.get("execution_id")
        operation = data.get("op")
        if not isinstance(request_id, str) or not request_id:
            raise RLMHostError("request id must be a non-empty string")
        if not isinstance(execution_id, str) or not execution_id:
            raise RLMHostError("execution id must be a non-empty string")
        if not isinstance(operation, str) or not operation:
            raise RLMHostError("host operation must be a non-empty string")
        reserved = {"version", "auth", "id", "execution_id", "op"}
        payload = {key: item for key, item in data.items() if key not in reserved}
        return request_id, decode_request_payload(
            operation,
            request_id,
            execution_id,
            payload,
        )

    @staticmethod
    def _failure(request_id: str, error: BaseException) -> dict[str, Any]:
        code = error.code if isinstance(error, RLMHostError) else FailureCode.OPERATION_FAILED
        return {
            "version": PROTOCOL_VERSION,
            "id": request_id,
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "error_code": code.value,
        }

    def _register(self, connection: socket.socket) -> bool:
        with self._lock:
            if self._stopping:
                return False
            self._connections.add(connection)
            return True

    def _unregister(self, connection: socket.socket) -> None:
        with self._lock:
            self._connections.discard(connection)

    def _send(self, connection: socket.socket, response: object) -> None:
        request_id = ""
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            request_id = response["id"]
        try:
            encoded = self.codec.encode(response, "RLM host response exceeded the size limit")
        except Exception as error:
            encoded = self.codec.encode(self._failure(request_id, error), "RLM host failure")
        try:
            connection.sendall(encoded)
        except (BrokenPipeError, ConnectionError, OSError):
            pass

    def stop(self) -> None:
        with self._lock:
            if self._stopping:
                return
            self._stopping = True
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join()
        with self._lock:
            self._connections.clear()
