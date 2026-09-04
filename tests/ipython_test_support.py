from __future__ import annotations

import json
import socket
import socketserver
import struct
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from rlm.core.child_execution import ChildExecution, ChildOutcome
from rlm.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary
from rlm.environments.ipython_async import AsyncRLMHost
from rlm.environments.ipython_client import RLMClient


@contextmanager
def started_host(host: AsyncRLMHost):
    host.start()
    try:
        yield host
    finally:
        host.stop()


def child_execution(runner, *, max_concurrent: int = 16) -> ChildExecution:
    return ChildExecution.from_outcome_callback(
        runner,
        max_concurrent=max_concurrent,
    )


@contextmanager
def running_host(runner, **kwargs):
    execution = child_execution(
        runner,
        max_concurrent=kwargs.get("max_concurrent", 16),
    )
    with started_host(AsyncRLMHost(execution, **kwargs)) as host:
        yield host


class ProtocolFaultProxy:
    """Inject request and response loss without reaching into host state."""

    def __init__(
        self,
        upstream: tuple[str, int],
        *,
        drop_response: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
    ) -> None:
        def read_exact(connection: socket.socket, size: int) -> bytes:
            chunks: list[bytes] = []
            while size:
                chunk = connection.recv(size)
                if not chunk:
                    raise ConnectionError("proxy connection closed")
                chunks.append(chunk)
                size -= len(chunk)
            return b"".join(chunks)

        def read_wire(connection: socket.socket) -> dict[str, Any]:
            length = struct.unpack(">I", read_exact(connection, 4))[0]
            value = json.loads(read_exact(connection, length))
            if not isinstance(value, dict):
                raise TypeError("proxy message must be an object")
            return value

        def encode_wire(value: dict[str, Any]) -> bytes:
            encoded = json.dumps(value, separators=(",", ":")).encode()
            return struct.pack(">I", len(encoded)) + encoded

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                request = read_wire(self.request)
                if not isinstance(request, dict):
                    return
                with socket.create_connection(upstream) as connection:
                    connection.sendall(encode_wire(request))
                    response = read_wire(connection)
                if not isinstance(response, dict):
                    return
                if drop_response is not None and drop_response(request, response):
                    return
                self.request.sendall(encode_wire(response))

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.address = ("127.0.0.1", int(self._server.server_address[1]))

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def __enter__(self) -> ProtocolFaultProxy:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def activate(client: RLMClient, execution_id: str) -> None:
    client.prepare_execution(execution_id)
    client.activate_execution()


def result(text: str, usage: Any = None) -> ChildOutcome:
    return ChildOutcome.external(
        status="ok",
        text=text,
        error=None,
        usage=usage if usage is not None else {},
        elapsed_ms=1,
        truncated=False,
    )


def completion(prompt: str, response: str, *, marker: str) -> RLMChatCompletion:
    return RLMChatCompletion(
        root_model="fake",
        prompt=prompt,
        response=response,
        usage_summary=UsageSummary(
            model_usage_summaries={
                marker: ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=1,
                    total_output_tokens=2,
                )
            }
        ),
        execution_time=0.01,
        metadata={"marker": marker},
    )
