"""IPython extension that hosts RLM child handles inside the kernel.

Load with ``%load_ext rlm.ipython_extension`` in a kernel whose environment has
``RLM_HOST_SOCKET`` and ``RLM_HOST_TOKEN``. Each non-silent cell is one host
execution; child completions are forwarded to the harness socket (host protocol 2).
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import threading
import time
from typing import Any

from rlm.environments.ipython_async import (
    AsyncRLMHost,
    ChildExecution,
    ChildOutcome,
    ChildRequest,
    ChildResult,
    ExecutionSummary,
)
from rlm.environments.ipython_kernel import install_kernel_runtime

HOST_PROTOCOL_VERSION = 2
_HOST_RESPONSE_LIMIT = 5 * 1024 * 1024
_HOST_TIMEOUT_SECONDS = 310
_CLIENT_KEY = "_rlm_client"

# A harness may name the next cell's execution; otherwise the kernel generates one.
next_execution_id: str | None = None
last_summary = ExecutionSummary()
_released: list[str] = []
_extension: _Extension | None = None


def request_host_child_transport(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
    """Send one admitted child request to the harness completion runtime."""
    socket_path = os.environ.get("RLM_HOST_SOCKET")
    auth_token = os.environ.get("RLM_HOST_TOKEN")
    if not socket_path or not auth_token:
        raise RuntimeError("RLM host socket configuration is missing")
    request_id = secrets.token_hex(16)
    payload = {
        "version": HOST_PROTOCOL_VERSION,
        "auth": auth_token,
        "id": request_id,
        "execution_id": request.execution_id,
        "op": "complete",
        "task": request.task,
        "context": request.context,
        "cwd": request.working_dir,
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
    )

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(0.1)
    try:
        connection.connect(socket_path)
        connection.sendall(encoded)
        response = bytearray()
        deadline = time.monotonic() + _HOST_TIMEOUT_SECONDS
        while b"\n" not in response:
            if cancel.is_set():
                raise RuntimeError("child completion cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("Host child completion timed out")
            try:
                chunk = connection.recv(64 * 1024)
            except TimeoutError:
                continue
            if not chunk:
                raise ConnectionError("RLM host closed the socket without a response")
            response.extend(chunk)
            if len(response) > _HOST_RESPONSE_LIMIT:
                raise RuntimeError("RLM host response exceeded the size limit")
    finally:
        connection.close()

    try:
        value = json.loads(bytes(response).split(b"\n", 1)[0])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("RLM host returned invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("RLM host response must be an object")
    if value.get("version") != HOST_PROTOCOL_VERSION or value.get("id") != request_id:
        raise RuntimeError("RLM host returned a mismatched response")
    if value.get("ok") is not True:
        message = value.get("error")
        raise RuntimeError(message if isinstance(message, str) else "Host child failed")
    result = value.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("RLM host returned a malformed child result")
    child = ChildResult.from_wire(result)
    return ChildOutcome.external(
        status=child.status,
        text=child.text,
        error=child.error,
        usage=child.usage,
        elapsed_ms=child.elapsed_ms,
        truncated=child.truncated,
    )


def empty_child_usage() -> dict[str, Any]:
    """Return the canonical zero value for the child-usage wire schema."""
    return {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }


def request_host_child(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
    """Adapt transport failures into a canonical typed child result."""
    started = time.monotonic()
    try:
        return request_host_child_transport(request, cancel)
    except Exception as error:
        return ChildOutcome.external(
            status="cancelled" if cancel.is_set() else "error",
            text=None,
            error=f"{type(error).__name__}: {error}",
            usage=empty_child_usage(),
            elapsed_ms=round((time.monotonic() - started) * 1000),
            truncated=False,
        )


def execution_report() -> str:
    """Return the last cell's summary and execution ids released since the last report."""
    released = [_released.pop(0) for _ in range(len(_released))]
    return json.dumps(
        {"host": last_summary.to_wire(), "released": released},
        ensure_ascii=False,
        allow_nan=False,
    )


class _Extension:
    def __init__(self, shell: Any) -> None:
        if not os.environ.get("RLM_HOST_SOCKET") or not os.environ.get("RLM_HOST_TOKEN"):
            raise RuntimeError("RLM host socket configuration is missing")
        self.shell = shell
        self.execution_id: str | None = None
        self.host = AsyncRLMHost(
            ChildExecution.from_outcome_callback(request_host_child, max_concurrent=16),
            on_release=_released.append,
        )
        self.host.start()
        # Registered before the client's activation hook so each cell begins first.
        shell.events.register("pre_run_cell", self.pre_run_cell)
        shell.events.register("post_run_cell", self.post_run_cell)
        self.client = self.install()

    def install(self, execution_id: str | None = None) -> Any:
        return install_kernel_runtime(
            self.shell,
            self.host.address,
            self.host.auth_token,
            execution_id=execution_id,
            client_key=_CLIENT_KEY,
        )

    def pre_run_cell(self, _info: Any) -> None:
        global next_execution_id
        execution_id, next_execution_id = next_execution_id or secrets.token_hex(12), None
        self.host.begin_execution(execution_id)
        self.execution_id = execution_id
        self.install(execution_id)

    def post_run_cell(self, result: Any) -> None:
        global last_summary
        execution_id, self.execution_id = self.execution_id, None
        if execution_id is None:
            last_summary = ExecutionSummary()
            return
        last_summary = self.host.end_execution(execution_id, successful=bool(result.success))

    def close(self) -> None:
        self.shell.events.unregister("pre_run_cell", self.pre_run_cell)
        self.shell.events.unregister("post_run_cell", self.post_run_cell)
        self.client.uninstall_ipython(self.shell)
        self.shell.user_ns.pop("rlm", None)
        self.host.stop()


def load_ipython_extension(shell: Any) -> None:
    global _extension
    if _extension is None:
        _extension = _Extension(shell)


def unload_ipython_extension(_shell: Any) -> None:
    global _extension
    if _extension is not None:
        _extension.close()
        _extension = None
