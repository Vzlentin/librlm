#!/usr/bin/env python3
"""Shared stdio Jupyter bridge for host-owned RLM completions."""

from __future__ import annotations

import json
import os
import queue
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from jupyter_client import KernelManager

_OUTPUT_MESSAGE_CHARS = 16_384
BRIDGE_PROTOCOL_VERSION = 6
HOST_PROTOCOL_VERSION = 2
_HOST_RESPONSE_LIMIT = 5 * 1024 * 1024
_HOST_TIMEOUT_SECONDS = 310
_LIBRLM_ROOT = Path(__file__).resolve().parent.parent
_SEND_LOCK = threading.Lock()


def load_librlm_async() -> Any:
    """Load the dependency-free librlm primitive under isolated Python mode."""
    sys.path.insert(0, str(_LIBRLM_ROOT))
    from rlm.environments import ipython_async

    return ipython_async


_LIBRLM = load_librlm_async()


def send(message: dict[str, Any]) -> None:
    encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _SEND_LOCK:
        sys.stdout.write(encoded)
        sys.stdout.flush()


def terminate_bridge(_signum: int, _frame: Any) -> None:
    """Let process termination unwind through main() so the kernel is reaped."""
    raise SystemExit(0)


def parent_id(message: dict[str, Any]) -> str | None:
    value = message.get("parent_header", {}).get("msg_id")
    return value if isinstance(value, str) else None


def request_host_child_transport(request: Any, cancel: threading.Event) -> Any:
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
    child = _LIBRLM.ChildResult.from_wire(result)
    return _LIBRLM.ChildOutcome.external(
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
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "total": 0,
        },
    }


def request_host_child(request: Any, cancel: threading.Event) -> Any:
    """Adapt transport failures into a canonical typed child result."""
    started = time.monotonic()
    try:
        return request_host_child_transport(request, cancel)
    except Exception as error:
        return _LIBRLM.ChildOutcome.external(
            status="cancelled" if cancel.is_set() else "error",
            text=None,
            error=f"{type(error).__name__}: {error}",
            usage=empty_child_usage(),
            elapsed_ms=round((time.monotonic() - started) * 1000),
            truncated=False,
        )


def wait_for_execution(
    client: Any,
    manager: KernelManager,
    msg_id: str,
    *,
    capture_output: Any | None = None,
) -> dict[str, Any]:
    shell_reply: dict[str, Any] | None = None
    idle = False
    while shell_reply is None or not idle:
        received = False
        try:
            message = client.get_iopub_msg(timeout=0.05)
            received = True
            if parent_id(message) == msg_id:
                msg_type = message.get("header", {}).get("msg_type")
                content = message.get("content", {})
                if msg_type == "status" and content.get("execution_state") == "idle":
                    idle = True
                elif capture_output is not None:
                    capture_output(msg_type, content)
        except queue.Empty:
            pass

        try:
            message = client.get_shell_msg(timeout=0.05)
            received = True
            if (
                parent_id(message) == msg_id
                and message.get("header", {}).get("msg_type") == "execute_reply"
            ):
                shell_reply = message
        except queue.Empty:
            pass

        if not received and not manager.is_alive():
            raise RuntimeError("IPython kernel exited during execution")

    return shell_reply or {}


def execute_hidden(client: Any, manager: KernelManager, code: str) -> None:
    msg_id = client.execute(
        code,
        silent=True,
        store_history=False,
        allow_stdin=False,
        stop_on_error=True,
    )
    reply = wait_for_execution(client, manager, msg_id)
    content = reply.get("content", {})
    if content.get("status") != "ok":
        name = str(content.get("ename", "BootstrapError"))
        value = str(content.get("evalue", ""))
        raise RuntimeError(f"{name}: {value}".rstrip())


def bootstrap_code(
    async_address: tuple[str, int],
    async_auth_token: str,
    execution_id: str | None = None,
    working_dir: str | None = None,
) -> str:
    return f"""
import sys as _rlm_sys
if {str(_LIBRLM_ROOT)!r} not in _rlm_sys.path:
    _rlm_sys.path.insert(0, {str(_LIBRLM_ROOT)!r})
from rlm.environments.ipython_kernel import install_kernel_runtime as _install_rlm
_host_rlm_client = _install_rlm(
    get_ipython(),
    {async_address!r},
    {async_auth_token!r},
    execution_id={execution_id!r},
    working_dir={working_dir!r},
    client_key="_host_rlm_client",
)
"""


def execute(
    client: Any,
    manager: KernelManager,
    host: Any,
    request_id: str,
    code: str,
    cwd: str,
) -> None:
    host.begin_execution(request_id)
    successful = False
    ended = False
    summary: dict[str, Any] | None = None
    try:
        # Set execution identity out of band so a native cell magic remains on line 1.
        execute_hidden(
            client,
            manager,
            bootstrap_code(host.address, host.auth_token, request_id, cwd),
        )
        msg_id = client.execute(
            code,
            silent=False,
            store_history=True,
            allow_stdin=False,
            stop_on_error=True,
        )

        def emit_output(event: dict[str, Any]) -> None:
            if event["type"] == "clear":
                send({"type": "clear", "request_id": request_id})
                return
            kind = event["kind"]
            text = event["text"]
            for start in range(0, len(text), _OUTPUT_MESSAGE_CHARS):
                end = min(start + _OUTPUT_MESSAGE_CHARS, len(text))
                send(
                    {
                        "type": "output",
                        "request_id": request_id,
                        "kind": kind,
                        "text": text[start:end],
                        "continuation": start > 0,
                        "final": end == len(text),
                    }
                )

        output_stream = _LIBRLM.IPythonOutputStream(emit_output)
        shell_reply = wait_for_execution(client, manager, msg_id, capture_output=output_stream.feed)
        content = shell_reply.get("content", {})
        status = str(content.get("status", "error"))
        successful = status == "ok"
        error: dict[str, str] | None = None
        if not successful:
            error = {
                "ename": str(content.get("ename", "ExecutionError")),
                "evalue": str(content.get("evalue", "")),
            }
        execution_summary = host.end_execution(request_id, successful=successful)
        ended = True
        summary = execution_summary.to_wire()
        send(
            {
                "type": "result",
                "request_id": request_id,
                "status": status,
                "execution_count": content.get("execution_count"),
                "error": error,
                "host": summary,
            }
        )
    finally:
        if not ended:
            host.end_execution(request_id, successful=False)


def main(*, kernel_manager: KernelManager | None = None) -> int:
    """Run the shared bridge, accepting a harness-owned kernelspec when supplied."""
    signal.signal(signal.SIGTERM, terminate_bridge)
    cwd = os.environ.get("RLM_KERNEL_CWD") or os.getcwd()
    manager = kernel_manager
    if manager is None:
        manager = KernelManager(kernel_name="python3")
        # The standalone bridge and its kernel must use the same interpreter.
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
    client: Any | None = None
    host: Any | None = None
    exit_code = 0
    cleanup_errors: list[BaseException] = []

    try:
        if not os.environ.get("RLM_HOST_SOCKET") or not os.environ.get("RLM_HOST_TOKEN"):
            raise RuntimeError("RLM host socket configuration is missing")
        child_execution = _LIBRLM.ChildExecution.from_outcome_callback(
            request_host_child,
            max_concurrent=16,
        )
        host = _LIBRLM.AsyncRLMHost(
            child_execution,
            on_activity=lambda execution_id, message: send(
                {"type": "activity", "request_id": execution_id, "message": message}
            ),
            on_release=lambda execution_id: send({"type": "release", "execution_id": execution_id}),
        )
        host.start()

        kernel_env = dict(os.environ)
        kernel_env["NO_COLOR"] = "1"
        manager.start_kernel(
            cwd=cwd,
            env=kernel_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        kernel_pgid = getattr(manager.provisioner, "pgid", None)
        if not isinstance(kernel_pgid, int) or kernel_pgid <= 1:
            raise RuntimeError("Jupyter did not report a valid kernel process group")
        send({"type": "kernel_started", "kernel_pgid": kernel_pgid})
        client = manager.client()
        client.start_channels()
        client.wait_for_ready(timeout=30)
        execute_hidden(client, manager, bootstrap_code(host.address, host.auth_token))
        send(
            {
                "type": "ready",
                "protocol": BRIDGE_PROTOCOL_VERSION,
                "host_protocol": HOST_PROTOCOL_VERSION,
                "kernel_pgid": kernel_pgid,
                "connection_file": str(Path(manager.connection_file).resolve()),
            }
        )

        for line in sys.stdin:
            message: Any = None
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("request must be a JSON object")
                msg_type = message.get("type")
                if msg_type == "execute":
                    request_id = message.get("request_id")
                    code = message.get("code")
                    request_cwd = message.get("cwd")
                    if (
                        not isinstance(request_id, str)
                        or not isinstance(code, str)
                        or not isinstance(request_cwd, str)
                    ):
                        raise ValueError("execute requires string request_id, code, and cwd")
                    if not os.path.isabs(request_cwd) or not os.path.isdir(request_cwd):
                        raise ValueError("execute cwd must be an accessible absolute directory")
                    execute(client, manager, host, request_id, code, request_cwd)
                elif msg_type == "shutdown":
                    break
                else:
                    raise ValueError(f"unknown request type: {msg_type!r}")
            except Exception as error:
                request_id = message.get("request_id") if isinstance(message, dict) else None
                send(
                    {
                        "type": "bridge_error",
                        "request_id": request_id,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception as error:
        send({"type": "fatal", "error": str(error), "traceback": traceback.format_exc()})
        exit_code = 1
    finally:
        if host is not None:
            try:
                host.stop()
            except BaseException as error:
                cleanup_errors.append(error)
        if client is not None:
            try:
                client.stop_channels()
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        except BaseException as error:
            cleanup_errors.append(error)

    if cleanup_errors:
        details = "\n".join("".join(traceback.format_exception(error)) for error in cleanup_errors)
        send(
            {
                "type": "fatal",
                "error": f"bridge cleanup failed ({len(cleanup_errors)} error(s))",
                "traceback": details,
            }
        )
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
