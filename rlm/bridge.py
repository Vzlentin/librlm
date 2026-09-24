#!/usr/bin/env python3
"""Stdio Jupyter bridge whose kernel hosts RLM through rlm.ipython_extension."""

from __future__ import annotations

import ast
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from jupyter_client import KernelManager

_OUTPUT_MESSAGE_CHARS = 16_384
BRIDGE_PROTOCOL_VERSION = 6
HOST_PROTOCOL_VERSION = 2
_LIBRLM_ROOT = Path(__file__).resolve().parent.parent
_SEND_LOCK = threading.Lock()
_REPORT = {"rlm": "__import__('rlm.ipython_extension', fromlist=['_']).execution_report()"}

sys.path.insert(0, str(_LIBRLM_ROOT))
from rlm.environments.ipython_kernel import IPythonOutputStream  # noqa: E402


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


def execute_hidden(
    client: Any,
    manager: KernelManager,
    code: str,
    user_expressions: dict[str, str] | None = None,
) -> dict[str, Any]:
    msg_id = client.execute(
        code,
        silent=True,
        store_history=False,
        user_expressions=user_expressions,
        allow_stdin=False,
        stop_on_error=True,
    )
    reply = wait_for_execution(client, manager, msg_id)
    content = reply.get("content", {})
    if content.get("status") != "ok":
        name = str(content.get("ename", "BootstrapError"))
        value = str(content.get("evalue", ""))
        raise RuntimeError(f"{name}: {value}".rstrip())
    return content


def bootstrap_code() -> str:
    root = str(_LIBRLM_ROOT)
    return (
        f"if {root!r} not in __import__('sys').path:\n"
        f"    __import__('sys').path.insert(0, {root!r})\n"
        "get_ipython().run_line_magic('load_ext', 'rlm.ipython_extension')\n"
    )


def execution_report(expressions: dict[str, Any]) -> dict[str, Any]:
    """Decode the kernel extension's report, evaluated after post_run_cell."""
    value = expressions.get("rlm", {})
    if value.get("status") != "ok":
        raise RuntimeError(f"RLM execution report failed: {value.get('evalue', value)}")
    return json.loads(ast.literal_eval(value["data"]["text/plain"]))


def execute(
    client: Any,
    manager: KernelManager,
    request_id: str,
    code: str,
    cwd: str,
) -> None:
    # Set execution identity out of band so a native cell magic remains on line 1.
    execute_hidden(
        client,
        manager,
        f"__import__('os').chdir({cwd!r})\n"
        f"__import__('rlm.ipython_extension', fromlist=['_']).next_execution_id = {request_id!r}",
    )
    msg_id = client.execute(
        code,
        silent=False,
        store_history=True,
        user_expressions=_REPORT,
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

    output_stream = IPythonOutputStream(emit_output)
    shell_reply = wait_for_execution(client, manager, msg_id, capture_output=output_stream.feed)
    content = shell_reply.get("content", {})
    status = str(content.get("status", "error"))
    error: dict[str, str] | None = None
    expressions = content.get("user_expressions", {})
    if status != "ok":
        error = {
            "ename": str(content.get("ename", "ExecutionError")),
            "evalue": str(content.get("evalue", "")),
        }
        # Jupyter skips user_expressions after a failed cell.
        expressions = execute_hidden(client, manager, "pass", _REPORT)["user_expressions"]
    report = execution_report(expressions)
    for execution_id in report["released"]:
        send({"type": "release", "execution_id": execution_id})
    send(
        {
            "type": "result",
            "request_id": request_id,
            "status": status,
            "execution_count": content.get("execution_count"),
            "error": error,
            "host": report["host"],
        }
    )


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
    exit_code = 0
    cleanup_errors: list[BaseException] = []

    try:
        if not os.environ.get("RLM_HOST_SOCKET") or not os.environ.get("RLM_HOST_TOKEN"):
            raise RuntimeError("RLM host socket configuration is missing")
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
        execute_hidden(client, manager, bootstrap_code())
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
                    execute(client, manager, request_id, code, request_cwd)
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
