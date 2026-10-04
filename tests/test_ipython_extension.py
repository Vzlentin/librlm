"""The in-kernel RLM extension against a real kernel and a fake harness socket."""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

import pytest

jupyter_client = pytest.importorskip("jupyter_client")

import rlm.ipython_extension as host_extension  # noqa: E402
from rlm.core.child_execution import ChildRequest  # noqa: E402
from rlm.ipython_extension import empty_child_usage, request_host_child  # noqa: E402
from rlm.prompts import load_ipython_prompt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TOKEN = "fixture-token"


class FakeHarness:
    """Answer child requests; tasks starting with 'block' wait for a disconnect."""

    def __init__(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "host.sock")
        self.requests: list[dict] = []
        self.disconnected: list[str] = []
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(connection,), daemon=True).start()

    def _handle(self, connection: socket.socket) -> None:
        with connection:
            data = b""
            while b"\n" not in data:
                data += connection.recv(65536)
            request = json.loads(data.split(b"\n", 1)[0])
            self.requests.append(request)
            if request["task"].startswith("block"):
                while connection.recv(65536):
                    pass
                self.disconnected.append(request["task"])
                return
            result = {
                "status": "ok",
                "text": f"done:{request['task']}",
                "error": None,
                "usage": empty_child_usage(),
                "elapsed_ms": 1,
                "truncated": False,
            }
            response = {"version": 2, "id": request["id"], "ok": True, "result": result}
            connection.sendall((json.dumps(response) + "\n").encode())

    def close(self) -> None:
        self.server.close()
        self.directory.cleanup()


def run(client, code: str) -> tuple[dict, str]:
    msg_id = client.execute(code, allow_stdin=False)
    output = ""
    while True:
        message = client.get_iopub_msg(timeout=30)
        if message["parent_header"].get("msg_id") != msg_id:
            continue
        content = message["content"]
        if message["msg_type"] == "stream":
            output += content["text"]
        if message["msg_type"] == "status" and content["execution_state"] == "idle":
            break
    return client.get_shell_msg(timeout=30)["content"], output


@pytest.fixture
def kernel():
    harness = FakeHarness()
    manager = jupyter_client.KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    env = dict(os.environ, RLM_HOST_SOCKET=harness.path, RLM_HOST_TOKEN=TOKEN)
    manager.start_kernel(cwd=str(REPO), env=env)
    client = manager.client()
    client.start_channels()
    client.wait_for_ready(timeout=30)
    try:
        reply, _ = run(client, "%load_ext rlm.ipython_extension")
        assert reply["status"] == "ok", reply
        yield client, harness
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
        harness.close()


def test_handles_survive_cells_and_each_cell_is_one_execution(kernel):
    client, harness = kernel
    reply, _ = run(client, "h = await rlm.spawn('first', context='ctx')")
    assert reply["status"] == "ok", reply
    reply, output = run(client, "[r] = await rlm.gather([h])\nprint(r.status, r.text)")
    assert reply["status"] == "ok", reply
    assert output.strip() == "ok done:first"
    assert [(r["task"], r["context"]) for r in harness.requests] == [("first", "ctx")]
    reply, output = run(
        client,
        "import rlm.ipython_extension as ext, json\n"
        "print(json.loads(ext.execution_report())['host']['gathered'])",
    )
    assert output.strip() == "1"


def test_failed_cell_cancels_its_children(kernel):
    client, harness = kernel
    reply, _ = run(client, "h = await rlm.spawn('block me')\nraise ValueError('fixture')")
    assert reply["status"] == "error"
    deadline = time.monotonic() + 10
    while not harness.disconnected and time.monotonic() < deadline:
        time.sleep(0.05)
    assert harness.disconnected == ["block me"]
    reply, output = run(client, "print('rlm' in globals())")
    assert output.strip() == "True"


@pytest.mark.parametrize("timeout_seconds", ["1", "0.25"])
def test_host_child_timeout_returns_error_and_closes_socket(monkeypatch, request, timeout_seconds):
    monkeypatch.setenv("RLM_HOST_CHILD_TIMEOUT_SECONDS", timeout_seconds)
    client, harness = request.getfixturevalue("kernel")
    reply, output = run(
        client,
        "import asyncio, json, time\n"
        "started = time.monotonic()\n"
        "h = await rlm.spawn('block until timeout')\n"
        "[r] = await asyncio.wait_for(rlm.gather([h]), 5)\n"
        "print(json.dumps({'status': r.status, 'error': r.error, "
        "'elapsed': time.monotonic() - started}))",
    )
    assert reply["status"] == "ok", reply
    result = json.loads(output)
    assert result["status"] == "error"
    assert result["error"] == "TimeoutError: Host child completion timed out"
    assert float(timeout_seconds) <= result["elapsed"] < 5
    deadline = time.monotonic() + 10
    while not harness.disconnected and time.monotonic() < deadline:
        time.sleep(0.05)
    assert harness.disconnected == ["block until timeout"]


@pytest.mark.parametrize("value", ["", "not-a-number", "0", "-1", "nan", "inf", "-inf"])
def test_invalid_host_child_timeout_becomes_typed_error(monkeypatch, tmp_path, value):
    monkeypatch.setenv("RLM_HOST_CHILD_TIMEOUT_SECONDS", value)
    monkeypatch.setenv("RLM_HOST_SOCKET", str(tmp_path / "host.sock"))
    monkeypatch.setenv("RLM_HOST_TOKEN", TOKEN)
    outcome = request_host_child(ChildRequest(task="invalid timeout"), threading.Event())
    assert outcome.status == "error"
    assert outcome.error is not None
    assert outcome.error.startswith("ValueError:")
    assert "RLM_HOST_CHILD_TIMEOUT_SECONDS" in outcome.error


class _ShiftedClock:
    """A `time` stand-in whose monotonic clock runs far ahead of the real one."""

    @staticmethod
    def monotonic() -> float:
        return time.monotonic() + 10_000


def test_unset_host_child_timeout_waits_until_cancelled(monkeypatch) -> None:
    harness = FakeHarness()
    try:
        monkeypatch.delenv("RLM_HOST_CHILD_TIMEOUT_SECONDS", raising=False)
        monkeypatch.setenv("RLM_HOST_SOCKET", harness.path)
        monkeypatch.setenv("RLM_HOST_TOKEN", TOKEN)
        monkeypatch.setattr(host_extension, "time", _ShiftedClock)
        cancel = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(request_host_child, ChildRequest(task="block forever"), cancel)
            with pytest.raises(FuturesTimeout):
                future.result(timeout=0.5)
            cancel.set()
            outcome = future.result(timeout=5)
        assert outcome.status == "cancelled"
        deadline = time.monotonic() + 10
        while not harness.disconnected and time.monotonic() < deadline:
            time.sleep(0.05)
        assert harness.disconnected == ["block forever"]
    finally:
        harness.close()


def test_shared_ipython_prompt_loads() -> None:
    load_ipython_prompt()


def test_named_execution_reaches_child_requests(kernel):
    client, harness = kernel
    run(client, "import rlm.ipython_extension as ext\next.next_execution_id = 'harness-cell'")
    reply, output = run(client, "[r] = await rlm.gather([await rlm.spawn('named')])\nprint(r.text)")
    assert output.strip() == "done:named", reply
    assert harness.requests[-1]["execution_id"] == "harness-cell"


def test_transport_failure_becomes_typed_child_result() -> None:
    def fail(_request, _cancel):
        raise ConnectionError("fixture disconnect")

    cancel = threading.Event()
    import rlm.ipython_extension as ext

    original = ext.request_host_child_transport
    ext.request_host_child_transport = fail
    try:
        outcome = request_host_child(object(), cancel)
        assert outcome.status == "error"
        assert outcome.error == "ConnectionError: fixture disconnect"
        assert outcome.usage_json == empty_child_usage()
        cancel.set()
        assert request_host_child(object(), cancel).status == "cancelled"
    finally:
        ext.request_host_child_transport = original


def test_missing_socket_configuration_fails_loading() -> None:
    manager = jupyter_client.KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith("RLM_HOST_")}
    manager.start_kernel(cwd=str(REPO), env=env)
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=30)
        reply, _ = run(client, "%load_ext rlm.ipython_extension")
        assert reply["status"] == "error"
        assert "socket configuration" in reply["evalue"]
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
