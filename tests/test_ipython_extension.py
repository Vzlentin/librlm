"""The in-kernel RLM extension against a real kernel and a fake harness socket."""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

jupyter_client = pytest.importorskip("jupyter_client")

from rlm.ipython_extension import empty_child_usage, request_host_child  # noqa: E402

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
