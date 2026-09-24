"""Shared bridge contracts that require neither Jupyter nor a model provider."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_bridge_import_and_failure_shape_under_isolated_python() -> None:
    bridge_path = Path(__file__).resolve().parents[1] / "rlm" / "bridge.py"
    script = f"""
import importlib.util
import threading
import sys
import types
jupyter = types.ModuleType("jupyter_client")
jupyter.KernelManager = object
sys.modules["jupyter_client"] = jupyter
spec = importlib.util.spec_from_file_location("shared_bridge", {str(bridge_path)!r})
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
assert (bridge.BRIDGE_PROTOCOL_VERSION, bridge.HOST_PROTOCOL_VERSION) == (6, 2)
def fail(request, cancel):
    raise ConnectionError("fixture disconnect")
bridge.request_host_child_transport = fail
cancel = threading.Event()
outcome = bridge.request_host_child(object(), cancel)
assert outcome.status == "error"
assert outcome.error == "ConnectionError: fixture disconnect"
assert outcome.usage_json == bridge.empty_child_usage()
cancel.set()
assert bridge.request_host_child(object(), cancel).status == "cancelled"
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True)
