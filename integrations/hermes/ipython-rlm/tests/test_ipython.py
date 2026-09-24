"""Model-free tests: real Jupyter cells; no research or provider calls."""

import importlib.util
import json
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]


def load_plugin():
    entry = PLUGIN / "__init__.py"
    assert entry.exists(), "Native Hermes IPython plugin is not implemented"
    spec = importlib.util.spec_from_file_location("ipython_rlm_test", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_kernel_persists_and_supports_top_level_await(tmp_path):
    module = load_plugin()
    with module.Kernel(REPO, tmp_path) as kernel:
        first = kernel.execute("items = [3, 5, 7]\nprint(sum(items))", timeout=30)
        assert first["ok"], first
        assert "15" in first["output"]
        second = kernel.execute(
            "import asyncio\nawait asyncio.sleep(0)\nprint(items[-1])", timeout=10
        )
        assert second["ok"], second
        assert "7" in second["output"]
        assert second["kernel_id"] == first["kernel_id"]


def test_kernel_uses_standalone_package_and_interpreter(tmp_path):
    module = load_plugin()
    with module.Kernel(REPO, tmp_path) as kernel:
        result = kernel.execute(
            "import json, sys, rlm as library\n"
            "print(json.dumps({'library': library.__file__, 'python': sys.executable}))",
            timeout=20,
        )
        assert result["ok"], result
        import json

        runtime = json.loads(result["output"])
        assert Path(runtime["library"]).resolve() == REPO / "rlm/__init__.py"
        assert runtime["python"] == str(REPO / ".venv/bin/python")
        assert kernel.process.args[-1] == str(REPO / "rlm/bridge.py")


def test_incompatible_bridge_is_rejected_before_execution(tmp_path):
    runtime = tmp_path / "incompatible"
    (runtime / ".venv/bin").mkdir(parents=True)
    (runtime / ".venv/bin/python").symlink_to(REPO / ".venv/bin/python")
    (runtime / "rlm").mkdir()
    (runtime / "rlm/bridge.py").write_text(
        "import json, sys\n"
        "print(json.dumps({'type': 'ready', 'protocol': 99, 'host_protocol': 2}), flush=True)\n"
        "sys.stdin.readline()\n"
    )
    with pytest.raises(RuntimeError, match="Unsupported bridge protocol"):
        load_plugin().Kernel(runtime, tmp_path)


def test_clear_output_updates_visible_result_and_preserves_audit(tmp_path):
    module = load_plugin()
    with module.Kernel(REPO, tmp_path) as kernel:
        result = kernel.execute(
            "from IPython.display import clear_output\n"
            "print('obsolete fixture')\nclear_output(wait=True)\nprint('current fixture')",
            timeout=20,
        )
        assert result["ok"], result
        assert result["output"].strip() == "current fixture"
        assert "obsolete fixture" in Path(result["audit_path"]).read_text()


def test_large_final_is_bounded_in_chat_and_retained_in_audit(tmp_path):
    module = load_plugin()
    with module.Kernel(REPO, tmp_path) as kernel:
        result = kernel.execute("await rlm.final('source-17:' + 'x' * 100000)", timeout=20)
        assert result["ok"], result
        visible = module.tool_response(result)
        assert len(visible) <= 16000
        assert json.loads(visible)["truncated"]
        events = [json.loads(line) for line in Path(result["audit_path"]).read_text().splitlines()]
        final_event = next(event for event in events if event["type"] == "result")
        assert final_event["host"] == result["host"]
        assert "source-17:" + "x" * 100000 in json.dumps(final_event)


def test_real_spawn_gather_routes_only_explicit_context_to_mock_provider(tmp_path):
    module = load_plugin()
    seen = []

    def complete(task, context, timeout):
        seen.append((task, context))
        return {"text": "fixture-only result", "usage": {}, "truncated": False}

    with module.Kernel(REPO, tmp_path, complete=complete) as kernel:
        result = kernel.execute(
            "h = await rlm.spawn('inspect fixture', context='only this fixture')\n"
            "r = await rlm.gather([h])\nprint(r)",
            timeout=20,
            max_child_calls=1,
        )
        assert result["ok"], result
        assert "fixture-only result" in result["output"]
        assert seen == [("inspect fixture", "only this fixture")]
        assert result["child_calls"] == 1


def test_delayed_admission_keeps_origin_budget_context_and_single_usage(tmp_path, monkeypatch):
    import contextvars
    import threading

    module = load_plugin()
    entered, admit = threading.Event(), threading.Event()
    marker = contextvars.ContextVar("origin")
    marker.set("originating cell")
    seen = []
    usage = {"input_tokens": 31, "output_tokens": 7, "cost_usd": None}

    def complete(task, context, timeout):
        seen.append((task, context, marker.get()))
        return {"text": "source-17: retained evidence", "usage": usage}

    with module.Kernel(REPO, tmp_path, complete=complete) as kernel:
        original = kernel.complete_request

        def delayed(request):
            entered.set()
            assert admit.wait(15), "test did not release delayed admission"
            return original(request)

        monkeypatch.setattr(kernel, "complete_request", delayed)
        try:
            first = kernel.execute(
                "handle = await rlm.spawn('inspect source-17', context='source-17 evidence')",
                timeout=30,
                max_child_calls=1,
            )
            assert first["ok"], first
            assert entered.wait(5)
            assert not seen
            assert len(kernel.executions) == 1
            marker.set("later cell")
            admit.set()
            second = kernel.execute(
                "reports = await rlm.gather([handle])\nprint(reports[0].text)",
                timeout=20,
                max_child_calls=0,
            )
            assert second["ok"], second
            assert seen == [("inspect source-17", "source-17 evidence", "originating cell")]
            assert second["child_calls"] == 0
            assert second["host"]["usages"] == [usage]
            assert not kernel.executions
            third = kernel.execute("print(reports[0].text)", timeout=10)
            assert third["host"]["usages"] == []
            assert "source-17: retained evidence" in third["output"]
        finally:
            admit.set()


def test_completed_child_survives_failed_export_without_rereading(tmp_path):
    module = load_plugin()
    seen = []
    text = "source-17: exact genuine completion"

    def complete(task, context, timeout):
        seen.append((task, context))
        return {"text": text, "usage": {"cost_usd": None}}

    with module.Kernel(REPO, tmp_path, complete=complete) as kernel:
        first = kernel.execute(
            "import json\nfrom pathlib import Path\n"
            "handle = await rlm.spawn('inspect source-17', context='source-17 evidence')\n"
            "reports = await rlm.gather([handle])\n"
            "Path('checkpoint.json').write_text(json.dumps(reports))",
            timeout=20,
            max_child_calls=1,
        )
        assert not first["ok"] and first["error"]["ename"] == "TypeError", first
        second = kernel.execute(
            "Path('checkpoint.json').write_text(json.dumps([r.to_wire() for r in reports]))\n"
            "print(reports[0].text)",
            timeout=10,
            max_child_calls=0,
        )
        assert second["ok"], second
        assert second["kernel_id"] == first["kernel_id"]
        assert second["child_calls"] == 0 and len(seen) == 1
        checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
        assert checkpoint[0]["text"] == text
        assert checkpoint[0]["usage"] == {"cost_usd": None}
        assert second["host"]["usages"] == []


def test_child_audit_failure_preserves_result_and_retries_only_checkpoint(tmp_path, monkeypatch):
    import os

    module = load_plugin()
    seen = []
    original_replace = os.replace

    def complete(task, context, timeout):
        seen.append(task)
        return {"text": "source-17: paid result", "usage": {"input_tokens": 31}}

    def fail_checkpoint(source, destination):
        if "-child-" in Path(destination).name:
            raise OSError("fixture disk full after provider returned")
        return original_replace(source, destination)

    with module.Kernel(REPO, tmp_path, complete=complete) as kernel:
        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", fail_checkpoint)
            first = kernel.execute(
                "handle = await rlm.spawn('inspect source-17', context='source-17')\n"
                "reports = await rlm.gather([handle])\nprint(reports[0].text)",
                timeout=20,
                max_child_calls=1,
            )
        assert first["ok"] and "source-17: paid result" in first["output"], first
        assert len(first["uncheckpointed_children"]) == 1
        request_id = next(iter(first["uncheckpointed_children"]))
        saved = kernel.uncheckpointed_children[request_id]
        assert saved["result"]["text"] == "source-17: paid result"
        second = kernel.execute("print(reports[0].text)", timeout=10, max_child_calls=0)
        assert second["ok"] and len(seen) == 1, second
        assert second["uncheckpointed_children"] == {}
        checkpoint = [
            json.loads(line) for line in Path(saved["audit_path"]).read_text().splitlines()
        ]
        assert checkpoint[-1] == saved["result"]


def test_checkpoint_retry_replaces_partial_write_and_is_idempotent(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    module = load_plugin()
    audit = tmp_path / "fixture-child.jsonl"
    request = {"id": "req", "task": "source-17 fixture"}
    result = {"status": "ok", "text": "source-17: genuine result", "usage": {"cost_usd": None}}
    audit.write_text(json.dumps(request) + '\n{"status":"ok","text":"partial')
    kernel = module.Kernel.__new__(module.Kernel)
    kernel.admission = threading.Lock()
    kernel.checkpoint_lock = threading.Lock()
    kernel.audit_warnings = []
    kernel.uncheckpointed_children = {
        "req": {
            "execution_id": "cell",
            "audit_path": str(audit),
            "request": request,
            "result": result,
        },
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(kernel.checkpoint_child, ["req", "req"]))
    assert [json.loads(line) for line in audit.read_text().splitlines()] == [request, result]
    assert audit.stat().st_mode & 0o777 == 0o600
    assert not kernel.uncheckpointed_children and not kernel.audit_warnings
    assert list(tmp_path.iterdir()) == [audit]


def test_sessions_are_isolated_and_reset_is_explicit(tmp_path):
    module = load_plugin()
    assert hasattr(module, "Sessions"), "Missing chat-scoped tool interface"
    sessions = module.Sessions(REPO, tmp_path / "audit")
    try:
        a = sessions.handle({"code": "secret_fixture = 41", "cwd": str(tmp_path)}, session_id="a")
        b = sessions.handle(
            {"code": "print('secret_fixture' in globals())", "cwd": str(tmp_path)}, session_id="b"
        )
        assert a["ok"] and b["ok"], (a, b)
        assert "False" in b["output"]
        assert a["kernel_id"] != b["kernel_id"]
        again = sessions.handle({"code": "print(secret_fixture + 1)"}, session_id="a")
        assert "42" in again["output"]
        assert again["kernel_id"] == a["kernel_id"]
        reset = sessions.handle({"action": "reset"}, session_id="a")
        assert reset["ok"] and reset["state_lost"]
        new = sessions.handle(
            {"code": "print('secret_fixture' in globals())", "cwd": str(tmp_path)}, session_id="a"
        )
        assert new["kernel_id"] != a["kernel_id"]
        assert "False" in new["output"]
    finally:
        sessions.close()


def test_register_exposes_native_tool_and_session_finalize_cleanup():
    module = load_plugin()
    assert hasattr(module, "register"), "Missing Hermes plugin registration"

    class Context:
        def __init__(self):
            self.tools, self.hooks = {}, {}

        def get_config(self, key, default=None):
            return default

        def register_tool(self, **kwargs):
            self.tools[kwargs["name"]] = kwargs

        def register_hook(self, name, callback):
            self.hooks[name] = callback

    import sys

    sys.path.insert(0, str(Path.home() / ".hermes/hermes-agent"))
    ctx = Context()
    module.register(ctx)
    assert "ipython" in ctx.tools
    assert "max_child_calls" in ctx.tools["ipython"]["schema"]["parameters"]["properties"]
    assert "on_session_finalize" in ctx.hooks
    assert "on_session_end" not in ctx.hooks  # Fires every turn: must not destroy state.
    from rlm.prompts import ipython_description

    assert ctx.tools["ipython"]["schema"]["description"].startswith(ipython_description() + "\n\n")


def test_native_cleanup_hooks_close_only_the_addressed_chat(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(Path.home() / ".hermes/hermes-agent"))
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    module = load_plugin()
    sessions = module.Sessions(REPO, tmp_path / "audit")
    monkeypatch.setattr(module, "Sessions", lambda *args: sessions)

    class Context:
        def __init__(self):
            self.hooks = {}

        def get_config(self, key, default=None):
            return default

        def register_tool(self, **kwargs):
            pass

        def register_hook(self, name, callback):
            self.hooks[name] = callback

    ctx = Context()
    module.register(ctx)
    try:
        for session_id in ("a", "b"):
            result = sessions.handle(
                {"code": "marker = 17", "cwd": str(tmp_path)}, session_id=session_id
            )
            assert result["ok"], result
        first = sessions.kernels["a"]
        ctx.hooks["on_session_finalize"](session_id="a")
        assert "a" not in sessions.kernels and first.process.poll() is not None
        second = sessions.handle({"code": "print(marker)"}, session_id="b")
        assert second["ok"] and second["output"].strip() == "17"
        remaining = sessions.kernels["b"]
        ctx.hooks["on_session_reset"](session_id="b")
        assert not sessions.kernels and remaining.process.poll() is not None
    finally:
        sessions.close()


def test_default_budget_blocks_calls_and_positive_budget_is_hard(tmp_path):
    module = load_plugin()
    seen = []

    def complete(task, context, timeout):
        seen.append(task)
        return {"text": "MOCK ONLY", "usage": {}}

    with module.Kernel(REPO, tmp_path, complete=complete) as kernel:
        result = kernel.execute(
            "h = await rlm.spawn('blocked')\nr = await rlm.gather([h])\nprint(r)", timeout=10
        )
        assert "budget exhausted" in result["output"], result
        assert seen == [] and result["child_calls"] == 0
        result = kernel.execute(
            "hs = [await rlm.spawn(str(i)) for i in range(3)]\nr = await rlm.gather(hs)\nprint(r)",
            timeout=10,
            max_child_calls=1,
        )
        assert len(seen) == 1 and result["child_calls"] == 1
        assert "budget exhausted" in result["output"]


def test_error_retains_previous_state_and_large_output_is_saved(tmp_path):
    import json

    module = load_plugin()
    with module.Kernel(REPO, tmp_path) as kernel:
        error = kernel.execute("retained = 8\nraise ValueError('fixture error')", timeout=10)
        assert not error["ok"] and error["error"]["ename"] == "ValueError"
        result = kernel.execute("print(retained)\nprint('x' * 20000)", timeout=10)
        assert result["ok"] and result["truncated"] and result["output"].startswith("8")
        assert len(result["output"]) <= 12000
        events = [json.loads(x) for x in Path(result["audit_path"]).read_text().splitlines()]
        complete = "".join(x.get("text", "") for x in events if x["type"] == "output")
        assert complete.count("x") == 20000
        assert Path(result["audit_path"]).stat().st_mode & 0o777 == 0o600


def test_timeout_discards_state_without_replaying_and_reaps_bridge(tmp_path):
    module = load_plugin()
    sessions = module.Sessions(REPO, tmp_path / "audit")
    try:
        sessions.handle({"code": "marker = 1", "cwd": str(tmp_path)}, session_id="a")
        kernel = sessions.kernels["a"]
        result = sessions.handle(
            {"code": "import time\ntime.sleep(30)", "timeout": 1}, session_id="a"
        )
        assert not result["ok"] and result["state_lost"], result
        assert kernel.process.poll() is not None
        retry = sessions.handle({"code": "print(marker)"}, session_id="a")
        assert not retry["ok"] and retry["state_lost"]
        assert "reset" in retry["error"].lower()
    finally:
        sessions.close()


@pytest.mark.parametrize(
    "args",
    [
        {"code": ""},
        {"code": "1", "timeout": float("nan")},
        {"code": "1", "timeout": True},
        {"code": "1", "max_child_calls": -1},
        {"code": "1", "max_child_calls": 5},
        {"code": "1", "max_child_calls": True},
        {"action": "bogus"},
        {"code": "1", "cwd": "relative"},
    ],
)
def test_invalid_parameters_never_start_kernel(tmp_path, args):
    sessions = load_plugin().Sessions(REPO, tmp_path)
    assert not sessions.handle(args, session_id="fixture")["ok"]
    assert sessions.kernels == {}
    assert not sessions.handle({"code": "1"})["ok"]
