"""Exercise actual Hermes discovery/dispatch. Provider calls are explicitly mocked."""

import contextvars
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    sys.path.insert(0, str(Path.home() / ".hermes/hermes-agent"))
    from agent.display import _detect_tool_failure
    from agent.plugin_llm import PluginLlmCompleteResult, PluginLlmUsage
    from hermes_cli.plugins import get_plugin_manager
    from hermes_constants import get_hermes_home

    manager = get_plugin_manager()
    manager.discover_and_load()
    from model_tools import handle_function_call
    from tools.registry import registry

    session = "ipython-model-free-native-verification"

    def call(args, **kwargs):
        raw = handle_function_call("ipython", args, session_id=session, task_id=session, **kwargs)
        result = json.loads(raw)
        assert result.get("ok"), result
        assert not _detect_tool_failure("ipython", raw)[0], raw
        return result

    seen = []
    marker = contextvars.ContextVar("fixture_marker")
    marker.set("caller context preserved")

    def mocked_complete(self, **kwargs):
        seen.append((kwargs, marker.get()))
        return PluginLlmCompleteResult(
            text="EXPLICIT MOCK PROVIDER RESULT",
            provider="mock",
            model="fixture",
            agent_id="fixture",
            usage=PluginLlmUsage(cost_usd=None),
        )

    try:
        assert registry.get_toolset_for_tool("ipython") == "ipython_rlm"
        assert call({"action": "status"})["running"] is False
        first = call({"code": "persistent_fixture = 6\nprint(persistent_fixture)", "cwd": "/tmp"})
        second = call(
            {"code": "import asyncio\nawait asyncio.sleep(0)\nprint(persistent_fixture * 7)"}
        )
        assert second["output"].strip() == "42"
        assert second["kernel_id"] == first["kernel_id"]
        failed = handle_function_call(
            "ipython",
            {"code": "raise ValueError('fixture failure ' + 'x' * 30000)"},
            session_id=session,
            task_id=session,
        )
        assert not json.loads(failed)["ok"] and _detect_tool_failure("ipython", failed)[0]
        assert json.loads(failed)["truncated"] and len(failed) <= 16000
        with patch("agent.plugin_llm.PluginLlm.complete", mocked_complete):
            denied = call(
                {"code": "h = await rlm.spawn('deny fixture')\nr = await rlm.gather([h])\nprint(r)"}
            )
            assert not seen and "budget exhausted" in denied["output"]
            child = call(
                {
                    "code": "h = await rlm.spawn('fixture task', context='fixture evidence')\nr = await rlm.gather([h])\nprint(r)",
                    "max_child_calls": 1,
                }
            )
            assert "EXPLICIT MOCK PROVIDER RESULT" in child["output"]
            assert len(seen) == 1 and seen[0][1] == "caller context preserved"
            request = seen[0][0]
            assert (
                request["messages"][1]["content"]
                == "TASK\nfixture task\n\nEVIDENCE\nfixture evidence"
            )
            assert request["max_tokens"] == 4096
            assert not {"provider", "model", "profile", "agent_id"} & set(request)
        with patch(
            "tools.terminal_tool._session_scope", return_value=SimpleNamespace(env_type="ssh")
        ):
            blocked = json.loads(
                handle_function_call("ipython", {"code": "1"}, session_id=session, task_id=session)
            )
            assert not blocked["ok"] and "sandbox" in blocked["error"]
        receipt = dict(
            native_discovery=True,
            native_dispatch=True,
            persistent_cells=True,
            top_level_await=True,
            output=second["output"].strip(),
            default_child_calls=0,
            mock_provider_calls=len(seen),
            live_provider_calls=0,
            contextvars_preserved=True,
            remote_backend_refused=True,
            success_error_classification=True,
            oversized_failure_classification=True,
            kernel_id=first["kernel_id"],
            audit_path=child["audit_path"],
        )
    finally:
        call({"action": "reset"})
        assert not call({"action": "status"})["running"]

    receipt["reset_verified"] = True
    path = get_hermes_home() / "cache/ipython-rlm/native-verification.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(receipt, receipt_path=str(path)), indent=2))


if __name__ == "__main__":
    main()
