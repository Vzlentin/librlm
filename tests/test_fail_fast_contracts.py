from __future__ import annotations

from unittest.mock import patch

import pytest

from rlm import RLM
from rlm.clients.anthropic import AnthropicClient
from rlm.core.types import REPLResult
from rlm.environments.base_env import BaseEnv, FinalAnswerDict, parse_tool_entry
from rlm.environments.ipython_async import AsyncRLMHost
from rlm.environments.ipython_client import RLMClient
from rlm.environments.ipython_protocol import FailureCode, RLMHostError
from rlm.environments.ipython_repl import IPythonREPL
from rlm.environments.local_repl import LocalREPL


@pytest.mark.parametrize(
    ("kwargs", "option"),
    [
        ({"environment": "modal", "persistent": True}, "persistent=True"),
        ({"environment": "ipython", "compaction": True}, "compaction=True"),
        ({"environment": "e2b", "custom_tools": {}}, "custom_tools"),
    ],
)
def test_rlm_rejects_unsupported_environment_capabilities(kwargs, option: str) -> None:
    with pytest.raises(ValueError, match=option):
        RLM(**kwargs)


def test_declared_capability_must_match_the_environment_interface() -> None:
    class MissingToolsEnvironment(BaseEnv):
        def setup(self) -> None:
            pass

        def load_context(self, context_payload: dict | list | str) -> None:
            pass

        def execute_code(self, code: str) -> REPLResult:
            return REPLResult(stdout="", stderr="", locals={})

        def cleanup(self) -> None:
            pass

    rlm = RLM(custom_tools={"tool": object()})
    with pytest.raises(RuntimeError, match="does not implement its interface"):
        rlm._validate_environment_instance(MissingToolsEnvironment())
    rlm.close()


def test_environment_constructor_rejects_unknown_configuration() -> None:
    with pytest.raises(TypeError, match="cell_timout"):
        LocalREPL(cell_timout=1)


def test_local_cleanup_reports_filesystem_failure_and_still_clears_state() -> None:
    repl = LocalREPL()
    repl.locals["value"] = 42
    with (
        patch(
            "rlm.environments.local_repl.shutil.rmtree",
            side_effect=OSError("filesystem busy"),
        ),
        pytest.raises(OSError, match="filesystem busy"),
    ):
        repl.cleanup()

    assert repl.globals == {}
    assert repl.locals == {}
    repl.cleanup()


def test_non_openai_client_rejects_unknown_configuration() -> None:
    with pytest.raises(TypeError, match="max_retires"):
        AnthropicClient(api_key="test", max_retires=3)


def test_final_answer_ready_is_atomic_with_callback() -> None:
    def fail(_content: object) -> None:
        raise RuntimeError("capture failed")

    answer = FinalAnswerDict(on_ready=fail)
    answer["content"] = "answer"

    with pytest.raises(RuntimeError, match="capture failed"):
        answer["ready"] = True

    assert answer == {"content": "answer", "ready": False}


def test_custom_tool_description_rejects_invalid_type() -> None:
    with pytest.raises(TypeError, match="description must be a string"):
        parse_tool_entry("tool", {"tool": object(), "description": 42})


@pytest.mark.parametrize("timeout", [-1, -0.1])
def test_ipython_rejects_negative_cell_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="cell_timeout must be positive"):
        IPythonREPL(cell_timeout=timeout)


def test_ending_unknown_execution_is_an_explicit_lifecycle_error() -> None:
    host = AsyncRLMHost(None)
    try:
        with pytest.raises(RLMHostError) as raised:
            host.end_execution("missing", successful=False)
        assert raised.value.code is FailureCode.EXECUTION_INACTIVE
    finally:
        host.stop()


@pytest.mark.asyncio
async def test_spawn_without_child_capability_fails_immediately() -> None:
    host = AsyncRLMHost(None)
    host.start()
    try:
        client = RLMClient(host.address, host.auth_token)
        with host.execution(client, "cell"):
            with pytest.raises(RLMHostError, match="no child runner"):
                await client.spawn("child")
    finally:
        host.stop()
