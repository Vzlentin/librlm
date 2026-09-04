from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def openai_style_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done"))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
    )


def anthropic_response() -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(text="done")],
        usage=SimpleNamespace(input_tokens=2, output_tokens=3),
    )


def test_anthropic_forwards_sampling_args() -> None:
    from rlm.clients.anthropic import AnthropicClient

    sync = MagicMock()
    sync.messages.create.return_value = anthropic_response()
    with (
        patch("rlm.clients.anthropic.anthropic.Anthropic", return_value=sync),
        patch("rlm.clients.anthropic.anthropic.AsyncAnthropic"),
    ):
        client = AnthropicClient(
            api_key="test",
            model_name="claude",
            sampling_args={"temperature": 0.2, "top_p": 0.9},
        )
        assert client.completion("hello") == "done"

    kwargs = sync.messages.create.call_args.kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.9


def test_portkey_forwards_sampling_args() -> None:
    from rlm.clients.portkey import PortkeyClient

    sync = MagicMock()
    sync.chat.completions.create.return_value = openai_style_response()
    with (
        patch("rlm.clients.portkey.Portkey", return_value=sync),
        patch("rlm.clients.portkey.AsyncPortkey"),
    ):
        client = PortkeyClient(
            api_key="test",
            model_name="model",
            sampling_args={"temperature": 0.2},
        )
        assert client.completion("hello") == "done"

    assert sync.chat.completions.create.call_args.kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_azure_forwards_sampling_args_to_sync_and_async_clients() -> None:
    from rlm.clients.azure_openai import AzureOpenAIClient

    sync = MagicMock()
    sync.chat.completions.create.return_value = openai_style_response()
    async_client = MagicMock()
    async_client.chat.completions.create = AsyncMock(return_value=openai_style_response())
    with (
        patch("rlm.clients.azure_openai.openai.AzureOpenAI", return_value=sync),
        patch(
            "rlm.clients.azure_openai.openai.AsyncAzureOpenAI",
            return_value=async_client,
        ),
    ):
        client = AzureOpenAIClient(
            api_key="test",
            model_name="model",
            azure_endpoint="https://example.test",
            sampling_args={"temperature": 0.2},
        )
        assert client.completion("hello") == "done"
        assert await client.acompletion("hello") == "done"

    assert sync.chat.completions.create.call_args.kwargs["temperature"] == 0.2
    assert async_client.chat.completions.create.call_args.kwargs["temperature"] == 0.2


def test_gemini_forwards_sampling_args_through_generate_config() -> None:
    from rlm.clients.gemini import GeminiClient

    sdk = MagicMock()
    sdk.models.generate_content.return_value = SimpleNamespace(
        text="done",
        usage_metadata=None,
    )
    with patch("rlm.clients.gemini.genai.Client", return_value=sdk):
        client = GeminiClient(
            api_key="test",
            model_name="gemini",
            sampling_args={"temperature": 0.2},
        )
        assert client.completion("hello") == "done"

    config = sdk.models.generate_content.call_args.kwargs["config"]
    assert config.temperature == 0.2
