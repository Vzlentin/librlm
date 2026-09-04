from __future__ import annotations

from dataclasses import dataclass

import pytest

from rlm.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary

pytest.importorskip("IPython")

HAS_SUBPROCESS = True
try:
    import ipykernel  # noqa: F401
    import jupyter_client  # noqa: F401
except ImportError:
    HAS_SUBPROCESS = False

BOTH_MODES = pytest.mark.parametrize(
    "kernel_mode",
    [
        "in_process",
        pytest.param(
            "subprocess",
            marks=pytest.mark.skipif(
                not HAS_SUBPROCESS,
                reason="jupyter_client/ipykernel not installed",
            ),
        ),
    ],
)


@dataclass
class _FakeSubcall:
    """Record recursive calls and return canned completions."""

    responses: list[str]
    calls: list[tuple[str, str | None]] | None = None

    def __post_init__(self) -> None:
        self.calls = []
        self._index = 0

    def __call__(self, prompt: str, model: str | None = None) -> RLMChatCompletion:
        assert self.calls is not None
        self.calls.append((prompt, model))
        response = self.responses[self._index % len(self.responses)]
        self._index += 1
        usage = UsageSummary(
            model_usage_summaries={
                "fake-model": ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=0,
                    total_output_tokens=0,
                )
            }
        )
        return RLMChatCompletion(
            root_model="fake-model",
            prompt=prompt,
            response=response,
            usage_summary=usage,
            execution_time=0.001,
        )
