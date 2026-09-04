from __future__ import annotations

import pytest

from rlm.core.comms_utils import LMRequest, LMResponse
from rlm.core.types import FinalValue, RLMChatCompletion, UsageSummary


def completion() -> RLMChatCompletion:
    return RLMChatCompletion(
        root_model="model",
        prompt="prompt",
        response="response",
        usage_summary=UsageSummary.empty(),
        execution_time=0.1,
        final=FinalValue.absent(),
    )


def test_lm_request_requires_exactly_one_prompt_variant() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        LMRequest()
    with pytest.raises(ValueError, match="exactly one"):
        LMRequest(prompt="one", prompts=["two"])
    with pytest.raises(TypeError, match="invalid fields"):
        LMRequest.from_dict({"prompt": "one", "depth": 0, "typo": True})


def test_lm_response_roundtrip_uses_one_explicit_variant() -> None:
    response = LMResponse.success_response(completion())

    assert LMResponse.from_dict(response.to_dict()) == response


@pytest.mark.parametrize(
    "payload",
    [
        {"error": None, "chat_completion": None, "chat_completions": None},
        {"error": "failed", "chat_completion": completion().to_dict(), "chat_completions": None},
        {"error": None, "chat_completion": None},
    ],
)
def test_lm_response_rejects_ambiguous_or_incomplete_wire_payloads(payload) -> None:
    with pytest.raises((TypeError, ValueError)):
        LMResponse.from_dict(payload)


def test_completion_decoder_rejects_missing_and_conflicting_fields() -> None:
    payload = completion().to_dict()
    del payload["usage_summary"]
    with pytest.raises(TypeError, match="missing required fields"):
        RLMChatCompletion.from_dict(payload)

    payload = completion().to_dict()
    payload.update({"has_final": False, "final_value": None})
    with pytest.raises(TypeError, match="conflicting final fields"):
        RLMChatCompletion.from_dict(payload)


def test_usage_decoder_rejects_contradictory_aggregate_cost() -> None:
    with pytest.raises(ValueError, match="contradicts"):
        UsageSummary.from_dict(
            {
                "model_usage_summaries": {},
                "total_cost": 1.0,
            }
        )
