from __future__ import annotations

import math

import pytest

from rlm.environments.ipython_protocol import (
    UNIT_RESULT,
    ChildResult,
    FinalRequest,
    GatherRequest,
    GatherResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReleaseRequest,
    RLMHostError,
    SpawnRequest,
    ValidateRequest,
    decode_request_payload,
    request_payload,
    result_from_wire,
    result_to_wire,
    retry_on_disconnect,
)


def child(text: str = "done") -> ChildResult:
    return ChildResult(
        status="ok",
        text=text,
        error=None,
        usage={"tokens": 2},
        elapsed_ms=3,
        truncated=False,
    )


@pytest.mark.parametrize(
    ("operation_request", "result"),
    [
        (SpawnRequest("spawn-id", "cell", "task", None, "/tmp"), UNIT_RESULT),
        (GatherRequest("gather-id", "cell", ("a", "a")), GatherResponse((child(), child()))),
        (ReleaseRequest("release-id", "cell", ("a",)), UNIT_RESULT),
        (
            QueryRequest("query-id", "cell", "llm", ("a", "b"), "model"),
            QueryResponse((QueryResult.success("A"), QueryResult.failure("bad"))),
        ),
        (FinalRequest("final-id", "cell", {"answer": [42, None]}), UNIT_RESULT),
        (ValidateRequest("validate-id", "cell"), UNIT_RESULT),
    ],
)
def test_typed_codecs_round_trip_every_operation(operation_request, result) -> None:
    decoded_request = decode_request_payload(
        operation_request.operation,
        operation_request.request_id,
        operation_request.execution_id,
        request_payload(operation_request),
    )
    wire_result = result_to_wire(operation_request, result)

    assert decoded_request == operation_request
    assert result_from_wire(operation_request, wire_result) == result


def test_results_and_retry_policy_are_operation_specific() -> None:
    assert result_to_wire(SpawnRequest("id", "cell", "task", None, "/tmp"), UNIT_RESULT) == {}
    assert result_to_wire(ReleaseRequest("id", "cell", ()), UNIT_RESULT) == {}
    assert result_to_wire(FinalRequest("id", "cell", None), UNIT_RESULT) == {}
    assert result_to_wire(ValidateRequest("id", "cell"), UNIT_RESULT) == {}
    assert retry_on_disconnect(QueryRequest("id", "cell", "llm", (), None)) is False
    assert all(
        retry_on_disconnect(request)
        for request in (
            SpawnRequest("spawn", "cell", "task", None, "/tmp"),
            GatherRequest("gather", "cell", ()),
            ReleaseRequest("release", "cell", ()),
            FinalRequest("final", "cell", None),
            ValidateRequest("validate", "cell"),
        )
    )


def test_child_result_is_immutable_and_copies_usage() -> None:
    usage = {"nested": {"tokens": 2}}
    result = ChildResult(
        status="ok",
        text="done",
        error=None,
        usage=usage,
        elapsed_ms=1,
        truncated=False,
    )
    usage["nested"]["tokens"] = 99
    returned = result.usage
    returned["nested"]["tokens"] = 88

    assert result.usage == {"nested": {"tokens": 2}}
    assert ChildResult.from_wire(result.to_wire()) == result


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "ok", "text": None, "error": None},
        {"status": "ok", "text": "done", "error": "bad"},
        {"status": "error", "text": None, "error": None},
        {"status": "unknown", "text": None, "error": "bad"},
    ],
)
def test_child_result_rejects_contradictory_states(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        ChildResult(usage={}, elapsed_ms=0, truncated=False, **kwargs)


def test_final_request_rejects_noncanonical_json() -> None:
    for value in ((1, 2), {1: "one"}, math.nan):
        with pytest.raises(TypeError, match="canonical JSON"):
            FinalRequest("id", "cell", value)


def test_decoders_reject_unknown_or_extra_operation_fields() -> None:
    with pytest.raises(RLMHostError, match="unknown host operation"):
        decode_request_payload("missing", "id", "cell", {})
    with pytest.raises(RLMHostError, match="invalid fields"):
        decode_request_payload(
            "spawn",
            "id",
            "cell",
            {"task": "x", "context": None, "cwd": "/tmp", "extra": True},
        )
