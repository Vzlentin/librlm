"""Tests for core types."""

import pytest

from rlm.core.types import (
    CodeBlock,
    FinalValue,
    ModelUsageSummary,
    QueryMetadata,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
    UsageSummary,
    _serialize_value,
)


class TestSerializeValue:
    """Tests for _serialize_value helper."""

    def test_primitives(self):
        assert _serialize_value(None) is None
        assert _serialize_value(True) is True
        assert _serialize_value(42) == 42
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value("hello") == "hello"

    def test_list(self):
        result = _serialize_value([1, 2, "three"])
        assert result == [1, 2, "three"]

    def test_dict(self):
        result = _serialize_value({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_callable(self):
        def my_func():
            pass

        result = _serialize_value(my_func)
        assert "function" in result.lower()
        assert "my_func" in result


class TestModelUsageSummary:
    """Tests for ModelUsageSummary."""

    def test_to_dict(self):
        summary = ModelUsageSummary(
            total_calls=10, total_input_tokens=1000, total_output_tokens=500
        )
        d = summary.to_dict()
        assert d["total_calls"] == 10
        assert d["total_input_tokens"] == 1000
        assert d["total_output_tokens"] == 500

    def test_from_dict(self):
        data = {
            "total_calls": 5,
            "total_input_tokens": 200,
            "total_output_tokens": 100,
        }
        summary = ModelUsageSummary.from_dict(data)
        assert summary.total_calls == 5
        assert summary.total_input_tokens == 200
        assert summary.total_output_tokens == 100


class TestUsageSummary:
    """Tests for UsageSummary."""

    def test_to_dict(self):
        model_summary = ModelUsageSummary(
            total_calls=1, total_input_tokens=10, total_output_tokens=5
        )
        summary = UsageSummary(model_usage_summaries={"gpt-4": model_summary})
        d = summary.to_dict()
        assert "gpt-4" in d["model_usage_summaries"]

    def test_from_dict(self):
        data = {
            "model_usage_summaries": {
                "gpt-4": {
                    "total_calls": 2,
                    "total_input_tokens": 50,
                    "total_output_tokens": 25,
                }
            }
        }
        summary = UsageSummary.from_dict(data)
        assert "gpt-4" in summary.model_usage_summaries
        assert summary.model_usage_summaries["gpt-4"].total_calls == 2


class TestREPLResult:
    """Tests for REPLResult."""

    def test_basic_creation(self):
        result = REPLResult(stdout="output", stderr="", locals={"x": 1})
        assert result.stdout == "output"
        assert result.stderr == ""
        assert result.locals == {"x": 1}

    def test_to_dict(self):
        result = REPLResult(stdout="hello", stderr="", locals={"num": 42}, execution_time=0.5)
        d = result.to_dict()
        assert d["stdout"] == "hello"
        assert d["locals"]["num"] == 42
        assert d["execution_time"] == 0.5

    def test_str_representation(self):
        result = REPLResult(stdout="test", stderr="", locals={})
        s = str(result)
        assert "REPLResult" in s
        assert "stdout=test" in s

    def test_final_presence_and_value_cannot_contradict(self):
        absent = REPLResult(stdout="", stderr="", locals={}, final_answer=None)
        explicit_null = REPLResult(stdout="", stderr="", locals={}, final=FinalValue.of(None))

        assert absent.has_final_answer is False
        assert explicit_null.has_final_answer is True
        assert explicit_null.final_answer is None
        with pytest.raises(AttributeError):
            explicit_null.has_final_answer = False
        with pytest.raises(AttributeError):
            explicit_null.final_answer = "contradiction"
        with pytest.raises(ValueError, match="absent final"):
            FinalValue(False, "contradiction")

    @pytest.mark.parametrize(
        "final",
        [FinalValue.absent(), FinalValue.of(None), FinalValue.of({"answer": 42})],
    )
    def test_final_wire_roundtrip_preserves_presence(self, final: FinalValue):
        assert FinalValue.from_dict(final.to_dict()) == final
        result = REPLResult(stdout="", stderr="", locals={}, final=final)
        assert result.to_dict()["final"] == final.to_dict()

    @pytest.mark.parametrize(
        "value",
        [
            (1, 2),
            {1: "one"},
            {"nested": (1, 2)},
            {"nested": {1: "one"}},
            float("nan"),
            float("inf"),
        ],
    )
    def test_final_rejects_noncanonical_json_shapes(self, value):
        with pytest.raises(TypeError, match="canonical JSON"):
            FinalValue.of(value)

    def test_final_copies_its_canonical_value(self):
        source = {"items": [1]}
        final = FinalValue.of(source)
        source["items"].append(2)
        returned = final.value
        returned["items"].append(3)
        assert final.value == {"items": [1]}


class TestCodeBlock:
    """Tests for CodeBlock."""

    def test_to_dict(self):
        result = REPLResult(stdout="3", stderr="", locals={"x": 3})
        block = CodeBlock(code="x = 1 + 2", result=result)
        d = block.to_dict()
        assert d["code"] == "x = 1 + 2"
        assert d["result"]["stdout"] == "3"


class TestRLMIteration:
    """Tests for RLMIteration."""

    def test_basic_creation(self):
        iteration = RLMIteration(prompt="test prompt", response="test response", code_blocks=[])
        assert iteration.prompt == "test prompt"
        assert iteration.final_answer is None

    def test_with_final_answer(self):
        iteration = RLMIteration(
            prompt="test",
            response="42",
            code_blocks=[],
            final_answer="42",
        )
        assert iteration.final_answer == "42"

    def test_explicit_null_differs_from_absent_in_metadata(self):
        absent = RLMIteration(prompt="test", response="", code_blocks=[])
        explicit_null = RLMIteration(
            prompt="test",
            response="null",
            code_blocks=[],
            final=FinalValue.of(None),
        )

        assert absent.final_answer is None
        assert explicit_null.final_answer is None
        assert absent.to_dict()["final"] != explicit_null.to_dict()["final"]

    def test_to_dict(self):
        result = REPLResult(stdout="", stderr="", locals={})
        block = CodeBlock(code="pass", result=result)
        iteration = RLMIteration(
            prompt="p",
            response="r",
            code_blocks=[block],
            iteration_time=1.5,
        )
        d = iteration.to_dict()
        assert d["prompt"] == "p"
        assert d["response"] == "r"
        assert len(d["code_blocks"]) == 1
        assert d["iteration_time"] == 1.5


class TestRLMChatCompletion:
    """Tests for RLMChatCompletion."""

    def test_metadata_default_none(self):
        usage = UsageSummary(model_usage_summaries={})
        c = RLMChatCompletion(
            root_model="gpt-4",
            prompt="hi",
            response="hello",
            usage_summary=usage,
            execution_time=1.0,
        )
        assert c.metadata is None
        d = c.to_dict()
        assert "metadata" not in d

    def test_metadata_roundtrip(self):
        usage = UsageSummary(model_usage_summaries={})
        trajectory = {"run_metadata": {"root_model": "gpt-4"}, "iterations": []}
        c = RLMChatCompletion(
            root_model="gpt-4",
            prompt="hi",
            response="hello",
            usage_summary=usage,
            execution_time=1.0,
            metadata=trajectory,
        )
        d = c.to_dict()
        assert d["metadata"] == trajectory
        c2 = RLMChatCompletion.from_dict(d)
        assert c2.metadata == trajectory

    @pytest.mark.parametrize(
        "final",
        [FinalValue.absent(), FinalValue.of(None), FinalValue.of({"answer": [42, None]})],
    )
    def test_structured_final_roundtrip_keeps_response_text_only(self, final: FinalValue):
        usage = UsageSummary(model_usage_summaries={})
        completion = RLMChatCompletion(
            root_model="gpt-4",
            prompt="hi",
            response='{"answer": [42, null]}',
            usage_summary=usage,
            execution_time=1.0,
            final=final,
        )

        restored = RLMChatCompletion.from_dict(completion.to_dict())

        assert isinstance(restored.response, str)
        assert restored.final == final

    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            ({}, FinalValue.absent()),
            ({"has_final": False, "final_value": None}, FinalValue.absent()),
            ({"has_final": True, "final_value": None}, FinalValue.of(None)),
            (
                {"has_final": True, "final_value": {"answer": 42}},
                FinalValue.of({"answer": 42}),
            ),
        ],
    )
    def test_legacy_flat_final_metadata_is_preserved(self, fields, expected):
        completion = RLMChatCompletion(
            root_model="gpt-4",
            prompt="hi",
            response="hello",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=1.0,
        )
        data = completion.to_dict()
        data.pop("final")
        data.update(fields)

        assert RLMChatCompletion.from_dict(data).final == expected

    def test_non_text_response_is_rejected(self):
        with pytest.raises(TypeError, match="response must be a string"):
            RLMChatCompletion(
                root_model="gpt-4",
                prompt="hi",
                response={"not": "text"},
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=1.0,
            )


class TestQueryMetadata:
    """Tests for QueryMetadata."""

    def test_string_prompt(self):
        meta = QueryMetadata("Hello, world!")
        assert meta.context_type == "str"
        assert meta.context_total_length == 13
        assert meta.context_lengths == [13]


class TestRLMMetadata:
    """Tests for RLMMetadata."""

    def test_to_dict(self):
        meta = RLMMetadata(
            root_model="gpt-4",
            max_depth=2,
            max_iterations=10,
            backend="openai",
            backend_kwargs={"api_key": "secret"},
            environment_type="local",
            environment_kwargs={},
        )
        d = meta.to_dict()
        assert d["root_model"] == "gpt-4"
        assert d["max_depth"] == 2
        assert d["backend"] == "openai"
