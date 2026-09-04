import json
import math
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Literal, TypeAlias, cast

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

ClientBackend = Literal[
    "openai",
    "portkey",
    "openrouter",
    "vercel",
    "vllm",
    "anthropic",
    "azure_openai",
    "gemini",
]
EnvironmentType = Literal["local", "ipython", "docker", "modal", "prime", "daytona", "e2b"]


def _serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, ModuleType):
        return f"<module '{value.__name__}'>"
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if callable(value):
        return f"<{type(value).__name__} '{getattr(value, '__name__', repr(value))}'>"
    # Try to convert to string for other types
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


########################################################
########    Types for LM Cost Tracking         #########
########################################################


def _non_negative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be a non-negative integer")
    return value


def _non_negative_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise TypeError(f"{label} must be a non-negative number")
    return float(value)


@dataclass
class ModelUsageSummary:
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float | None = None  # Cost in USD, if available from provider

    def to_dict(self):
        result = {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
        if self.total_cost is not None:
            result["total_cost"] = self.total_cost
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ModelUsageSummary":
        required = {"total_calls", "total_input_tokens", "total_output_tokens"}
        if not isinstance(data, dict) or not required <= data.keys():
            raise TypeError("model usage summary is missing required fields")
        if set(data) - (required | {"total_cost"}):
            raise TypeError("model usage summary has invalid fields")
        cost = data.get("total_cost")
        return cls(
            total_calls=_non_negative_int(data["total_calls"], "total_calls"),
            total_input_tokens=_non_negative_int(data["total_input_tokens"], "total_input_tokens"),
            total_output_tokens=_non_negative_int(
                data["total_output_tokens"], "total_output_tokens"
            ),
            total_cost=(None if cost is None else _non_negative_number(cost, "total_cost")),
        )


@dataclass
class UsageSummary:
    model_usage_summaries: dict[str, ModelUsageSummary]

    @classmethod
    def empty(cls) -> "UsageSummary":
        return cls(model_usage_summaries={})

    @classmethod
    def aggregate(cls, summaries: list["UsageSummary"]) -> "UsageSummary":
        merged: dict[str, ModelUsageSummary] = {}
        for summary in summaries:
            for model, usage in summary.model_usage_summaries.items():
                current = merged.get(model)
                if current is None:
                    merged[model] = ModelUsageSummary(
                        total_calls=usage.total_calls,
                        total_input_tokens=usage.total_input_tokens,
                        total_output_tokens=usage.total_output_tokens,
                        total_cost=usage.total_cost,
                    )
                    continue
                costs = [
                    cost for cost in (current.total_cost, usage.total_cost) if cost is not None
                ]
                current.total_calls += usage.total_calls
                current.total_input_tokens += usage.total_input_tokens
                current.total_output_tokens += usage.total_output_tokens
                current.total_cost = sum(costs) if costs else None
        return cls(merged)

    @property
    def total_cost(self) -> float | None:
        """Aggregate cost across all models. Returns None if no cost data available."""
        costs = [
            summary.total_cost
            for summary in self.model_usage_summaries.values()
            if summary.total_cost is not None
        ]
        return sum(costs) if costs else None

    @property
    def total_input_tokens(self) -> int:
        """Aggregate input tokens across all models."""
        return sum(summary.total_input_tokens for summary in self.model_usage_summaries.values())

    @property
    def total_output_tokens(self) -> int:
        """Aggregate output tokens across all models."""
        return sum(summary.total_output_tokens for summary in self.model_usage_summaries.values())

    def to_dict(self):
        result = {
            "model_usage_summaries": {
                model: usage_summary.to_dict()
                for model, usage_summary in self.model_usage_summaries.items()
            },
        }
        if self.total_cost is not None:
            result["total_cost"] = self.total_cost
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "UsageSummary":
        if not isinstance(data, dict) or "model_usage_summaries" not in data:
            raise TypeError("usage summary is missing model_usage_summaries")
        if set(data) - {"model_usage_summaries", "total_cost"}:
            raise TypeError("usage summary has invalid fields")
        summaries = data["model_usage_summaries"]
        if not isinstance(summaries, dict) or any(
            not isinstance(model, str) for model in summaries
        ):
            raise TypeError("model_usage_summaries must be a string-keyed object")
        summary = cls(
            model_usage_summaries={
                model: ModelUsageSummary.from_dict(usage_summary)
                for model, usage_summary in summaries.items()
            },
        )
        if "total_cost" in data:
            reported_cost = data["total_cost"]
            if reported_cost is not None:
                reported_cost = _non_negative_number(reported_cost, "total_cost")
            if reported_cost != summary.total_cost:
                raise ValueError("usage summary total_cost contradicts its model totals")
        return summary


########################################################
########   Types for REPL and RLM Iterations   #########
########################################################
_FINAL_UNSET = object()


def _same_json_shape(source: Any, canonical: Any) -> bool:
    if type(source) is not type(canonical):
        return False
    if isinstance(source, list):
        return len(source) == len(canonical) and all(
            _same_json_shape(left, right) for left, right in zip(source, canonical, strict=True)
        )
    if isinstance(source, dict):
        return source.keys() == canonical.keys() and all(
            _same_json_shape(source[key], canonical[key]) for key in source
        )
    return source == canonical


def canonical_json_value(value: Any, *, label: str) -> JSONValue:
    """Copy one value while rejecting shapes changed by JSON serialization."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        canonical = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must use canonical JSON values: {error}") from error
    if not _same_json_shape(value, canonical):
        raise TypeError(f"{label} must use canonical JSON values")
    return cast(JSONValue, canonical)


@dataclass(frozen=True, init=False)
class FinalValue:
    """Atomic presence and canonical JSON value for a final answer."""

    is_present: bool
    _value: JSONValue = field(repr=False)

    def __init__(self, is_present: bool, value: JSONValue = None) -> None:
        if type(is_present) is not bool:
            raise TypeError("final presence must be a boolean")
        if not is_present and value is not None:
            raise ValueError("an absent final cannot carry a value")
        object.__setattr__(self, "is_present", is_present)
        object.__setattr__(
            self,
            "_value",
            canonical_json_value(value, label="final value"),
        )

    @property
    def value(self) -> JSONValue:
        """Return an isolated canonical value so the final cannot mutate in place."""
        return canonical_json_value(self._value, label="final value")

    def __repr__(self) -> str:
        return f"FinalValue(is_present={self.is_present!r}, value={self.value!r})"

    @classmethod
    def absent(cls) -> "FinalValue":
        return cls(False)

    @classmethod
    def of(cls, value: JSONValue) -> "FinalValue":
        return cls(True, value)

    @classmethod
    def resolve(
        cls,
        *,
        final: "FinalValue | None" = None,
        legacy_value: JSONValue | object = _FINAL_UNSET,
    ) -> "FinalValue":
        """Resolve canonical and legacy constructor forms without collapsing null."""
        if final is not None and legacy_value is not _FINAL_UNSET:
            raise ValueError("pass final or final_answer, not both")
        if final is not None:
            if not isinstance(final, FinalValue):
                raise TypeError("final must be a FinalValue")
            return final
        if legacy_value is _FINAL_UNSET or legacy_value is None:
            return cls.absent()
        return cls.of(cast(JSONValue, legacy_value))

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the canonical presence-preserving wire form."""
        return {"has_final": self.is_present, "final_value": self.value}

    @classmethod
    def from_dict(cls, data: Any) -> "FinalValue":
        """Restore the canonical presence-preserving wire form."""
        if not isinstance(data, dict) or set(data) != {"has_final", "final_value"}:
            raise TypeError("final value has invalid fields")
        return cls(is_present=data["has_final"], value=data["final_value"])

    @classmethod
    def from_container(cls, data: dict[str, Any]) -> "FinalValue":
        """Decode nested current or flat legacy completion metadata."""
        if "final" in data:
            return cls.from_dict(data["final"])
        has_presence = "has_final" in data
        has_value = "final_value" in data
        if has_presence or has_value:
            if not has_presence or not has_value:
                raise TypeError("legacy final value requires has_final and final_value")
            return cls.from_dict(
                {"has_final": data["has_final"], "final_value": data["final_value"]}
            )
        return cls.absent()


@dataclass
class RLMChatCompletion:
    """A text completion with an optional structured RLM final."""

    root_model: str
    prompt: str | dict[str, Any] | list[dict[str, Any]]
    response: str
    usage_summary: UsageSummary
    execution_time: float
    metadata: dict | None = (
        None  # Full trajectory (run_metadata + iterations) when logger captures it
    )
    error: str | None = (
        None  # Set when this call failed; response may carry user-facing error text.
    )
    final: FinalValue = field(default_factory=FinalValue.absent)

    def __post_init__(self) -> None:
        if not isinstance(self.root_model, str) or not self.root_model:
            raise TypeError("completion root_model must be a non-empty string")
        if not isinstance(self.prompt, (str, dict, list)):
            raise TypeError("completion prompt must be a string, object, or message list")
        if not isinstance(self.response, str):
            raise TypeError("completion response must be a string")
        if not isinstance(self.usage_summary, UsageSummary):
            raise TypeError("completion usage_summary must be a UsageSummary")
        if (
            isinstance(self.execution_time, bool)
            or not isinstance(self.execution_time, (int, float))
            or not math.isfinite(self.execution_time)
            or self.execution_time < 0
        ):
            raise TypeError("completion execution_time must be a non-negative number")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("completion metadata must be an object or None")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("completion error must be a string or None")
        if not isinstance(self.final, FinalValue):
            raise TypeError("completion final must be a FinalValue")

    def to_dict(self):
        out = {
            "root_model": self.root_model,
            "prompt": self.prompt,
            "response": self.response,
            "usage_summary": self.usage_summary.to_dict(),
            "execution_time": self.execution_time,
        }
        if self.metadata is not None:
            out["metadata"] = self.metadata
        if self.error is not None:
            out["error"] = self.error
        out["final"] = self.final.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "RLMChatCompletion":
        required = {"root_model", "prompt", "response", "usage_summary", "execution_time"}
        if not isinstance(data, dict) or not required <= data.keys():
            raise TypeError("completion is missing required fields")
        allowed = required | {
            "metadata",
            "error",
            "final",
            "has_final",
            "final_value",
        }
        if set(data) - allowed:
            raise TypeError("completion has invalid fields")
        if "final" in data and ({"has_final", "final_value"} & data.keys()):
            raise TypeError("completion contains conflicting final fields")
        return cls(
            root_model=data["root_model"],
            prompt=data["prompt"],
            response=data["response"],
            usage_summary=UsageSummary.from_dict(data["usage_summary"]),
            execution_time=data["execution_time"],
            metadata=data.get("metadata"),
            error=data.get("error"),
            final=FinalValue.from_container(data),
        )


@dataclass(init=False)
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float | None
    rlm_calls: list["RLMChatCompletion"]
    final: FinalValue

    def __init__(
        self,
        stdout: str,
        stderr: str,
        locals: dict,
        execution_time: float | None = None,
        rlm_calls: list["RLMChatCompletion"] | None = None,
        final_answer: JSONValue | object = _FINAL_UNSET,
        *,
        final: FinalValue | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.locals = locals
        self.execution_time = execution_time
        self.rlm_calls = rlm_calls or []
        # Compatibility: omitted and explicit None historically meant absent.
        self.final = FinalValue.resolve(final=final, legacy_value=final_answer)

    @property
    def final_answer(self) -> JSONValue:
        return self.final.value

    @property
    def has_final_answer(self) -> bool:
        return self.final.is_present

    def __str__(self):
        return f"REPLResult(stdout={self.stdout}, stderr={self.stderr}, locals={self.locals}, execution_time={self.execution_time}, rlm_calls={len(self.rlm_calls)})"

    def to_dict(self):
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "locals": {k: _serialize_value(v) for k, v in self.locals.items()},
            "execution_time": self.execution_time,
            "rlm_calls": [call.to_dict() for call in self.rlm_calls],
            "final": self.final.to_dict(),
        }


@dataclass
class CodeBlock:
    code: str
    result: REPLResult

    def to_dict(self):
        return {"code": self.code, "result": self.result.to_dict()}


@dataclass(init=False)
class RLMIteration:
    prompt: str | dict[str, Any]
    response: str
    code_blocks: list[CodeBlock]
    final: FinalValue
    iteration_time: float | None

    def __init__(
        self,
        prompt: str | dict[str, Any],
        response: str,
        code_blocks: list[CodeBlock],
        final_answer: JSONValue | object = _FINAL_UNSET,
        iteration_time: float | None = None,
        *,
        final: FinalValue | None = None,
    ) -> None:
        self.prompt = prompt
        self.response = response
        self.code_blocks = code_blocks
        self.iteration_time = iteration_time
        # Compatibility: explicit None historically represented absence.
        self.final = FinalValue.resolve(final=final, legacy_value=final_answer)

    @property
    def final_answer(self) -> JSONValue:
        return self.final.value

    @property
    def has_final_answer(self) -> bool:
        return self.final.is_present

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "response": self.response,
            "code_blocks": [code_block.to_dict() for code_block in self.code_blocks],
            "final": self.final.to_dict(),
            "iteration_time": self.iteration_time,
        }


########################################################
########   Types for RLM Metadata   #########
########################################################


@dataclass
class RLMMetadata:
    """Metadata about the RLM configuration."""

    root_model: str
    max_depth: int
    max_iterations: int
    backend: str
    backend_kwargs: dict[str, Any]
    environment_type: str
    environment_kwargs: dict[str, Any]
    other_backends: list[str] | None = None

    def to_dict(self):
        return {
            "root_model": self.root_model,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "backend": self.backend,
            "backend_kwargs": {k: _serialize_value(v) for k, v in self.backend_kwargs.items()},
            "environment_type": self.environment_type,
            "environment_kwargs": {
                k: _serialize_value(v) for k, v in self.environment_kwargs.items()
            },
            "other_backends": self.other_backends,
        }


########################################################
########   Types for RLM Prompting   #########
########################################################


@dataclass
class QueryMetadata:
    context_lengths: list[int]
    context_total_length: int
    context_type: str

    def __init__(self, prompt: str | list[str] | dict[Any, Any] | list[dict[Any, Any]]):
        if isinstance(prompt, str):
            self.context_lengths = [len(prompt)]
            self.context_type = "str"
        elif isinstance(prompt, dict):
            self.context_type = "dict"
            self.context_lengths = []
            for chunk in prompt.values():
                if isinstance(chunk, str):
                    self.context_lengths.append(len(chunk))
                    continue
                try:
                    import json

                    self.context_lengths.append(len(json.dumps(chunk, default=str)))
                except Exception:
                    self.context_lengths.append(len(repr(chunk)))
            self.context_type = "dict"
        elif isinstance(prompt, list):
            self.context_type = "list"
            if len(prompt) == 0:
                self.context_lengths = [0]
            elif isinstance(prompt[0], dict):
                if "content" in prompt[0]:
                    self.context_lengths = [len(str(chunk.get("content", ""))) for chunk in prompt]
                else:
                    self.context_lengths = []
                    for chunk in prompt:
                        try:
                            import json

                            self.context_lengths.append(len(json.dumps(chunk, default=str)))
                        except Exception:
                            self.context_lengths.append(len(repr(chunk)))
            else:
                self.context_lengths = [len(chunk) for chunk in prompt]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        self.context_total_length = sum(self.context_lengths)
