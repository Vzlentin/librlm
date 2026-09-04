"""Pure operation values and codecs for the async IPython host channel."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Literal, TypeAlias, cast

from rlm.core.types import JSONValue, canonical_json_value

PROTOCOL_VERSION = 5


class FailureCode(StrEnum):
    """Stable machine-readable host failure states."""

    OPERATION_FAILED = "operation_failed"
    EXECUTION_INACTIVE = "execution_inactive"
    GATHER_PENDING = "gather_pending"
    MESSAGE_TOO_LARGE = "message_too_large"


class RLMHostError(RuntimeError):
    """A typed host failure that does not itself invalidate the kernel."""

    def __init__(
        self,
        message: str,
        code: FailureCode = FailureCode.OPERATION_FAILED,
    ) -> None:
        super().__init__(message)
        self.code = code


def _object(value: object, label: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RLMHostError(f"{label} must be an object")
    if set(value) != fields:
        raise RLMHostError(f"{label} has invalid fields")
    return cast(dict[str, object], value)


def _string(value: object, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise RLMHostError(f"{label} must be a {qualifier}string")
    return value


@dataclass(frozen=True, slots=True, init=False)
class ChildResult(Mapping[str, JSONValue]):
    """Validated immutable child outcome and canonical public representation."""

    status: Literal["ok", "error", "cancelled", "timeout"]
    text: str | None
    error: str | None
    _usage_json: str = field(repr=False)
    elapsed_ms: int
    truncated: bool

    def __init__(
        self,
        *,
        status: Literal["ok", "error", "cancelled", "timeout"],
        text: str | None,
        error: str | None,
        usage: dict[str, JSONValue],
        elapsed_ms: int,
        truncated: bool,
    ) -> None:
        if status not in ("ok", "error", "cancelled", "timeout"):
            raise ValueError(f"invalid child status: {status!r}")
        if text is not None and not isinstance(text, str):
            raise TypeError("child text must be a string or None")
        if error is not None and not isinstance(error, str):
            raise TypeError("child error must be a string or None")
        if not isinstance(usage, dict) or any(not isinstance(key, str) for key in usage):
            raise TypeError("child usage must be a string-keyed object")
        if type(elapsed_ms) is not int or elapsed_ms < 0:
            raise TypeError("child elapsed_ms must be a non-negative integer")
        if type(truncated) is not bool:
            raise TypeError("child truncated must be a boolean")
        if status == "ok":
            if error is not None:
                raise ValueError("a successful child cannot carry an error")
            if text is None:
                raise ValueError("a successful child must carry text")
        elif error is None or not error:
            raise ValueError("an unsuccessful child must carry an error")
        canonical_usage = canonical_json_value(usage, label="child usage")
        assert isinstance(canonical_usage, dict)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "error", error)
        object.__setattr__(
            self,
            "_usage_json",
            json.dumps(
                canonical_usage,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
        object.__setattr__(self, "elapsed_ms", elapsed_ms)
        object.__setattr__(self, "truncated", truncated)

    @property
    def usage(self) -> dict[str, JSONValue]:
        return json.loads(self._usage_json)

    def to_wire(self) -> dict[str, JSONValue]:
        return {
            "status": self.status,
            "text": self.text,
            "error": self.error,
            "usage": self.usage,
            "elapsed_ms": self.elapsed_ms,
            "truncated": self.truncated,
        }

    @classmethod
    def from_wire(cls, value: object) -> ChildResult:
        data = _object(
            value,
            "child result",
            {"status", "text", "error", "usage", "elapsed_ms", "truncated"},
        )
        return cls(
            status=cast(Literal["ok", "error", "cancelled", "timeout"], data["status"]),
            text=cast(str | None, data["text"]),
            error=cast(str | None, data["error"]),
            usage=cast(dict[str, JSONValue], data["usage"]),
            elapsed_ms=cast(int, data["elapsed_ms"]),
            truncated=cast(bool, data["truncated"]),
        )

    def __getitem__(self, key: str) -> JSONValue:
        return self.to_wire()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("status", "text", "error", "usage", "elapsed_ms", "truncated"))

    def __len__(self) -> int:
        return 6


@dataclass(frozen=True, slots=True)
class QueryResult:
    """One ordered compatibility-query result."""

    response: str | None
    error: str | None

    def __post_init__(self) -> None:
        if (self.response is None) == (self.error is None):
            raise ValueError("query result must contain exactly one of response or error")
        if self.response is not None and not isinstance(self.response, str):
            raise TypeError("query response must be a string")
        if self.error is not None and (not isinstance(self.error, str) or not self.error):
            raise TypeError("query error must be a non-empty string")

    @classmethod
    def success(cls, response: str) -> QueryResult:
        return cls(response=response, error=None)

    @classmethod
    def failure(cls, error: str) -> QueryResult:
        return cls(response=None, error=error)

    def to_wire(self) -> dict[str, JSONValue]:
        return {"response": self.response, "error": self.error}

    @classmethod
    def from_wire(cls, value: object) -> QueryResult:
        data = _object(value, "query result", {"response", "error"})
        return cls(
            response=cast(str | None, data["response"]),
            error=cast(str | None, data["error"]),
        )


@dataclass(frozen=True, slots=True)
class UnitResult:
    """The single successful value for commands with no meaningful output."""


UNIT_RESULT = UnitResult()


@dataclass(frozen=True, slots=True)
class GatherResponse:
    results: tuple[ChildResult, ...]


@dataclass(frozen=True, slots=True)
class QueryResponse:
    results: tuple[QueryResult, ...]


@dataclass(frozen=True, slots=True)
class OperationRequest:
    request_id: str
    execution_id: str
    operation: ClassVar[str]

    def __post_init__(self) -> None:
        _string(self.request_id, "request id", nonempty=True)
        _string(self.execution_id, "execution id", nonempty=True)


@dataclass(frozen=True, slots=True)
class SpawnRequest(OperationRequest):
    operation: ClassVar[str] = "spawn"
    task: str
    context: str | None
    cwd: str

    def __post_init__(self) -> None:
        OperationRequest.__post_init__(self)
        _string(self.task, "spawn task", nonempty=True)
        if not self.task.strip():
            raise RLMHostError("spawn task must not be blank")
        if self.context is not None:
            _string(self.context, "spawn context")
        _string(self.cwd, "spawn cwd", nonempty=True)


@dataclass(frozen=True, slots=True)
class GatherRequest(OperationRequest):
    operation: ClassVar[str] = "gather"
    handles: tuple[str, ...]

    def __post_init__(self) -> None:
        OperationRequest.__post_init__(self)
        if any(not isinstance(handle, str) or not handle for handle in self.handles):
            raise TypeError("gather handles must be non-empty strings")


@dataclass(frozen=True, slots=True)
class ReleaseRequest(OperationRequest):
    operation: ClassVar[str] = "release"
    handles: tuple[str, ...]

    def __post_init__(self) -> None:
        OperationRequest.__post_init__(self)
        if any(not isinstance(handle, str) or not handle for handle in self.handles):
            raise TypeError("release handles must be non-empty strings")


QueryKind: TypeAlias = Literal["llm", "rlm"]


@dataclass(frozen=True, slots=True)
class QueryRequest(OperationRequest):
    operation: ClassVar[str] = "query"
    kind: QueryKind
    prompts: tuple[str, ...]
    model: str | None

    def __post_init__(self) -> None:
        OperationRequest.__post_init__(self)
        if self.kind not in ("llm", "rlm"):
            raise RLMHostError("query kind must be 'llm' or 'rlm'")
        if any(not isinstance(prompt, str) for prompt in self.prompts):
            raise TypeError("query prompts must be strings")
        if self.model is not None:
            _string(self.model, "query model")


@dataclass(frozen=True, slots=True)
class FinalRequest(OperationRequest):
    operation: ClassVar[str] = "final"
    value: JSONValue

    def __post_init__(self) -> None:
        OperationRequest.__post_init__(self)
        object.__setattr__(self, "value", canonical_json_value(self.value, label="final value"))


@dataclass(frozen=True, slots=True)
class ValidateRequest(OperationRequest):
    operation: ClassVar[str] = "validate"


HostRequest: TypeAlias = (
    SpawnRequest | GatherRequest | ReleaseRequest | QueryRequest | FinalRequest | ValidateRequest
)
HostResponse: TypeAlias = UnitResult | GatherResponse | QueryResponse


def _handles_from_payload(
    request_type: type[GatherRequest] | type[ReleaseRequest],
    label: str,
    request_id: str,
    execution_id: str,
    payload: dict[str, object],
) -> GatherRequest | ReleaseRequest:
    data = _object(payload, label, {"handles"})
    handles = data["handles"]
    if not isinstance(handles, list):
        raise RLMHostError(f"{label} handles must be a list")
    return request_type(request_id, execution_id, tuple(cast(list[str], handles)))


def request_payload(request: OperationRequest) -> dict[str, JSONValue]:
    """Encode one typed operation without erasing its request type."""
    match request:
        case SpawnRequest(task=task, context=context, cwd=cwd):
            return {"task": task, "context": context, "cwd": cwd}
        case GatherRequest(handles=handles) | ReleaseRequest(handles=handles):
            return {"handles": list(handles)}
        case QueryRequest(kind=kind, prompts=prompts, model=model):
            return {"kind": kind, "prompts": list(prompts), "model": model}
        case FinalRequest(value=value):
            return {"value": value}
        case ValidateRequest():
            return {}
        case _:
            raise TypeError(f"unsupported host operation: {type(request).__name__}")


def decode_request_payload(
    operation: str,
    request_id: str,
    execution_id: str,
    payload: dict[str, object],
) -> HostRequest:
    """Decode exact wire fields into constructors that own value validation."""
    match operation:
        case "spawn":
            data = _object(payload, "spawn request", {"task", "context", "cwd"})
            return SpawnRequest(
                request_id,
                execution_id,
                cast(str, data["task"]),
                cast(str | None, data["context"]),
                cast(str, data["cwd"]),
            )
        case "gather":
            return _handles_from_payload(
                GatherRequest,
                "gather request",
                request_id,
                execution_id,
                payload,
            )
        case "release":
            return _handles_from_payload(
                ReleaseRequest,
                "release request",
                request_id,
                execution_id,
                payload,
            )
        case "query":
            data = _object(payload, "query request", {"kind", "prompts", "model"})
            prompts = data["prompts"]
            if not isinstance(prompts, list):
                raise RLMHostError("query prompts must be a list")
            return QueryRequest(
                request_id,
                execution_id,
                cast(QueryKind, data["kind"]),
                tuple(cast(list[str], prompts)),
                cast(str | None, data["model"]),
            )
        case "final":
            data = _object(payload, "final request", {"value"})
            return FinalRequest(request_id, execution_id, cast(JSONValue, data["value"]))
        case "validate":
            _object(payload, "validate request", set())
            return ValidateRequest(request_id, execution_id)
        case _:
            raise RLMHostError("unknown host operation")


def result_to_wire(
    request: OperationRequest,
    result: HostResponse,
) -> dict[str, JSONValue]:
    """Encode the result shape owned by one typed request."""
    match request:
        case SpawnRequest() | ReleaseRequest() | FinalRequest() | ValidateRequest():
            if not isinstance(result, UnitResult):
                raise TypeError("command handler must return UnitResult")
            return {}
        case GatherRequest():
            if not isinstance(result, GatherResponse):
                raise TypeError("gather handler must return GatherResponse")
            return {"results": [item.to_wire() for item in result.results]}
        case QueryRequest():
            if not isinstance(result, QueryResponse):
                raise TypeError("query handler must return QueryResponse")
            return {"results": [item.to_wire() for item in result.results]}
        case _:
            raise TypeError(f"unsupported host operation: {type(request).__name__}")


def result_from_wire(request: OperationRequest, value: object) -> HostResponse:
    """Decode the result shape owned by one typed request."""
    try:
        match request:
            case SpawnRequest() | ReleaseRequest() | FinalRequest() | ValidateRequest():
                _object(value, "command result", set())
                return UNIT_RESULT
            case GatherRequest():
                data = _object(value, "gather result", {"results"})
                results = data["results"]
                if not isinstance(results, list):
                    raise RLMHostError("gather results must be a list")
                return GatherResponse(tuple(ChildResult.from_wire(item) for item in results))
            case QueryRequest():
                data = _object(value, "query result", {"results"})
                results = data["results"]
                if not isinstance(results, list):
                    raise RLMHostError("query results must be a list")
                return QueryResponse(tuple(QueryResult.from_wire(item) for item in results))
            case _:
                raise TypeError(f"unsupported host operation: {type(request).__name__}")
    except RLMHostError:
        raise
    except (TypeError, ValueError) as error:
        raise RLMHostError(str(error)) from error


def retry_on_disconnect(request: OperationRequest) -> bool:
    """Queries are non-idempotent; every other operation has a stable key."""
    match request:
        case QueryRequest():
            return False
        case (
            SpawnRequest() | GatherRequest() | ReleaseRequest() | FinalRequest() | ValidateRequest()
        ):
            return True
        case _:
            raise TypeError(f"unsupported host operation: {type(request).__name__}")


def query_result_text(kind: QueryKind, result: QueryResult) -> str:
    """Render one query result identically in direct and transported modes."""
    if result.response is not None:
        return result.response
    label = "LM" if kind == "llm" else "RLM"
    return f"Error: {label} query failed - {result.error}"
