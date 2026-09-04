"""Shared recursive and plain-query coordination for both IPython modes."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from rlm.core.child_execution import ChildOutcome, ChildRequest
from rlm.core.comms_utils import send_lm_request_batched
from rlm.core.types import RLMChatCompletion
from rlm.environments.ipython_protocol import QueryKind, QueryResult


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """One ordered query result plus any paid parent-only completion."""

    result: QueryResult
    completion: RLMChatCompletion | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, QueryResult):
            raise TypeError("QueryOutcome.result must be a QueryResult")
        if self.result.response is not None:
            if self.completion is None or self.completion.response != self.result.response:
                raise ValueError("a successful query outcome must retain its completion")
        elif self.completion is not None and self.completion.error is None:
            raise ValueError("a failed query outcome completion must report an error")

    @classmethod
    def success(cls, completion: RLMChatCompletion) -> QueryOutcome:
        return cls(QueryResult.success(completion.response), completion)

    @classmethod
    def failure(
        cls,
        error: str,
        completion: RLMChatCompletion | None = None,
    ) -> QueryOutcome:
        return cls(QueryResult.failure(error), completion)


ChildQueryRunner = Callable[[ChildRequest, threading.Event], ChildOutcome]


def child_query_outcome(outcome: ChildOutcome) -> QueryOutcome:
    """Project one settled child outcome into compatibility-query delivery."""
    if outcome.completion is not None and not outcome.failures:
        return QueryOutcome.success(outcome.completion)
    message = outcome.error or "child execution failed"
    return QueryOutcome.failure(message, outcome.completion)


class QueryCoordinator:
    """Route LM queries and delegate recursive policy to one mode executor."""

    def __init__(
        self,
        *,
        lm_handler_address: Callable[[], tuple[str, int] | None],
        depth: int,
        child_execution: ChildQueryRunner | None,
    ) -> None:
        self._lm_handler_address = lm_handler_address
        self._depth = depth
        self._child_execution = child_execution

    @property
    def has_recursive_queries(self) -> bool:
        return self._child_execution is not None

    def run(
        self,
        kind: QueryKind,
        prompts: list[str],
        model: str | None,
        cancel: threading.Event,
    ) -> list[QueryOutcome]:
        """Return one ordered outcome for every prompt."""
        if kind == "rlm" and self._child_execution is not None:
            return [
                child_query_outcome(
                    self._child_execution(
                        ChildRequest(task=prompt, model=model),
                        cancel,
                    )
                )
                for prompt in prompts
            ]
        return self._lm_outcomes(prompts, model, cancel)

    def _lm_outcomes(
        self,
        prompts: list[str],
        model: str | None,
        cancel: threading.Event,
    ) -> list[QueryOutcome]:
        if cancel.is_set():
            return [QueryOutcome.failure("query cancelled before startup") for _ in prompts]
        address = self._lm_handler_address()
        if address is None:
            return [QueryOutcome.failure("No LM handler configured") for _ in prompts]
        try:
            responses = send_lm_request_batched(
                address,
                cast(list[str | dict[str, Any]], prompts),
                model=model,
                depth=self._depth,
            )
        except Exception as error:
            message = str(error) or type(error).__name__
            return [QueryOutcome.failure(message) for _ in prompts]

        outcomes: list[QueryOutcome] = []
        for response in responses:
            completion = response.chat_completion
            if not response.success:
                outcomes.append(
                    QueryOutcome.failure(
                        response.error or "LM query failed",
                        completion,
                    )
                )
            elif completion is None:
                outcomes.append(QueryOutcome.failure("LM query returned no completion"))
            else:
                outcomes.append(QueryOutcome.success(completion))
        return outcomes
