"""Per-completion child execution and postpaid budget accounting."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from rlm.core.types import UsageSummary
from rlm.utils.exceptions import BudgetExceededError


@dataclass(frozen=True, slots=True)
class CompletionFinalization:
    """Usage and an optional error discovered during environment finalization."""

    usage: UsageSummary = field(default_factory=UsageSummary.empty)
    error: BaseException | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.usage, UsageSummary):
            raise TypeError("CompletionFinalization.usage must be a UsageSummary")
        if self.error is not None and not isinstance(self.error, BaseException):
            raise TypeError("CompletionFinalization.error must be a BaseException or None")


class UsageLedger:
    """Own one completion's live root usage and settled child outcomes.

    The root source is cumulative and is read live. Child usage is authoritative
    per outcome and is added exactly once by the capability that receives it.
    Budget values are postpaid observations; they do not reserve provider spend.
    """

    def __init__(self, max_budget: float | None) -> None:
        self._lock = threading.Lock()
        self._max_budget = max_budget
        self._root_usage = UsageSummary.empty
        self._root_bound = False
        self._children = UsageSummary.empty()

    def reset(self, max_budget: float | None) -> None:
        """Start a new completion after all prior work has been finalized."""
        with self._lock:
            self._max_budget = max_budget
            self._root_usage = UsageSummary.empty
            self._root_bound = False
            self._children = UsageSummary.empty()

    def bind_root(self, root_usage: Callable[[], UsageSummary]) -> None:
        """Bind the completion's cumulative live root-usage source once."""
        if not callable(root_usage):
            raise TypeError("root usage source must be callable")
        with self._lock:
            if self._root_bound:
                raise RuntimeError("root usage source is already bound")
            self._root_usage = root_usage
            self._root_bound = True

    def settle(self, usage: UsageSummary) -> None:
        """Add one authoritative child or environment accounting outcome."""
        if not isinstance(usage, UsageSummary):
            raise TypeError("settled usage must be a UsageSummary")
        with self._lock:
            self._children = UsageSummary.aggregate([self._children, usage])

    def remaining_budget(self) -> float | None:
        """Return the latest observed postpaid budget without reserving it."""
        with self._lock:
            max_budget = self._max_budget
        if max_budget is None:
            return None
        spent = self.summary().total_cost or 0.0
        remaining = max_budget - spent
        if remaining <= 0:
            raise BudgetExceededError(spent=spent, budget=max_budget)
        return remaining

    def summary(self) -> UsageSummary:
        with self._lock:
            root_usage = self._root_usage
            children = UsageSummary.aggregate([self._children])
        root = root_usage()
        if not isinstance(root, UsageSummary):
            raise TypeError("root usage source must return a UsageSummary")
        return UsageSummary.aggregate([root, children])


_Result = TypeVar("_Result")
_RESULT_MISSING = object()


class CompletionTransaction(Generic[_Result]):
    """Own finalization, teardown, and accounting for one completion.

    Every phase runs even when an earlier phase fails. Error precedence is body,
    environment finalization, handler stop, environment cleanup, accounting,
    then final limit checks.
    """

    def __init__(
        self,
        finalize_environment: Callable[[], CompletionFinalization],
        stop_handler: Callable[[], None],
        cleanup_environment: Callable[[], None] | None,
        ledger: UsageLedger,
        check_limits: Callable[[UsageSummary], None],
    ) -> None:
        self._finalize_environment = finalize_environment
        self._stop_handler = stop_handler
        self._cleanup_environment = cleanup_environment
        self._ledger = ledger
        self._check_limits = check_limits

    def execute(
        self,
        body: Callable[[], _Result],
    ) -> tuple[_Result, UsageSummary]:
        result: _Result | object = _RESULT_MISSING
        body_error: BaseException | None = None
        try:
            result = body()
        except BaseException as error:
            body_error = error

        finalization: CompletionFinalization | None = None
        finalization_error: BaseException | None = None
        try:
            finalization = self._finalize_environment()
            if not isinstance(finalization, CompletionFinalization):
                raise TypeError("environment finalization must return CompletionFinalization")
            finalization_error = finalization.error
        except BaseException as error:
            finalization_error = error

        handler_stop_error: BaseException | None = None
        try:
            self._stop_handler()
        except BaseException as error:
            handler_stop_error = error

        cleanup_error: BaseException | None = None
        if self._cleanup_environment is not None:
            try:
                self._cleanup_environment()
            except BaseException as error:
                cleanup_error = error

        accounting_error: BaseException | None = None
        if finalization is not None:
            try:
                self._ledger.settle(finalization.usage)
            except BaseException as error:
                accounting_error = error

        usage: UsageSummary | None = None
        try:
            usage = self._ledger.summary()
        except BaseException as error:
            if accounting_error is None:
                accounting_error = error

        limit_error: BaseException | None = None
        if usage is not None:
            try:
                self._check_limits(usage)
            except BaseException as error:
                limit_error = error

        errors = (
            body_error,
            finalization_error,
            handler_stop_error,
            cleanup_error,
            accounting_error,
            limit_error,
        )
        for error in errors:
            if error is not None:
                raise error.with_traceback(error.__traceback__)

        assert result is not _RESULT_MISSING and usage is not None
        return cast(_Result, result), usage
