from __future__ import annotations

import threading
from typing import Any, Literal, cast
from unittest.mock import Mock, patch

import pytest

import rlm.core.rlm as rlm_module
from rlm import RLM
from rlm.core.accounting import CompletionTransaction, UsageLedger
from rlm.core.child_execution import ChildExecution, ChildOutcome, ChildRequest
from rlm.core.types import (
    FinalValue,
    ModelUsageSummary,
    REPLResult,
    RLMChatCompletion,
    UsageSummary,
)
from rlm.environments.base_env import CompletionFinalization
from rlm.environments.ipython_repl import IPythonREPL
from rlm.utils.exceptions import BudgetExceededError


def usage(model: str, cost: float) -> UsageSummary:
    return UsageSummary(
        model_usage_summaries={
            model: ModelUsageSummary(
                total_calls=1,
                total_input_tokens=1,
                total_output_tokens=1,
                total_cost=cost,
            )
        }
    )


def client(response: str, model: str = "root") -> Mock:
    value = Mock()
    value.model_name = model
    value.completion.return_value = response
    value.get_usage_summary.return_value = usage(model, 0.1)
    value.get_last_usage.return_value = ModelUsageSummary(
        total_calls=1,
        total_input_tokens=1,
        total_output_tokens=1,
        total_cost=0.1,
    )
    return value


def child_completion(
    child_usage: UsageSummary,
    *,
    status: Literal["ok", "error", "cancelled", "timeout"] = "ok",
    error: str | None = None,
) -> RLMChatCompletion:
    response = "done" if status == "ok" else f"Error: {error}"
    completion = RLMChatCompletion(
        root_model="child",
        prompt="child",
        response=response,
        usage_summary=child_usage,
        execution_time=0.001,
        error=error,
    )
    return completion


@pytest.mark.parametrize(
    ("failures", "winner"),
    [
        (
            {
                "body",
                "finalization",
                "handler stop",
                "environment cleanup",
                "accounting",
                "limit",
            },
            "body",
        ),
        (
            {
                "finalization",
                "handler stop",
                "environment cleanup",
                "accounting",
                "limit",
            },
            "finalization",
        ),
        (
            {"handler stop", "environment cleanup", "accounting", "limit"},
            "handler stop",
        ),
        (
            {"environment cleanup", "accounting", "limit"},
            "environment cleanup",
        ),
        ({"accounting", "limit"}, "accounting"),
        ({"limit"}, "limit"),
    ],
)
def test_completion_lifecycle_error_precedence(
    failures: set[str],
    winner: str,
) -> None:
    phases: list[str] = []

    def fail(phase: str) -> None:
        if phase in failures:
            raise RuntimeError(f"{phase} failure")

    def body() -> str:
        phases.append("body")
        fail("body")
        return "done"

    def finalize_environment() -> CompletionFinalization:
        phases.append("finalization")
        error = RuntimeError("finalization failure") if "finalization" in failures else None
        return CompletionFinalization(error=error)

    def stop_handler() -> None:
        phases.append("handler stop")
        fail("handler stop")

    def cleanup_environment() -> None:
        phases.append("environment cleanup")
        fail("environment cleanup")

    class Ledger:
        def settle(self, settled_usage: UsageSummary) -> None:
            assert settled_usage == UsageSummary.empty()
            phases.append("accounting")
            fail("accounting")

        def summary(self) -> UsageSummary:
            phases.append("summary")
            return UsageSummary.empty()

    def check_limits(final_usage: UsageSummary) -> None:
        assert final_usage == UsageSummary.empty()
        phases.append("limit")
        fail("limit")

    transaction = CompletionTransaction(
        finalize_environment=finalize_environment,
        stop_handler=stop_handler,
        cleanup_environment=cleanup_environment,
        ledger=cast(Any, Ledger()),
        check_limits=check_limits,
    )

    with pytest.raises(RuntimeError, match=rf"^{winner} failure$"):
        transaction.execute(body)

    assert phases == [
        "body",
        "finalization",
        "handler stop",
        "environment cleanup",
        "accounting",
        "summary",
        "limit",
    ]


def test_terminal_child_outcome_retains_the_original_exception() -> None:
    class ForcedInterrupt(BaseException):
        pass

    original = ForcedInterrupt("terminal")

    def operation(_request, _cancel):
        raise original

    execution = ChildExecution.from_completion_callback(
        operation,
        settle=lambda _outcome: None,
        max_concurrent=1,
    )
    outcome = execution.run(ChildRequest(task="child"))

    assert outcome.failures[0] is original
    with pytest.raises(ForcedInterrupt, match="terminal") as raised:
        outcome.unwrap_completion()
    assert raised.value is original
    with pytest.raises(ForcedInterrupt, match="terminal"):
        execution.close()


def test_terminal_child_outcome_restores_keyword_only_exception_state() -> None:
    class ForcedInterrupt(BaseException):
        def __init__(self, message: str, *, code: int) -> None:
            super().__init__(message)
            self.code = code

    def operation(_request, _cancel):
        raise ForcedInterrupt("stop", code=17)

    execution = ChildExecution.from_completion_callback(
        operation,
        settle=lambda _outcome: None,
        max_concurrent=1,
    )
    outcome = execution.run(ChildRequest(task="child"))

    with pytest.raises(ForcedInterrupt, match="stop") as raised:
        outcome.unwrap_completion()
    assert raised.value.code == 17
    with pytest.raises(ForcedInterrupt, match="stop"):
        execution.close()


def test_child_execution_settles_each_outcome_exactly_once() -> None:
    child_usage = usage("child", 0.2)
    completion = child_completion(child_usage)
    ledger = UsageLedger(None)
    calls = 0

    def operation(_request, _cancel):
        nonlocal calls
        calls += 1
        return completion

    execution = ChildExecution.from_completion_callback(
        operation,
        settle=lambda outcome: ledger.settle(cast(UsageSummary, outcome.usage)),
        max_concurrent=1,
    )

    assert execution.run(ChildRequest(task="child")).unwrap_completion().response == "done"
    assert calls == 1
    assert ledger.summary().total_cost == pytest.approx(0.2)
    execution.close()


def test_queued_cancellation_still_publishes_one_settled_outcome() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    entered: list[str] = []
    settled: list[ChildOutcome] = []

    def operation(request: ChildRequest, _cancel: threading.Event) -> RLMChatCompletion:
        entered.append(request.task)
        if request.task == "first":
            first_started.set()
            assert release_first.wait(2)
        return child_completion(UsageSummary.empty())

    execution = ChildExecution.from_completion_callback(
        operation,
        settle=settled.append,
        max_concurrent=1,
    )
    first = execution.submit(ChildRequest(task="first"))
    assert first_started.wait(1)
    second_cancel = threading.Event()
    second = execution.submit(ChildRequest(task="queued"), second_cancel)
    second_cancel.set()
    release_first.set()

    assert first.result(timeout=1).status == "ok"
    assert second.result(timeout=1).status == "cancelled"
    assert entered == ["first"]
    assert [outcome.status for outcome in settled] == ["ok", "cancelled"]
    execution.close()


def test_finalization_settles_accepted_queued_requests() -> None:
    first_started = threading.Event()
    entered: list[str] = []
    settled: list[ChildOutcome] = []

    def operation(request: ChildRequest, cancel: threading.Event) -> RLMChatCompletion:
        entered.append(request.task)
        first_started.set()
        assert cancel.wait(1)
        return child_completion(UsageSummary.empty())

    execution = ChildExecution.from_completion_callback(
        operation,
        settle=settled.append,
        max_concurrent=1,
    )
    first = execution.submit(ChildRequest(task="first"))
    assert first_started.wait(1)
    queued = execution.submit(ChildRequest(task="queued"))

    execution.finalize()

    assert first.result(timeout=1).status == "cancelled"
    assert queued.result(timeout=1).status == "cancelled"
    assert entered == ["first"]
    assert [outcome.status for outcome in settled] == ["cancelled", "cancelled"]
    execution.close()


def test_cancelled_submitted_future_still_settles_its_request() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    entered: list[str] = []
    settled: list[ChildOutcome] = []

    def operation(request: ChildRequest, _cancel: threading.Event) -> RLMChatCompletion:
        entered.append(request.task)
        first_started.set()
        assert release_first.wait(1)
        return child_completion(UsageSummary.empty())

    execution = ChildExecution.from_completion_callback(
        operation,
        settle=settled.append,
        max_concurrent=1,
    )
    first = execution.submit(ChildRequest(task="first"))
    assert first_started.wait(1)
    queued = execution.submit(ChildRequest(task="queued"))

    assert queued.cancel()
    assert [outcome.status for outcome in settled] == ["cancelled"]
    release_first.set()
    assert first.result(timeout=1).status == "ok"
    assert entered == ["first"]
    assert [outcome.status for outcome in settled] == ["cancelled", "ok"]
    execution.close()


def test_child_execution_adapters_share_one_admission_limit() -> None:
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def operation(request, _cancel):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        threading.Event().wait(0.05)
        with lock:
            in_flight -= 1
        return child_completion(UsageSummary.empty())

    first = ChildExecution.from_completion_callback(
        operation,
        settle=lambda _outcome: None,
        max_concurrent=1,
    )
    second = first.with_completion_adapter(
        operation,
        settle=lambda _outcome: None,
    )
    threads = [
        threading.Thread(target=execution.run, args=(ChildRequest(task="child"),))
        for execution in (first, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak == 1
    first.close()
    second.close()


def test_teardown_failure_propagates_after_paid_usage_settlement() -> None:
    paid = usage("child", 0.4)

    class PaidChild:
        def __init__(self, *args, **kwargs):
            self._usage_ledger = UsageLedger(kwargs["max_budget"])
            self._usage_ledger.settle(paid)

        def completion(self, prompt, root_prompt=None):
            return child_completion(paid)

        def close(self) -> None:
            raise RuntimeError("close failed")

    parent = RLM(
        backend="openai",
        backend_kwargs={"model_name": "root"},
        max_depth=2,
    )
    with (
        patch.object(rlm_module, "RLM", PaidChild),
        pytest.raises(RuntimeError, match="close failed"),
    ):
        parent.subcall("child")

    assert parent._usage_ledger.summary().total_cost == pytest.approx(0.4)


def test_teardown_failure_is_part_of_the_settled_authoritative_outcome() -> None:
    paid = usage("child", 0.4)
    completion = child_completion(paid)
    settled: list[ChildOutcome] = []

    class Runtime:
        def execute(self, _cancel):
            return completion

        def paid_usage(self):
            return paid

        def close(self):
            raise RuntimeError("close failed")

    execution = ChildExecution(
        lambda _request, _cancel: Runtime(),
        settle=settled.append,
        max_concurrent=1,
    )
    outcome = execution.run(ChildRequest(task="child"))

    assert settled == [outcome]
    assert outcome.usage == paid
    assert outcome.status == "error"
    with pytest.raises(RuntimeError, match="close failed"):
        execution.close()


def test_standalone_child_outcome_retains_arbitrary_wire_usage() -> None:
    outcome = ChildOutcome.external(
        status="ok",
        text="done",
        error=None,
        usage={"tokens": 7},
        elapsed_ms=1,
        truncated=False,
    )
    assert outcome.usage_json == {"tokens": 7}
    assert outcome.completion is None


def test_child_execution_rejects_contradictory_runtime_usage() -> None:
    completion = child_completion(usage("reported", 0.2))
    authoritative = usage("authoritative", 0.3)
    settled: list[ChildOutcome] = []

    class Runtime:
        def execute(self, _cancel):
            return completion

        def paid_usage(self):
            return authoritative

        def close(self):
            pass

    execution = ChildExecution(
        lambda _request, _cancel: Runtime(),
        settle=settled.append,
        max_concurrent=1,
    )
    outcome = execution.run(ChildRequest(task="child"))

    assert outcome.status == "error"
    assert outcome.usage == authoritative
    assert settled == [outcome]
    with pytest.raises(TypeError, match="must match"):
        execution.close()


def test_custom_child_without_typed_accounting_fails_explicitly(tmp_path) -> None:
    def runner(_request, _cancel):
        return {"text": "untyped"}

    pytest.importorskip("jupyter_client")
    with IPythonREPL(
        kernel_mode="subprocess",
        async_child_runner=cast(Any, runner),
    ) as repl:
        assert repl._spawn_children is not None
        outcome = repl._spawn_children.run(
            ChildRequest(task="child", working_dir=str(tmp_path)),
            threading.Event(),
        )
        assert outcome.status == "error"
        with pytest.raises(TypeError, match="RLMChatCompletion"):
            repl._spawn_children.finalize()


def test_released_incompatible_custom_accounting_fails_outer_completion() -> None:
    pytest.importorskip("jupyter_client")
    root_client = client(
        "```repl\nh = await rlm.spawn('child')\nawait h.release()\nawait rlm.final('done')\n```"
    )

    def runner(request: ChildRequest, cancel: threading.Event) -> object:
        return {"text": "untyped"}

    with patch("rlm.core.rlm.get_client", return_value=root_client):
        with pytest.raises(TypeError, match="RLMChatCompletion"):
            RLM(
                backend="openai",
                backend_kwargs={"model_name": "root"},
                environment="ipython",
                environment_kwargs={
                    "kernel_mode": "subprocess",
                    "async_child_runner": runner,
                },
                max_depth=1,
            ).completion("prompt")


def test_released_raised_custom_callback_fails_outer_completion() -> None:
    pytest.importorskip("jupyter_client")
    root_client = client(
        "```repl\nh = await rlm.spawn('child')\nawait h.release()\nawait rlm.final('done')\n```"
    )

    def runner(request: ChildRequest, cancel: threading.Event) -> RLMChatCompletion:
        raise RuntimeError("custom runner failed before returning accounting")

    with patch("rlm.core.rlm.get_client", return_value=root_client):
        with pytest.raises(RuntimeError, match="custom runner failed"):
            RLM(
                backend="openai",
                backend_kwargs={"model_name": "root"},
                environment="ipython",
                environment_kwargs={
                    "kernel_mode": "subprocess",
                    "async_child_runner": runner,
                },
                max_depth=1,
            ).completion("prompt")


def test_custom_child_authoritative_usage_is_charged_once() -> None:
    pytest.importorskip("jupyter_client")
    root_client = client(
        "```repl\n"
        "h = await rlm.spawn('child')\n"
        "value = (await rlm.gather([h]))[0]\n"
        "await rlm.final(value['text'])\n"
        "```"
    )
    child_usage = usage("child", 0.2)

    def runner(request: ChildRequest, cancel: threading.Event) -> RLMChatCompletion:
        return child_completion(child_usage)

    with patch.object(rlm_module, "get_client", return_value=root_client):
        completion = RLM(
            backend="openai",
            backend_kwargs={"model_name": "root"},
            environment="ipython",
            environment_kwargs={"kernel_mode": "subprocess", "async_child_runner": runner},
            max_depth=1,
        ).completion("prompt")

    assert completion.response == "done"
    assert completion.usage_summary.total_cost == pytest.approx(0.3)
    assert completion.usage_summary.model_usage_summaries["child"].total_calls == 1


def test_released_running_child_settles_before_completion_usage_snapshot() -> None:
    pytest.importorskip("jupyter_client")
    root_usage = usage("root", 0.1)
    child_usage = usage("child", 0.2)
    root_client = client(
        "```repl\n"
        "import time\n"
        "h = await rlm.spawn('child')\n"
        "time.sleep(0.05)\n"
        "await h.release()\n"
        "await rlm.final('done')\n"
        "```"
    )
    root_client.get_usage_summary.return_value = root_usage

    def runner(request: ChildRequest, cancel: threading.Event) -> RLMChatCompletion:
        assert cancel.wait(2)
        return child_completion(
            child_usage,
            status="cancelled",
            error="cancelled after billed work",
        )

    with patch("rlm.core.rlm.get_client", return_value=root_client):
        completion = RLM(
            backend="openai",
            backend_kwargs={"model_name": "root"},
            environment="ipython",
            environment_kwargs={
                "kernel_mode": "subprocess",
                "async_child_runner": runner,
            },
            max_depth=1,
        ).completion("prompt")

    assert completion.usage_summary.total_cost == pytest.approx(0.3)


def test_sequential_async_children_see_prior_paid_usage_before_admission() -> None:
    pytest.importorskip("jupyter_client")
    root_client = client(
        "```repl\n"
        "first = await rlm.spawn('first')\n"
        "await rlm.gather([first])\n"
        "second = await rlm.spawn('second')\n"
        "await rlm.gather([second])\n"
        "await rlm.final('done')\n"
        "```"
    )
    paid_usage = usage("child", 0.6)
    child_budgets: list[float | None] = []

    class PaidFailChild:
        def __init__(self, *args, **kwargs):
            child_budgets.append(kwargs["max_budget"])
            self._usage_ledger = UsageLedger(kwargs["max_budget"])
            self._usage_ledger.settle(paid_usage)

        def completion(self, prompt, root_prompt=None):
            raise BudgetExceededError(spent=0.6, budget=0.4)

        def close(self) -> None:
            pass

    parent = RLM(
        backend="openai",
        backend_kwargs={"model_name": "root"},
        environment="ipython",
        environment_kwargs={"kernel_mode": "subprocess"},
        max_depth=2,
        max_budget=0.5,
        max_concurrent_subcalls=2,
    )
    with (
        patch.object(rlm_module, "get_client", return_value=root_client),
        patch.object(rlm_module, "RLM", PaidFailChild),
        pytest.raises(BudgetExceededError) as failure,
    ):
        parent.completion("prompt")

    assert child_budgets == [pytest.approx(0.4)]
    assert failure.value.spent == pytest.approx(0.7)


def test_exceptional_persistent_completion_finalizes_before_next_reset() -> None:
    pytest.importorskip("jupyter_client")
    started = threading.Event()
    cancelled = threading.Event()
    child_usage = usage("child", 0.2)

    def runner(request: ChildRequest, cancel: threading.Event) -> RLMChatCompletion:
        started.set()
        assert cancel.wait(5)
        cancelled.set()
        return child_completion(
            child_usage,
            status="cancelled",
            error="cancelled after billed work",
        )

    first = client("```repl\nimport time\nh = await rlm.spawn('child')\ntime.sleep(0.05)\n```")
    second = client("```repl\nawait rlm.final('done')\n```")
    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "root"},
        environment="ipython",
        environment_kwargs={"kernel_mode": "subprocess", "async_child_runner": runner},
        max_depth=1,
        max_budget=0.05,
        persistent=True,
    )
    try:
        with patch.object(rlm_module, "get_client", side_effect=[first, second]):
            with pytest.raises(BudgetExceededError):
                rlm.completion("first")
            assert started.is_set()
            assert cancelled.is_set()
            assert rlm._usage_ledger.summary().total_cost == pytest.approx(0.3)

            rlm.max_budget = None
            completion = rlm.completion("second")
        assert completion.response == "done"
        assert completion.usage_summary.total_cost == pytest.approx(0.1)
    finally:
        rlm.close()


def test_body_failure_wins_after_finalization_and_final_limit_check() -> None:
    root_client = client("```repl\npass\n```")
    finalized = Mock()

    class FailingEnvironment:
        def execute_code(self, code: str):
            raise ValueError("primary")

        def finalize_completion(self):
            finalized()
            return CompletionFinalization(
                usage("child", 0.2),
                RuntimeError("secondary"),
            )

        def cleanup(self) -> None:
            pass

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "root"},
        max_depth=1,
    )
    limit_check = Mock(side_effect=RuntimeError("limit"))
    with (
        patch.object(rlm_module, "get_client", return_value=root_client),
        patch.object(rlm_module, "get_environment", return_value=FailingEnvironment()),
        patch.object(rlm, "_check_final_usage_limits", limit_check),
        pytest.raises(ValueError, match="primary"),
    ):
        rlm.completion("prompt")

    finalized.assert_called_once_with()
    limit_check.assert_called_once()
    assert rlm._usage_ledger.summary().total_cost == pytest.approx(0.3)


def test_rlm_completion_preserves_body_failure_over_cleanup_failure() -> None:
    root_client = client("```repl\npass\n```")
    cleanup_called = Mock()

    class FailingEnvironment:
        def execute_code(self, code: str) -> REPLResult:
            raise ValueError("body failed")

        def finalize_completion(self) -> CompletionFinalization:
            return CompletionFinalization()

        def cleanup(self) -> None:
            cleanup_called()
            raise RuntimeError("cleanup failed")

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "root"},
        max_depth=1,
    )
    with (
        patch.object(rlm_module, "get_client", return_value=root_client),
        patch.object(rlm_module, "get_environment", return_value=FailingEnvironment()),
        pytest.raises(ValueError, match="body failed"),
    ):
        rlm.completion("prompt")

    cleanup_called.assert_called_once_with()


def test_finalization_failure_wins_over_final_limit_failure() -> None:
    root_client = client("```repl\npass\n```")

    class FinalEnvironment:
        def execute_code(self, code: str):
            return REPLResult(
                stdout="",
                stderr="",
                locals={},
                final=FinalValue.of("done"),
            )

        def finalize_completion(self):
            return CompletionFinalization(
                usage("child", 0.2),
                RuntimeError("secondary"),
            )

        def cleanup(self) -> None:
            pass

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "root"},
        max_depth=1,
    )
    limit_check = Mock(side_effect=RuntimeError("limit"))
    with (
        patch.object(rlm_module, "get_client", return_value=root_client),
        patch.object(rlm_module, "get_environment", return_value=FinalEnvironment()),
        patch.object(rlm, "_check_final_usage_limits", limit_check),
        pytest.raises(RuntimeError, match="secondary"),
    ):
        rlm.completion("prompt")

    limit_check.assert_called_once()


def test_live_cumulative_root_usage_is_read_and_children_are_added() -> None:
    ledger = UsageLedger(None)
    current_root = usage("root", 0.1)
    ledger.bind_root(lambda: current_root)
    assert ledger.summary().total_cost == pytest.approx(0.1)

    current_root = usage("root", 0.15)
    ledger.settle(usage("child", 0.2))

    assert ledger.summary().total_cost == pytest.approx(0.35)


def test_root_usage_is_live_during_environment_setup() -> None:
    root_client = client("```repl\npass\n```")
    root_client.get_usage_summary.return_value = usage("root", 0.2)
    child_budgets: list[float | None] = []

    class SetupChild:
        def __init__(self, *args, **kwargs):
            child_budgets.append(kwargs["max_budget"])

        def completion(self, prompt, root_prompt=None):
            return RLMChatCompletion(
                root_model="child",
                prompt=prompt,
                response="setup child",
                usage_summary=UsageSummary.empty(),
                execution_time=0.0,
            )

        def close(self) -> None:
            pass

    class SetupEnvironment:
        def execute_code(self, code: str):
            return REPLResult(
                stdout="",
                stderr="",
                locals={},
                final=FinalValue.of("done"),
            )

        def finalize_completion(self):
            return CompletionFinalization()

        def cleanup(self) -> None:
            pass

    def make_environment(environment_type, kwargs):
        kwargs["subcall_fn"]("setup child")
        return SetupEnvironment()

    parent = RLM(
        backend="openai",
        backend_kwargs={"model_name": "root"},
        max_depth=2,
        max_budget=0.5,
        max_concurrent_subcalls=1,
    )
    with (
        patch.object(rlm_module, "get_client", return_value=root_client),
        patch.object(rlm_module, "get_environment", side_effect=make_environment),
        patch.object(rlm_module, "RLM", SetupChild),
    ):
        completion = parent.completion("prompt")

    assert completion.response == "done"
    assert child_budgets == [pytest.approx(0.3)]


def test_remaining_budget_is_a_snapshot_without_reservation_waiting() -> None:
    ledger = UsageLedger(1.0)

    assert ledger.remaining_budget() == 1.0
    assert ledger.remaining_budget() == 1.0
    ledger.settle(usage("child", 0.6))
    assert ledger.remaining_budget() == pytest.approx(0.4)
