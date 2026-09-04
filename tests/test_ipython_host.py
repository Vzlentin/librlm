from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Any

import pytest

from rlm.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary
from rlm.environments.ipython_async import (
    AsyncRLMHost,
    ChildOutcome,
    ChildRequest,
    IPythonOutputStream,
    RLMClient,
    RLMHostError,
)
from rlm.environments.ipython_queries import QueryOutcome
from tests.ipython_test_support import (
    activate,
    child_execution,
    completion,
    result,
    running_host,
)


@pytest.mark.asyncio
async def test_query_batch_preserves_order_partial_success_and_metadata() -> None:
    calls: list[tuple[list[str], str | None]] = []

    def batch_runner(
        kind: str,
        prompts: list[str],
        model: str | None,
        cancel: threading.Event,
    ) -> list[QueryOutcome]:
        assert kind == "llm"
        calls.append((list(prompts), model))
        assert not cancel.is_set()
        return [
            QueryOutcome.success(completion(prompts[0], "GOOD", marker="first")),
            QueryOutcome.failure("bad slot"),
            QueryOutcome.success(completion(prompts[2], "ALSO-GOOD", marker="third")),
        ]

    with running_host(
        lambda request, cancel: result("child"),
        query_runner=batch_runner,
        max_live_handles=1,
    ) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)

        def query() -> list[str]:
            activate(client, "cell")
            return client.llm_query_batched(["good", "bad", "also-good"], "model")

        responses = await asyncio.to_thread(query)
        summary = host.end_execution("cell", successful=True)

    assert calls == [(["good", "bad", "also-good"], "model")]
    assert responses == ["GOOD", "Error: LM query failed - bad slot", "ALSO-GOOD"]
    assert [item.response for item in summary.completions] == ["GOOD", "ALSO-GOOD"]
    assert [item.metadata for item in summary.completions] == [
        {"marker": "first"},
        {"marker": "third"},
    ]
    assert [item.usage_summary.total_output_tokens for item in summary.completions] == [2, 2]


@pytest.mark.asyncio
async def test_failed_query_retains_paid_completion_metadata() -> None:
    paid = completion("paid", "", marker="paid")
    paid.error = "provider failed after billing"

    def query_runner(
        kind: str,
        prompts: list[str],
        model: str | None,
        cancel: threading.Event,
    ) -> list[QueryOutcome]:
        return [QueryOutcome.failure(paid.error or "failed", paid)]

    with running_host(
        lambda request, cancel: result("unused"),
        query_runner=query_runner,
    ) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        response = client.llm_query("paid")
        summary = host.end_execution("cell", successful=True)

    assert response == "Error: LM query failed - provider failed after billing"
    assert summary.completions == [paid]


def test_terminal_query_failure_is_typed_before_transport() -> None:
    class ForcedInterrupt(BaseException):
        pass

    def query_runner(
        kind: str,
        prompts: list[str],
        model: str | None,
        cancel: threading.Event,
    ) -> list[QueryOutcome]:
        raise ForcedInterrupt("terminal query failure")

    with running_host(
        lambda request, cancel: result("unused"),
        query_runner=query_runner,
    ) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        response = client.llm_query("interrupt")
        host.end_execution("cell", successful=True)

    assert response == "Error: LM query failed - ForcedInterrupt: terminal query failure"
    assert "bytes read" not in response


@pytest.mark.asyncio
async def test_query_batch_exceeds_live_handle_limit_without_weakening_child_limit() -> None:
    child_gate = threading.Event()

    def child_runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        child_gate.wait(1)
        return result(request.task)

    def batch_runner(
        kind: str,
        prompts: list[str],
        model: str | None,
        cancel: threading.Event,
    ) -> list[QueryOutcome]:
        assert kind == "llm"
        return [
            QueryOutcome.success(completion(prompt, prompt.upper(), marker=str(index)))
            for index, prompt in enumerate(prompts)
        ]

    with running_host(
        child_runner,
        query_runner=batch_runner,
        max_live_handles=1,
    ) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        first = await client.spawn("first")
        with pytest.raises(Exception, match="too many live child handles"):
            await client.spawn("second")

        prompts = [f"p{index}" for index in range(17)]

        def query() -> list[str]:
            activate(client, "cell")
            return client.llm_query_batched(prompts)

        assert await asyncio.to_thread(query) == [prompt.upper() for prompt in prompts]
        child_gate.set()
        await client.gather([first])
        host.end_execution("cell", successful=True)


@pytest.mark.asyncio
async def test_ended_execution_cancels_queued_query_before_callback_entry() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_submitted = threading.Event()

    def recursive_runner(
        kind: str,
        prompts: list[str],
        model: str | None,
        cancel: threading.Event,
    ) -> list[QueryOutcome]:
        assert kind == "rlm" and len(prompts) == 1
        prompt = prompts[0]
        if prompt == "first":
            first_started.set()
            release_first.wait(1)
        else:
            second_started.set()
        return [QueryOutcome.success(completion(prompt, prompt, marker=prompt))]

    with running_host(
        lambda request, cancel: result("child"),
        query_runner=recursive_runner,
        recursive_queries=True,
        max_concurrent=1,
    ) as host:
        original_submit = host.query_scheduler.submit
        submit_count = 0
        submit_lock = threading.Lock()

        def tracked_submit(*args: Any, **kwargs: Any):
            nonlocal submit_count
            future = original_submit(*args, **kwargs)
            with submit_lock:
                submit_count += 1
                if submit_count == 2:
                    second_submitted.set()
            return future

        host.query_scheduler.submit = tracked_submit
        host.begin_execution("first-cell")
        host.begin_execution("second-cell")
        first_client = RLMClient(host.address, host.auth_token)
        second_client = RLMClient(host.address, host.auth_token)
        responses: dict[str, str] = {}

        def query(client: RLMClient, execution_id: str, prompt: str) -> None:
            activate(client, execution_id)
            responses[prompt] = client.rlm_query(prompt)

        first_thread = threading.Thread(target=query, args=(first_client, "first-cell", "first"))
        second_thread = threading.Thread(
            target=query, args=(second_client, "second-cell", "second")
        )
        first_thread.start()
        assert first_started.wait(1)
        second_thread.start()
        assert second_submitted.wait(1)

        host.end_execution("second-cell", successful=False)
        assert not second_started.is_set()
        release_first.set()
        first_thread.join(1)
        second_thread.join(1)
        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        host.end_execution("first-cell", successful=True)

    assert responses["first"] == "first"
    assert responses["second"].startswith("Error: RLM query failed -")


@pytest.mark.asyncio
async def test_recursive_batch_coordinates_outside_single_scheduler_slot() -> None:
    entered: list[str] = []

    def recursive_runner(
        kind: str,
        prompts: list[str],
        model: str | None,
        cancel: threading.Event,
    ) -> list[QueryOutcome]:
        assert kind == "rlm" and len(prompts) == 1
        prompt = prompts[0]
        entered.append(prompt)
        return [QueryOutcome.success(completion(prompt, prompt.upper(), marker=prompt))]

    with running_host(
        lambda request, cancel: result("child"),
        query_runner=recursive_runner,
        recursive_queries=True,
        max_concurrent=1,
    ) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)

        def query() -> list[str]:
            activate(client, "cell")
            return client.rlm_query_batched(["a", "b", "c"])

        responses = await asyncio.wait_for(asyncio.to_thread(query), 2)
        summary = host.end_execution("cell", successful=True)

    assert entered == ["a", "b", "c"]
    assert responses == ["A", "B", "C"]
    assert [item.response for item in summary.completions] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_host_metadata_survives_wire_result_conversion() -> None:
    usage = UsageSummary(
        model_usage_summaries={
            "fake": ModelUsageSummary(1, 2, 3, total_cost=0.2),
        }
    )
    completion = RLMChatCompletion(
        root_model="fake",
        prompt="child",
        response="done",
        usage_summary=usage,
        execution_time=0.1,
        metadata={"trajectory": "parent-only"},
    )

    def runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        return ChildOutcome.from_completion(completion, usage, elapsed_ms=1)

    with running_host(runner) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        handle = await client.spawn("child")
        await client.gather([handle])
        summary = host.end_execution("cell", successful=True)

    assert summary.completions == [completion]
    assert summary.completions[0].metadata == {"trajectory": "parent-only"}
    assert summary.usages == [usage.to_dict()]


@pytest.mark.asyncio
async def test_terminal_child_failure_is_typed_before_transport() -> None:
    class ForcedInterrupt(BaseException):
        pass

    def runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        raise ForcedInterrupt("terminal child failure")

    with pytest.raises(ForcedInterrupt, match="terminal child failure"):
        with running_host(runner) as host:
            host.begin_execution("cell")
            client = RLMClient(host.address, host.auth_token)
            activate(client, "cell")
            handle = await client.spawn("interrupt")
            value = (await client.gather([handle]))[0]
            host.end_execution("cell", successful=True)

    assert value.status == "error"
    assert value.error is not None
    assert "ForcedInterrupt: terminal child failure" in value.error
    assert "bytes read" not in value.error


@pytest.mark.asyncio
async def test_child_runner_rejects_untyped_wire_dictionary() -> None:
    def runner(request: ChildRequest, cancel: threading.Event) -> Any:
        return result("invalid").usage_json

    with pytest.raises(TypeError, match="must return a ChildOutcome"):
        with running_host(runner) as host:
            host.begin_execution("cell")
            client = RLMClient(host.address, host.auth_token)
            activate(client, "cell")
            handle = await client.spawn("child")
            value = (await client.gather([handle]))[0]
            host.end_execution("cell", successful=True)

    assert value.status == "error"
    assert value.error is not None
    assert "child callback must return a ChildOutcome" in value.error


@pytest.mark.asyncio
async def test_json_final_is_first_write_and_success_gated() -> None:
    with running_host(lambda request, cancel: result("unused")) as host:
        host.begin_execution("ok")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "ok")
        value = {"items": [1, True, None], "nested": {"answer": 42}}
        await client.final(value)
        await client.final({"replacement": True})
        success = host.end_execution("ok", successful=True)

        host.begin_execution("failed")
        activate(client, "failed")
        await client.final({"must": "not surface"})
        failed = host.end_execution("failed", successful=False)

    assert success.final.is_present is True
    assert success.final.value == value
    assert failed.final.is_present is False
    assert failed.final.value is None


def test_host_execution_exception_after_final_cannot_commit_success() -> None:
    with running_host(lambda request, cancel: result("unused")) as host:
        client = RLMClient(host.address, host.auth_token)
        with pytest.raises(RuntimeError, match="cell failed"):
            with host.execution(client, "cell") as execution:
                client.final_sync({"premature": True})
                raise RuntimeError("cell failed")

        assert execution.summary is not None
        assert execution.summary.final.is_present is False


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [(1, 2), {1: "one"}])
async def test_client_final_rejects_noncanonical_json(value: Any) -> None:
    with running_host(lambda request, cancel: result("unused")) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        with pytest.raises(TypeError, match="canonical JSON"):
            await client.final(value)
        summary = host.end_execution("cell", successful=True)

    assert summary.final.is_present is False


@pytest.mark.asyncio
async def test_failed_execution_cancels_owned_child_and_releases_handle() -> None:
    started = threading.Event()
    cancelled = threading.Event()

    def runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        started.set()
        assert cancel.wait(1)
        cancelled.set()
        raise RuntimeError("cancelled")

    with running_host(runner) as host:
        host.begin_execution("failed")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "failed")
        handle = await client.spawn("slow")
        assert started.wait(1)
        host.end_execution("failed", successful=False)
        assert cancelled.wait(1)
        while host.live_handle_count:
            await asyncio.sleep(0.01)

        host.begin_execution("next")
        activate(client, "next")
        with pytest.raises(Exception, match="unknown or already released"):
            await client.gather([handle])
        host.end_execution("next", successful=True)


@pytest.mark.asyncio
async def test_cancelled_running_child_remains_accounted_until_exit() -> None:
    started = threading.Event()
    release = threading.Event()

    def runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        started.set()
        release.wait(1)
        return result("late")

    with running_host(runner, max_live_handles=1) as host:
        host.begin_execution("failed")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "failed")
        await client.spawn("first")
        assert started.wait(1)
        host.end_execution("failed", successful=False)
        assert host.live_handle_count == 1

        host.begin_execution("next")
        activate(client, "next")
        with pytest.raises(Exception, match="too many live child handles"):
            await client.spawn("second")
        release.set()
        for _ in range(100):
            if host.live_handle_count == 0:
                break
            await asyncio.sleep(0.01)
        assert host.live_handle_count == 0
        host.end_execution("next", successful=True)


@pytest.mark.asyncio
async def test_releasing_queued_child_immediately_frees_live_capacity() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    queued_started = threading.Event()

    def runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        if request.task == "first":
            first_started.set()
            release_first.wait(2)
        else:
            queued_started.set()
        return result(request.task)

    with running_host(runner, max_concurrent=1, max_live_handles=2) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        first = await client.spawn("first")
        assert first_started.wait(1)
        queued = await client.spawn("queued")
        await queued.release()
        assert host.live_handle_count == 1
        assert not queued_started.is_set()

        replacement = await client.spawn("replacement")
        assert host.live_handle_count == 2
        await replacement.release()
        assert host.live_handle_count == 1
        await first.release()
        release_first.set()
        for _ in range(100):
            if host.live_handle_count == 0:
                break
            await asyncio.sleep(0.01)
        assert host.live_handle_count == 0
        assert not queued_started.is_set()
        host.end_execution("cell", successful=True)


@pytest.mark.asyncio
async def test_released_handles_free_capacity_without_discarding_others() -> None:
    started = threading.Event()

    def runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        if request.task == "released":
            started.set()
            assert cancel.wait(1)
        return result(request.task)

    with running_host(runner, max_live_handles=2) as host:
        host.begin_execution("origin")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "origin")
        released = await client.spawn("released")
        survivor = await client.spawn("survivor")
        assert started.wait(1)
        host.end_execution("origin", successful=True)

        host.begin_execution("next")
        activate(client, "next")
        await client.release([released])
        await client.release([released])
        for _ in range(100):
            if host.live_handle_count == 1:
                break
            await asyncio.sleep(0.01)
        assert host.live_handle_count == 1
        replacement = await client.spawn("replacement")
        values = await client.gather([survivor, replacement])
        summary = host.end_execution("next", successful=True)

    assert [value["text"] for value in values] == ["survivor", "replacement"]
    assert summary.gathered == 2


@pytest.mark.asyncio
async def test_stop_reaps_owned_worker_threads() -> None:
    started = threading.Event()

    def runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        started.set()
        cancel.wait(1)
        return result("stopped")

    host = AsyncRLMHost(child_execution(runner))
    host.start()
    host.begin_execution("cell")
    client = RLMClient(host.address, host.auth_token)
    activate(client, "cell")
    await client.spawn("x")
    assert started.wait(1)
    host.stop()
    assert not any(thread.name.startswith("rlm-subcall") for thread in threading.enumerate())


def test_stopped_host_rejects_restart_without_allocating_transport() -> None:
    host = AsyncRLMHost(child_execution(lambda request, cancel: result("unused")))
    host.start()
    host.stop()

    try:
        with pytest.raises(RLMHostError, match="stopped"):
            host.start()
        assert host._transport._server is None
        assert host._transport._thread is None
    finally:
        host._transport.stop()


def test_stop_cancels_and_joins_plain_query_runner() -> None:
    started = threading.Event()
    finished = threading.Event()

    def query_runner(
        kind: str,
        prompts: list[str],
        model: str | None,
        cancel: threading.Event,
    ) -> list[QueryOutcome]:
        assert kind == "llm"
        started.set()
        assert cancel.wait(2)
        finished.set()
        return [QueryOutcome.failure("cancelled") for _ in prompts]

    host = AsyncRLMHost(
        child_execution(lambda request, cancel: result("unused"), max_concurrent=1),
        query_runner=query_runner,
        max_concurrent=1,
    )
    host.start()
    host.begin_execution("cell")
    client = RLMClient(host.address, host.auth_token)

    def query() -> None:
        activate(client, "cell")
        client.llm_query("wait")

    worker = threading.Thread(target=query)
    worker.start()
    assert started.wait(1)
    host.stop()
    worker.join(1)

    assert finished.is_set()
    assert not worker.is_alive()
    assert not any(thread.name.startswith("rlm-query") for thread in threading.enumerate())


def test_stop_closes_stalled_protocol_connections() -> None:
    host = AsyncRLMHost(child_execution(lambda request, cancel: result("unused")))
    host.start()
    baseline = set(threading.enumerate())
    connection = socket.create_connection(host.address)
    connection.sendall(b"\x00\x00")
    try:
        for _ in range(100):
            handlers = [
                thread
                for thread in threading.enumerate()
                if thread not in baseline and "process_request_thread" in thread.name
            ]
            if handlers:
                break
            time.sleep(0.01)
        else:
            pytest.fail("stalled protocol handler did not start")

        host.stop()
        for _ in range(100):
            if not any(thread.is_alive() for thread in handlers):
                break
            time.sleep(0.01)
        assert not any(thread.is_alive() for thread in handlers)
        connection.settimeout(1)
        try:
            assert connection.recv(1) == b""
        except ConnectionResetError:
            pass
    finally:
        connection.close()
        host.stop()


def test_released_execution_replay_cache_is_bounded_and_recent() -> None:
    host = AsyncRLMHost(child_execution(lambda request, cancel: result("done")))
    try:
        for index in range(1025):
            execution_id = f"execution-{index}"
            host.begin_execution(execution_id)
            host.end_execution(execution_id, successful=True)

        assert len(host._released_execution_ids) == 1024
        with pytest.raises(RLMHostError, match="duplicate execution id"):
            host.begin_execution("execution-1024")

        host.begin_execution("execution-0")
        host.end_execution("execution-0", successful=True)
    finally:
        host.stop()


@pytest.mark.asyncio
async def test_release_callback_fires_once_for_same_cell_gather() -> None:
    releases: list[str] = []

    with running_host(lambda request, cancel: result("done"), on_release=releases.append) as host:
        host.begin_execution("same")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "same")
        handle = await client.spawn("x")
        await client.gather([handle])
        assert releases == []
        host.end_execution("same", successful=True)

    assert releases == ["same"]


@pytest.mark.asyncio
async def test_release_callback_waits_for_origin_handles() -> None:
    releases: list[str] = []

    with running_host(lambda request, cancel: result("done"), on_release=releases.append) as host:
        host.begin_execution("origin")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "origin")
        handle = await client.spawn("x")
        host.end_execution("origin", successful=True)
        assert releases == []

        host.begin_execution("gather")
        activate(client, "gather")
        await client.gather([handle])
        host.end_execution("gather", successful=True)

    assert releases.count("origin") == 1


@pytest.mark.asyncio
async def test_reset_releases_each_origin_once() -> None:
    started = threading.Event()
    finish_running = threading.Event()
    releases: list[str] = []

    def runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        if request.execution_id == "running":
            started.set()
            finish_running.wait(2)
        return result(request.task)

    with running_host(
        runner,
        max_concurrent=1,
        on_release=releases.append,
    ) as host:
        host.begin_execution("running")
        host.begin_execution("queued")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "running")
        await client.spawn("first")
        assert started.wait(1)
        activate(client, "queued")
        await client.spawn("second")

        host.reset()
        for _ in range(100):
            if releases.count("queued") == 1:
                break
            await asyncio.sleep(0.01)
        assert releases.count("queued") == 1
        assert releases.count("running") == 0

        finish_running.set()
        for _ in range(100):
            if host.live_handle_count == 0:
                break
            await asyncio.sleep(0.01)

    assert releases.count("running") == 1
    assert releases.count("queued") == 1


@pytest.mark.asyncio
async def test_spawn_submit_failure_does_not_publish_phantom_handle() -> None:
    with running_host(lambda request, cancel: result("unused")) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")

        def fail_submit(*_args: Any, **_kwargs: Any):
            raise RuntimeError("submit boom")

        host.child_execution.submit = fail_submit
        with pytest.raises(Exception, match="submit boom"):
            await client.spawn("never-started")

        assert host.live_handle_count == 0
        summary = host.end_execution("cell", successful=True)

    assert summary.spawned == 0


def test_output_stream_normalizes_clear_wait_display_and_errors() -> None:
    events: list[dict[str, Any]] = []
    stream = IPythonOutputStream(events.append)
    stream.feed("stream", {"name": "stdout", "text": "before"})
    stream.feed("clear_output", {"wait": True})
    assert events == [{"type": "text", "kind": "stdout", "text": "before"}]
    stream.feed("display_data", {"data": {"text/plain": ["af", "ter"]}})
    stream.feed("error", {"traceback": ["one", "two"]})
    assert events[1:] == [
        {"type": "clear"},
        {"type": "text", "kind": "display", "text": "after"},
        {"type": "text", "kind": "error", "text": "one\ntwo"},
    ]
