from __future__ import annotations

import asyncio
import contextvars
import threading
from typing import Any, cast

import pytest

from rlm.environments.ipython_async import AsyncRLMHost, ChildOutcome, ChildRequest
from rlm.environments.ipython_client import RLMAnswerDict, RLMClient, RLMHandle
from rlm.environments.ipython_protocol import RLMHostError, SpawnRequest
from rlm.environments.ipython_queries import QueryOutcome
from tests.ipython_test_support import (
    ProtocolFaultProxy,
    activate,
    child_execution,
    completion,
    result,
    running_host,
)


@pytest.mark.asyncio
async def test_spawn_response_loss_retries_one_stable_handle() -> None:
    calls: list[str] = []

    def runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        calls.append(request.task)
        return result(request.task)

    with running_host(runner) as host:
        dropped = False

        def drop_first_spawn(request: dict[str, Any], _response: dict[str, Any]) -> bool:
            nonlocal dropped
            if request.get("op") == "spawn" and not dropped:
                dropped = True
                return True
            return False

        with ProtocolFaultProxy(host.address, drop_response=drop_first_spawn) as proxy:
            host.begin_execution("cell")
            client = RLMClient(proxy.address, host.auth_token)
            activate(client, "cell")
            handle = await client.spawn("once")
            value = (await client.gather([handle]))[0]
            summary = host.end_execution("cell", successful=True)

    assert dropped is True
    assert value.text == "once"
    assert len(calls) == 1
    assert summary.spawned == 1
    assert summary.gathered == 1
    assert host.live_handle_count == 0


def test_non_idempotent_query_is_not_retried_after_response_loss() -> None:
    calls = 0
    dropped = False

    def query_runner(
        kind: str,
        prompts: list[str],
        model: str | None,
        cancel: threading.Event,
    ) -> list[QueryOutcome]:
        nonlocal calls
        assert kind == "llm"
        calls += 1
        assert not cancel.is_set()
        return [
            QueryOutcome.success(completion(prompt, "DONE", marker="query")) for prompt in prompts
        ]

    def drop_first_query(request: dict[str, Any], _response: dict[str, Any]) -> bool:
        nonlocal dropped
        if request.get("op") == "query" and not dropped:
            dropped = True
            return True
        return False

    with running_host(
        lambda request, cancel: result("unused"),
        query_runner=query_runner,
    ) as host:
        with ProtocolFaultProxy(host.address, drop_response=drop_first_query) as proxy:
            host.begin_execution("cell")
            client = RLMClient(proxy.address, host.auth_token)
            activate(client, "cell")
            response = client.llm_query("once")
            summary = host.end_execution("cell", successful=True)

    assert dropped is True
    assert response.startswith("Error: LM query failed - RLM host connection failed:")
    assert calls == 1
    assert [item.response for item in summary.completions] == ["DONE"]


@pytest.mark.asyncio
async def test_spawn_idempotency_key_rejects_different_input() -> None:
    with running_host(lambda request, cancel: result("done")) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        request = SpawnRequest("stable", "cell", "first", None, "/tmp")
        await client.channel.request(request)
        await client.channel.request(request)
        with pytest.raises(RLMHostError, match="different input"):
            await client.channel.request(SpawnRequest("stable", "cell", "second", None, "/tmp"))
        await client.release([RLMHandle(client, "stable")])
        summary = host.end_execution("cell", successful=True)

    assert summary.spawned == 1


@pytest.mark.asyncio
async def test_gather_preserves_order_and_duplicate_positions_but_attributes_once() -> None:
    with running_host(lambda request, cancel: result(request.task, {"tokens": 7})) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        first, second = await asyncio.gather(client.spawn("first"), client.spawn("second"))
        values = await client.gather([second, first, second])
        summary = host.end_execution("cell", successful=True)

    assert [value.text for value in values] == ["second", "first", "second"]
    assert summary.gathered == 2
    assert summary.usages == [{"tokens": 7}, {"tokens": 7}]
    assert host.live_handle_count == 0


@pytest.mark.asyncio
async def test_handle_survives_successful_origin_cell() -> None:
    gate = threading.Event()

    def runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        gate.wait(1)
        return result(request.task)

    with running_host(runner) as host:
        host.begin_execution("origin")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "origin")
        handle = await client.spawn("cross-cell")
        origin = host.end_execution("origin", successful=True)

        host.begin_execution("next")
        activate(client, "next")
        gate.set()
        value = (await client.gather([handle]))[0]
        next_summary = host.end_execution("next", successful=True)

    assert value.text == "cross-cell"
    assert origin.spawned == 1
    assert next_summary.gathered == 1


@pytest.mark.asyncio
async def test_gather_recovers_after_committed_response_loss_with_single_usage() -> None:
    with running_host(lambda request, cancel: result("done", {"tokens": 11})) as host:
        dropped = False

        def drop_first_gather(request: dict[str, Any], _response: dict[str, Any]) -> bool:
            nonlocal dropped
            if request.get("op") == "gather" and not dropped:
                dropped = True
                return True
            return False

        with ProtocolFaultProxy(host.address, drop_response=drop_first_gather) as proxy:
            host.begin_execution("cell")
            client = RLMClient(proxy.address, host.auth_token)
            activate(client, "cell")
            handle = await client.spawn("recover")
            values = await client.gather([handle])
            summary = host.end_execution("cell", successful=True)

    assert dropped is True
    assert values[0].text == "done"
    assert summary.gathered == 1
    assert summary.usages == [{"tokens": 11}]
    assert host.live_handle_count == 0


@pytest.mark.asyncio
async def test_committed_delivery_recovers_in_later_cell_after_all_responses_are_lost() -> None:
    with running_host(lambda request, cancel: result("done", {"tokens": 5})) as host:
        lose_gather = True

        def drop_gathers(request: dict[str, Any], _response: dict[str, Any]) -> bool:
            return lose_gather and request.get("op") == "gather"

        with ProtocolFaultProxy(host.address, drop_response=drop_gathers) as proxy:
            host.begin_execution("origin")
            client = RLMClient(proxy.address, host.auth_token)
            activate(client, "origin")
            handle = await client.spawn("child")
            with pytest.raises(RLMHostError, match="connection failed"):
                await client.gather([handle])
            origin = host.end_execution("origin", successful=True)

            lose_gather = False
            host.begin_execution("next")
            activate(client, "next")
            recovered = await client.gather([handle])
            next_summary = host.end_execution("next", successful=True)

    assert recovered[0].text == "done"
    assert origin.gathered == 1
    assert origin.usages == [{"tokens": 5}]
    assert next_summary.gathered == 0
    assert next_summary.usages == []
    assert host.live_handle_count == 0


@pytest.mark.asyncio
async def test_cleanup_loss_keeps_local_value_recoverable_until_release_is_acknowledged() -> None:
    with running_host(lambda request, cancel: result("done", {"tokens": 3})) as host:
        lose_release = True

        def drop_releases(request: dict[str, Any], _response: dict[str, Any]) -> bool:
            return lose_release and request.get("op") == "release"

        with ProtocolFaultProxy(host.address, drop_response=drop_releases) as proxy:
            host.begin_execution("cell")
            client = RLMClient(proxy.address, host.auth_token)
            activate(client, "cell")
            handle = await client.spawn("child")
            with pytest.raises(RLMHostError, match="connection failed"):
                await client.gather([handle])
            assert repr(handle) == "<rlm.Handle recoverable>"
            assert host.live_handle_count == 0

            lose_release = False
            value = (await client.gather([handle]))[0]
            summary = host.end_execution("cell", successful=True)

    assert value.text == "done"
    assert repr(handle) == "<rlm.Handle consumed>"
    assert summary.gathered == 1
    assert summary.usages == [{"tokens": 3}]


@pytest.mark.asyncio
async def test_empty_gather_short_circuits_without_host_recovery_state() -> None:
    with running_host(lambda request, cancel: result("unused"), max_live_handles=1) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        for _ in range(100):
            assert await client.gather([]) == []
        assert host.live_handle_count == 0
        handle = await client.spawn("capacity")
        await handle.release()
        host.end_execution("cell", successful=True)


@pytest.mark.asyncio
async def test_concurrent_local_gathers_have_one_winner_and_one_usage() -> None:
    gate = threading.Event()

    def runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        gate.wait(1)
        return result("done", {"tokens": 7})

    with running_host(runner) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        handle = await client.spawn("race")

        async def gather_once() -> str:
            try:
                await client.gather([handle])
                return "ok"
            except RLMHostError:
                return "error"

        first = asyncio.create_task(gather_once())
        second = asyncio.create_task(gather_once())
        await asyncio.sleep(0.02)
        gate.set()
        statuses = await asyncio.gather(first, second)
        summary = host.end_execution("cell", successful=True)

    assert sorted(statuses) == ["error", "ok"]
    assert summary.gathered == 1
    assert summary.usages == [{"tokens": 7}]


@pytest.mark.asyncio
async def test_release_is_idempotent_and_frees_capacity() -> None:
    gate = threading.Event()

    def runner(request: ChildRequest, cancel: threading.Event) -> ChildOutcome:
        gate.wait(1)
        return result(request.task)

    with running_host(runner, max_live_handles=1) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        handle = await client.spawn("first")
        await handle.release()
        await handle.release()
        for _ in range(100):
            if host.live_handle_count == 0:
                break
            await asyncio.sleep(0.01)
        replacement = await client.spawn("replacement")
        await replacement.release()
        gate.set()
        host.end_execution("cell", successful=True)


@pytest.mark.asyncio
async def test_stale_execution_context_cannot_operate_on_handle() -> None:
    with running_host(lambda request, cancel: result("done")) as host:
        host.begin_execution("origin")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "origin")
        stale = contextvars.copy_context()
        handle = await client.spawn("child")
        host.end_execution("origin", successful=True)

        host.begin_execution("next")
        activate(client, "next")
        task = asyncio.create_task(client.gather([handle]), context=stale)
        with pytest.raises(RLMHostError, match="no longer active"):
            await task
        await handle.release()
        host.end_execution("next", successful=True)


def test_public_execution_session_binds_host_and_client_ordering() -> None:
    host = AsyncRLMHost(child_execution(lambda request, cancel: result("unused")))
    host.start()
    try:
        client = RLMClient(host.address, host.auth_token)
        with host.execution(client, "cell") as execution:
            client.final_sync({"answer": 42})
        assert execution.summary is not None
        assert execution.summary.final.value == {"answer": 42}
        with pytest.raises(RLMHostError, match="not bound"):
            client.validate_execution()
    finally:
        host.stop()


def test_nested_execution_sessions_restore_the_outer_client_binding() -> None:
    host = AsyncRLMHost(child_execution(lambda request, cancel: result("unused")))
    host.start()
    try:
        client = RLMClient(host.address, host.auth_token)
        with host.execution(client, "outer") as outer:
            with host.execution(client, "inner") as inner:
                client.final_sync("inner")
            assert inner.summary is not None
            assert inner.summary.final.value == "inner"
            client.final_sync("outer")
        assert outer.summary is not None
        assert outer.summary.final.value == "outer"
    finally:
        host.stop()


def test_execution_session_exception_restores_the_previous_client_context() -> None:
    host = AsyncRLMHost(child_execution(lambda request, cancel: result("unused")))
    host.start()
    try:
        client = RLMClient(host.address, host.auth_token)
        with pytest.raises(ValueError, match="body failed"):
            with host.execution(client, "cell"):
                client.validate_execution()
                raise ValueError("body failed")
        with pytest.raises(RLMHostError, match="not bound"):
            client.validate_execution()
    finally:
        host.stop()


def test_concurrent_ipython_preparation_is_rejected_explicitly() -> None:
    client = RLMClient(("127.0.0.1", 1), "token")
    client.prepare_execution("first")
    failures: list[BaseException] = []

    def prepare_second() -> None:
        try:
            client.prepare_execution("second")
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=prepare_second)
    worker.start()
    worker.join(1)
    client.activate_execution()

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "overlapping IPython execution preparation" in str(failures[0])


def test_answer_ready_is_not_committed_when_final_push_fails() -> None:
    client = RLMClient(("127.0.0.1", 1), "token")
    answer = RLMAnswerDict(client)
    answer["content"] = "stale"

    def fail_final(_value) -> None:
        raise RLMHostError("transport failed")

    cast(Any, client).final_sync = fail_final
    with pytest.raises(RLMHostError, match="transport failed"):
        answer["ready"] = True
    assert answer == {"content": "stale", "ready": False}


def test_ipython_installation_does_not_patch_threading() -> None:
    class Events:
        def __init__(self) -> None:
            self.handlers: list[tuple[str, object]] = []

        def register(self, name: str, handler: object) -> None:
            self.handlers.append((name, handler))

        def unregister(self, name: str, handler: object) -> None:
            self.handlers.remove((name, handler))

    class Shell:
        def __init__(self) -> None:
            self.events = Events()

    original_start = threading.Thread.start
    client = RLMClient(("127.0.0.1", 1), "token")
    shell = Shell()

    client.install_ipython(shell)
    assert threading.Thread.start is original_start
    client.uninstall_ipython(shell)
    assert threading.Thread.start is original_start


def test_raw_background_thread_is_rejected_without_execution_binding() -> None:
    with running_host(lambda request, cancel: result("unused")) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        failures: list[BaseException] = []

        def validate() -> None:
            try:
                client.validate_execution()
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=validate)
        worker.start()
        worker.join(1)
        host.end_execution("cell", successful=True)

    assert len(failures) == 1
    assert isinstance(failures[0], RLMHostError)
    assert "raw background threads do not inherit" in str(failures[0])


def test_client_timeout_default_and_validation() -> None:
    assert RLMClient(("127.0.0.1", 1), "token").timeout == 310.0
    for invalid in (0, -0.1):
        with pytest.raises(ValueError, match="positive or None"):
            RLMClient(("127.0.0.1", 1), "token", timeout=invalid)
