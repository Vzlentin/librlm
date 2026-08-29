from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from typing import Any

import pytest

from rlm.environments.ipython_async import (
    AsyncRLMHost,
    ChildRequest,
    IPythonOutputStream,
    RLMClient,
)


@contextmanager
def running_host(runner, **kwargs):
    host = AsyncRLMHost(runner, **kwargs)
    host.start()
    try:
        yield host
    finally:
        host.stop()


def activate(client: RLMClient, execution_id: str) -> None:
    client._set_execution(execution_id)
    client._activate_execution()


def result(text: str, usage: Any = None) -> dict[str, Any]:
    return {
        "status": "ok",
        "text": text,
        "error": None,
        "usage": usage if usage is not None else {},
        "elapsed_ms": 1,
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_spawn_returns_live_handle_and_gather_preserves_order() -> None:
    release = threading.Event()
    started: list[str] = []

    def runner(request: ChildRequest, cancel: threading.Event) -> dict[str, Any]:
        started.append(request.task)
        while not release.wait(0.01):
            assert not cancel.is_set()
        return result(request.task.upper(), {"tokens": 2})

    with running_host(runner) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        first, second = await asyncio.gather(client.spawn("a"), client.spawn("b"))
        await asyncio.sleep(0.02)
        assert sorted(started) == ["a", "b"]
        release.set()
        values = await client.gather([second, first])
        assert [value["text"] for value in values] == ["B", "A"]
        summary = host.end_execution("cell", successful=True)

    assert summary["spawned"] == 2
    assert summary["gathered"] == 2
    assert summary["usages"] == [{"tokens": 2}, {"tokens": 2}]


@pytest.mark.asyncio
async def test_handle_survives_successful_origin_cell() -> None:
    release = threading.Event()

    def runner(request: ChildRequest, cancel: threading.Event) -> dict[str, Any]:
        release.wait(1)
        return result(request.task)

    with running_host(runner) as host:
        host.begin_execution("origin")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "origin")
        handle = await client.spawn("cross-cell")
        origin = host.end_execution("origin", successful=True)
        assert origin["spawned"] == 1

        host.begin_execution("next")
        activate(client, "next")
        release.set()
        assert (await client.gather([handle]))[0]["text"] == "cross-cell"
        next_summary = host.end_execution("next", successful=True)

    assert next_summary["gathered"] == 1


@pytest.mark.asyncio
async def test_concurrent_gathers_commit_once() -> None:
    gate = threading.Event()

    def runner(request: ChildRequest, cancel: threading.Event) -> dict[str, Any]:
        gate.wait(1)
        return result("done", {"tokens": 7})

    with running_host(runner) as host:
        host.begin_execution("cell")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "cell")
        handle = await client.spawn("race")

        async def gather_once() -> tuple[str, Any]:
            try:
                return "ok", await client.gather([handle])
            except Exception as error:
                return "error", str(error)

        first = asyncio.create_task(gather_once())
        second = asyncio.create_task(gather_once())
        await asyncio.sleep(0.05)
        gate.set()
        attempts = await asyncio.gather(first, second)
        summary = host.end_execution("cell", successful=True)

    assert sorted(status for status, _ in attempts) == ["error", "ok"]
    assert summary["gathered"] == 1
    assert summary["usages"] == [{"tokens": 7}]


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

    assert success["has_final"] is True
    assert success["final_value"] == value
    assert failed["has_final"] is False
    assert failed["final_value"] is None


@pytest.mark.asyncio
async def test_failed_execution_cancels_owned_child_and_releases_handle() -> None:
    cancelled = threading.Event()

    def runner(request: ChildRequest, cancel: threading.Event) -> dict[str, Any]:
        assert cancel.wait(1)
        cancelled.set()
        raise RuntimeError("cancelled")

    with running_host(runner) as host:
        host.begin_execution("failed")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "failed")
        handle = await client.spawn("slow")
        host.end_execution("failed", successful=False)
        assert cancelled.wait(1)

        host.begin_execution("next")
        activate(client, "next")
        with pytest.raises(Exception, match="unknown or already gathered"):
            await client.gather([handle])
        host.end_execution("next", successful=True)


@pytest.mark.asyncio
async def test_background_gather_loses_claim_when_cell_ends() -> None:
    gate = threading.Event()

    def runner(request: ChildRequest, cancel: threading.Event) -> dict[str, Any]:
        gate.wait(1)
        return result("survived")

    with running_host(runner) as host:
        host.begin_execution("origin")
        client = RLMClient(host.address, host.auth_token)
        activate(client, "origin")
        handle = await client.spawn("slow")
        background = asyncio.create_task(client.gather([handle]))
        await asyncio.sleep(0.05)
        host.end_execution("origin", successful=True)

        host.begin_execution("next")
        activate(client, "next")
        gate.set()
        assert (await client.gather([handle]))[0]["text"] == "survived"
        with pytest.raises(Exception, match="no longer active"):
            await background
        host.end_execution("next", successful=True)


def test_release_callback_waits_for_origin_handles() -> None:
    releases: list[str] = []

    with running_host(lambda request, cancel: result("done"), on_release=releases.append) as host:
        host.begin_execution("origin")
        response = host._dispatch(
            {
                "version": 1,
                "auth": host.auth_token,
                "id": "spawn",
                "execution_id": "origin",
                "op": "spawn",
                "task": "x",
                "context": None,
                "cwd": str(__import__("pathlib").Path.cwd()),
            }
        )
        handle = response["result"]["handle"]
        host.end_execution("origin", successful=True)
        assert releases == []

        host.begin_execution("gather")
        gathered = host._dispatch(
            {
                "version": 1,
                "auth": host.auth_token,
                "id": "gather",
                "execution_id": "gather",
                "op": "gather",
                "handles": [handle],
            }
        )
        assert gathered["ok"] is True
        host.end_execution("gather", successful=True)

    assert releases.count("origin") == 1


def test_stale_context_identity_does_not_adopt_later_cell() -> None:
    client = RLMClient(("127.0.0.1", 1), "token")
    activate(client, "first")
    first_context = __import__("contextvars").copy_context()
    activate(client, "second")
    assert client._execution_id.get() == "second"
    assert first_context.run(client._execution_id.get) == "first"


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
