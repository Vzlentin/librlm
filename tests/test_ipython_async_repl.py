"""Subprocess async-handle acceptance tests for IPythonREPL."""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("IPython")

from rlm.core.types import ModelUsageSummary, REPLResult, RLMChatCompletion, UsageSummary
from rlm.environments.ipython_kernel import kernel_bootstrap_code
from rlm.environments.ipython_queries import QueryOutcome
from rlm.environments.ipython_repl import IPythonREPL
from rlm.environments.ipython_sessions import SubprocessKernelSession
from tests.ipython_repl_test_support import HAS_SUBPROCESS as _has_subprocess
from tests.ipython_repl_test_support import _FakeSubcall


def test_kernel_bootstrap_binds_canonical_client_without_protocol_codegen():
    code = kernel_bootstrap_code(("127.0.0.1", 1234), "token", None)
    assert "install_kernel_runtime as _RLM_INSTALL_KERNEL" in code
    assert "_RLM_CLIENT = _RLM_INSTALL_KERNEL" in code
    assert "def " not in code
    assert "socket" not in code
    assert "struct" not in code
    assert "json.dumps" not in code
    assert "timeout=None" in code


def test_async_spawn_gather_and_json_final():
    fake = _FakeSubcall(["ALPHA", "BETA"])
    with IPythonREPL(kernel_mode="subprocess", subcall_fn=fake) as repl:
        result = repl.execute_code(
            "hs = [await rlm.spawn('a', context='ctx'), await rlm.spawn('b')]\n"
            "values = await rlm.gather(hs)\n"
            "await rlm.final({'statuses': [v['status'] for v in values], "
            "'texts': [v['text'] for v in values]})"
        )
    assert result.stderr == ""
    assert result.final_answer == {"statuses": ["ok", "ok"], "texts": ["ALPHA", "BETA"]}
    assert [call.response for call in result.rlm_calls] == ["ALPHA", "BETA"]
    assert fake.calls[0][0] == "a\n\n<context>\nctx\n</context>"


def test_async_handle_survives_successful_cell():
    fake = _FakeSubcall(["CROSSCELL"])
    with IPythonREPL(kernel_mode="subprocess", subcall_fn=fake) as repl:
        first = repl.execute_code("h = await rlm.spawn('cross-cell')")
        second = repl.execute_code(
            "value = (await rlm.gather([h]))[0]\n"
            "await rlm.final({'status': value['status'], 'text': value['text']})"
        )
    assert first.stderr == ""
    assert second.final_answer == {"status": "ok", "text": "CROSSCELL"}


def test_async_concurrent_gather_has_one_winner():
    fake = _FakeSubcall(["DONE"])
    code = """
import asyncio
h = await rlm.spawn("race")
async def gather_once():
    try:
        value = await rlm.gather([h])
        return {"status": "ok", "value": value}
    except Exception as error:
        return {"status": "error", "error": str(error)}
a, b = await asyncio.gather(gather_once(), gather_once())
await rlm.final({"attempts": [a, b]})
"""
    with IPythonREPL(kernel_mode="subprocess", subcall_fn=fake) as repl:
        result = repl.execute_code(code)
    statuses = sorted(item["status"] for item in result.final_answer["attempts"])
    assert statuses == ["error", "ok"]


def test_async_final_from_failed_cell_is_not_surfaced():
    with IPythonREPL(kernel_mode="subprocess") as repl:
        result = repl.execute_code("await rlm.final({'hidden': True})\nraise RuntimeError('boom')")
    assert "RuntimeError" in result.stderr
    assert result.final_answer is None
    assert result.has_final_answer is False


def test_failed_legacy_final_does_not_leave_stale_answer_readiness():
    with IPythonREPL(kernel_mode="subprocess") as repl:
        failed = repl.execute_code(
            "answer['content'] = 'stale'\nanswer['ready'] = True\nraise RuntimeError('boom')"
        )
        following = repl.execute_code("print(answer)")

    assert "RuntimeError" in failed.stderr
    assert failed.has_final_answer is False
    assert following.stderr == ""
    assert following.stdout.strip() == "{'content': '', 'ready': False}"
    assert following.has_final_answer is False


def test_async_null_final_preserves_presence():
    with IPythonREPL(kernel_mode="subprocess") as repl:
        result = repl.execute_code("await rlm.final(None)")
    assert result.final_answer is None
    assert result.has_final_answer is True


def test_answer_and_async_final_share_first_write_precedence():
    with IPythonREPL(kernel_mode="subprocess") as repl:
        answer_first = repl.execute_code(
            "answer['content'] = 'legacy'\nanswer['ready'] = True\nawait rlm.final({'async': True})"
        )
        final_first = repl.execute_code(
            "await rlm.final({'async': True})\n"
            "answer['content'] = 'late legacy'\n"
            "answer['ready'] = True"
        )

    assert answer_first.final_answer == "legacy"
    assert final_first.final_answer == {"async": True}


def test_async_scaffold_is_restored_between_cells():
    with IPythonREPL(kernel_mode="subprocess") as repl:
        repl.execute_code(
            "answer = {}; rlm = None; llm_query = None; llm_query_batched = None; "
            "rlm_query = None; rlm_query_batched = None; SHOW_VARS = None"
        )
        result = repl.execute_code(
            "helpers = [llm_query, llm_query_batched, rlm_query, rlm_query_batched, SHOW_VARS]\n"
            "print(all(callable(helper) for helper in helpers))\n"
            "print(type(answer).__name__)\n"
            "answer['content'] = 'restored'\n"
            "answer['ready'] = True"
        )
    assert result.stderr == ""
    assert result.stdout.splitlines() == ["True", "RLMAnswerDict"]
    assert result.final_answer == "restored"


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_activation_failure_prevents_user_cell_execution(tmp_path):
    marker = tmp_path / "user-cell-ran"
    with IPythonREPL(
        kernel_mode="subprocess",
        working_dir=str(tmp_path),
    ) as repl:
        kernel = repl._kernel
        assert isinstance(kernel, SubprocessKernelSession)
        patch_result = kernel._execute_control(
            "import rlm.environments.ipython_kernel as _kernel_module\n"
            "def _broken_install(*args, **kwargs):\n"
            "    raise RuntimeError('activation boom')\n"
            "_kernel_module.install_kernel_runtime = _broken_install",
            timeout=2,
        )
        assert patch_result.stderr == ""

        with pytest.raises(RuntimeError, match="activation boom"):
            repl.execute_code(f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')")

        assert marker.exists() is False
        assert kernel.host._executions == {}


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_activation_requires_positive_host_validation(tmp_path):
    marker = tmp_path / "user-cell-ran"
    with IPythonREPL(
        kernel_mode="subprocess",
        working_dir=str(tmp_path),
    ) as repl:
        kernel = repl._kernel
        assert isinstance(kernel, SubprocessKernelSession)
        patch_result = kernel._execute_control(
            "import rlm.environments.ipython_kernel as _kernel_module\n"
            "_kernel_module.install_kernel_runtime = lambda *args, **kwargs: None",
            timeout=2,
        )
        assert patch_result.stderr == ""

        with pytest.raises(RuntimeError, match="validate_execution"):
            repl.execute_code(f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')")

    assert marker.exists() is False


def test_async_child_completion_error_is_a_canonical_child_error():
    usage = UsageSummary(model_usage_summaries={})

    def failed_subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        return RLMChatCompletion(
            root_model="fake",
            prompt=prompt,
            response="Error: child failed",
            usage_summary=usage,
            execution_time=0.0,
            error="child failed",
        )

    with IPythonREPL(kernel_mode="subprocess", subcall_fn=failed_subcall) as repl:
        result = repl.execute_code(
            "h = await rlm.spawn('failure')\n"
            "value = (await rlm.gather([h]))[0]\n"
            "await rlm.final(value)"
        )

    assert result.final_answer["status"] == "error"
    assert result.final_answer["text"] is None
    assert result.final_answer["error"] == "child failed"


def test_async_child_runner_receives_cell_cancellation():
    cancelled = threading.Event()

    def runner(request, cancel):
        assert cancel.wait(2)
        cancelled.set()
        raise RuntimeError("cancelled")

    with IPythonREPL(
        kernel_mode="subprocess",
        async_child_runner=runner,
        cell_timeout=0.2,
    ) as repl:
        result = repl.execute_code("h = await rlm.spawn('slow')\nawait rlm.gather([h])")
    assert "TimeoutError" in result.stderr
    assert cancelled.wait(1)


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_subprocess_timeout_restarts_kernel_that_catches_interrupt(tmp_path):
    release_side_effect = tmp_path / "release-side-effect"
    late_side_effect = tmp_path / "late-side-effect.txt"
    interrupt_caught = threading.Event()

    def observe_output(event: dict) -> None:
        if event.get("text") == "interrupt-caught\n":
            interrupt_caught.set()

    with IPythonREPL(
        kernel_mode="subprocess",
        working_dir=str(tmp_path),
        cell_timeout=0.2,
        output_callback=observe_output,
    ) as repl:
        before = repl.execute_code("import os; print(os.getpid())")
        timed_out = repl.execute_code(
            "import time\n"
            "from pathlib import Path\n"
            "try:\n"
            "    time.sleep(10)\n"
            "except KeyboardInterrupt:\n"
            "    print('interrupt-caught', flush=True)\n"
            f"    while not Path({str(release_side_effect)!r}).exists():\n"
            "        time.sleep(0.01)\n"
            f"    Path({str(late_side_effect)!r}).write_text('late')\n"
        )
        after = repl.execute_code("import os; print(os.getpid()); print(6 * 7)")
        release_side_effect.touch()
        checked_side_effect = repl.execute_code(
            f"from pathlib import Path; print(Path({str(late_side_effect)!r}).exists())"
        )

    assert "TimeoutError" in timed_out.stderr
    assert interrupt_caught.is_set()
    assert before.stdout.strip() != after.stdout.splitlines()[0]
    assert after.stderr == ""
    assert after.stdout.splitlines()[1] == "42"
    assert checked_side_effect.stdout.strip() == "False"
    assert not late_side_effect.exists()


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_subprocess_working_dir_and_streaming_callback(tmp_path):
    events: list[dict] = []
    (tmp_path / "value.txt").write_text("from-cwd")
    code = """
from pathlib import Path
from IPython.display import clear_output
print(Path("value.txt").read_text())
clear_output(wait=False)
print("after-clear")
"""
    with IPythonREPL(
        kernel_mode="subprocess",
        working_dir=str(tmp_path),
        output_callback=events.append,
    ) as repl:
        result = repl.execute_code(code)
    assert result.stdout == "after-clear\n"
    assert any(
        event == {"type": "text", "kind": "stdout", "text": "after-clear\n"} for event in events
    )
    assert any(event["type"] == "clear" for event in events)
    assert (tmp_path / "value.txt").read_text() == "from-cwd"


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_working_dir_tree_is_unchanged_by_context_and_history_transport(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "sentinel.bin").write_bytes(b"\x00caller-owned\xff")
    (tmp_path / "context_0.json").write_bytes(b'{"sentinel":true}\n')
    (tmp_path / "context_1.txt").write_bytes(b"existing context\n")
    (tmp_path / "history_0.json").write_bytes(b"existing history\n")

    def snapshot() -> dict[str, tuple[str, bytes | None]]:
        return {
            str(path.relative_to(tmp_path)): (
                "dir" if path.is_dir() else "file",
                None if path.is_dir() else path.read_bytes(),
            )
            for path in sorted(tmp_path.rglob("*"))
        }

    before = snapshot()
    with IPythonREPL(
        kernel_mode="subprocess",
        working_dir=str(tmp_path),
        context_payload={"loaded": 0},
    ) as repl:
        kernel = repl._kernel
        assert isinstance(kernel, SubprocessKernelSession)
        transport_dir = kernel.transport_dir
        repl.add_context("loaded one")
        repl.add_history([{"role": "user", "content": "remember"}])
        cwd = repl.execute_code("import os; print(os.getcwd())")

        assert cwd.stdout.strip() == str(tmp_path)
        assert snapshot() == before
        assert {path.name for path in transport_dir.iterdir()} >= {
            "context_0.json",
            "context_1.txt",
            "history_0.json",
        }

    assert snapshot() == before
    assert not transport_dir.exists()
    assert tmp_path.exists()


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_control_assignments_do_not_open_user_executions(monkeypatch: pytest.MonkeyPatch):
    opened: list[str] = []
    with IPythonREPL(kernel_mode="subprocess") as repl:
        kernel = repl._kernel
        assert isinstance(kernel, SubprocessKernelSession)
        original = kernel.host.begin_execution

        def record_begin(execution_id: str) -> None:
            opened.append(execution_id)
            original(execution_id)

        monkeypatch.setattr(kernel.host, "begin_execution", record_begin)
        repl.add_context({"control": True})
        repl.add_history([{"role": "user", "content": "control"}])
        assert opened == []

        result = repl.execute_code("print(context['control'], history[0]['role'])")

    assert result.stderr == ""
    assert result.stdout.strip() == "True user"
    assert len(opened) == 1


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_output_callback_failure_drains_and_preserves_kernel_state(tmp_path):
    class CallbackFailure(RuntimeError):
        pass

    events: list[dict] = []

    def fail_once(event: dict) -> None:
        events.append(event)
        raise CallbackFailure("output consumer failed")

    usage = UsageSummary(
        model_usage_summaries={
            "child": ModelUsageSummary(
                total_calls=1,
                total_input_tokens=2,
                total_output_tokens=3,
                total_cost=0.01,
            )
        }
    )

    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        return RLMChatCompletion(
            root_model="child",
            prompt=prompt,
            response="child-result",
            usage_summary=usage,
            execution_time=0.0,
        )

    with IPythonREPL(
        kernel_mode="subprocess",
        working_dir=str(tmp_path),
        output_callback=fail_once,
        subcall_fn=subcall,
    ) as repl:
        with pytest.warns(RuntimeWarning, match="output callback failed"):
            first = repl.execute_code(
                "import os; seeded = 41; print(os.getpid())\n"
                "h = await rlm.spawn('child')\n"
                "value = (await rlm.gather([h]))[0]\n"
                "await rlm.final({'text': value['text'], 'seeded': seeded})"
            )

        first_pid = "".join(
            event.get("text", "") for event in events if event.get("type") == "text"
        ).strip()
        second = repl.execute_code("print(os.getpid()); print(os.getcwd()); print(seeded + 1)")

    assert first.final_answer == {"text": "child-result", "seeded": 41}
    assert len(first.rlm_calls) == 1
    assert first.rlm_calls[0].usage_summary.total_cost == 0.01
    assert second.stderr == ""
    assert second.stdout.splitlines() == [first_pid, str(tmp_path), "42"]
    assert len(events) == 1


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_output_callback_streams_before_cell_completion():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    result: list[REPLResult] = []

    def callback(event: dict) -> None:
        if event.get("text") == "streamed\n":
            callback_entered.set()
            assert release_callback.wait(1)

    with IPythonREPL(kernel_mode="subprocess", output_callback=callback) as repl:
        worker = threading.Thread(
            target=lambda: result.append(repl.execute_code("print('streamed'); value = 42"))
        )
        worker.start()
        assert callback_entered.wait(1)
        assert worker.is_alive()
        release_callback.set()
        worker.join(2)
        assert not worker.is_alive()

    assert result[0].stdout == "streamed\n"


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_recursive_query_batch_starts_slots_concurrently_with_cell_timeout():
    active = 0
    lock = threading.Lock()
    parallel = threading.Event()
    observed: list[bool] = []

    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        nonlocal active
        with lock:
            active += 1
            if active == 2:
                parallel.set()
        observed.append(parallel.wait(1))
        with lock:
            active -= 1
        return RLMChatCompletion(
            root_model="fake",
            prompt=prompt,
            response=prompt.upper(),
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    with IPythonREPL(
        kernel_mode="subprocess",
        subcall_fn=subcall,
        cell_timeout=3.0,
        max_concurrent_subcalls=2,
    ) as repl:
        result = repl.execute_code("print(rlm_query_batched(['a', 'b', 'c']))")

    assert parallel.is_set()
    assert all(observed)
    assert "['A', 'B', 'C']" in result.stdout


# -----------------------------------------------------------------------------
# Scheduling, attribution, and subprocess lifecycle
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Stale subcall attribution (subprocess mode)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_stale_subcall_completion_not_misattributed_to_next_cell(tmp_path):
    """A late cell-A subcall must not be attributed to an active cell B."""
    subcall_started = threading.Event()
    release_subcall = threading.Event()
    subcall_finished = threading.Event()
    cell_b_started = threading.Event()
    release_cell_b = tmp_path / "release-cell-b"

    def observe_output(event: dict) -> None:
        if event.get("text") == "cell-b-started\n":
            cell_b_started.set()

    def blocked_subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        subcall_started.set()
        assert release_subcall.wait(5)
        subcall_finished.set()
        usage = UsageSummary(
            model_usage_summaries={
                "fake-model": ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=0,
                    total_output_tokens=0,
                )
            }
        )
        return RLMChatCompletion(
            root_model="fake-model",
            prompt=prompt,
            response="late",
            usage_summary=usage,
            execution_time=0.0,
        )

    with IPythonREPL(
        kernel_mode="subprocess",
        subcall_fn=blocked_subcall,
        cell_timeout=0.1,
        output_callback=observe_output,
    ) as repl:
        first = repl.execute_code('rlm_query("p")')
        assert "TimeoutError" in first.stderr
        assert subcall_started.is_set()
        assert not subcall_finished.is_set()

        repl.cell_timeout = None
        results: list[REPLResult] = []
        errors: list[BaseException] = []

        def run_cell_b() -> None:
            try:
                results.append(
                    repl.execute_code(
                        "import time\n"
                        "from pathlib import Path\n"
                        "print('cell-b-started', flush=True)\n"
                        f"while not Path({str(release_cell_b)!r}).exists():\n"
                        "    time.sleep(0.01)"
                    )
                )
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=run_cell_b)
        worker.start()
        try:
            assert cell_b_started.wait(5)
            assert worker.is_alive()
            release_subcall.set()
            assert subcall_finished.wait(5)
            assert worker.is_alive()
        finally:
            release_subcall.set()
            release_cell_b.touch()
            worker.join(5)

        assert not worker.is_alive()
        assert errors == []
        assert len(results) == 1
        second = results[0]

    assert second.rlm_calls == []


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_stale_final_answer_not_misattributed_to_next_cell():
    """A final answer set by a prior cell's delayed code path must not
    surface as the next cell's final answer."""

    barrier = threading.Event()

    def slow_subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        # Hold the kernel-side rlm_query call long enough that we get
        # to issue cell B before subcall_fn ever returns.
        barrier.wait(timeout=5.0)
        usage = UsageSummary(
            model_usage_summaries={
                "fake-model": ModelUsageSummary(
                    total_calls=1, total_input_tokens=0, total_output_tokens=0
                )
            }
        )
        return RLMChatCompletion(
            root_model="fake-model",
            prompt=prompt,
            response="late",
            usage_summary=usage,
            execution_time=0.0,
        )

    with IPythonREPL(
        kernel_mode="subprocess",
        subcall_fn=slow_subcall,
        cell_timeout=0.1,
    ) as repl:
        r1 = repl.execute_code(
            'rlm_query("p")\nanswer["content"] = "late-A"\nanswer["ready"] = True'
        )
        assert "TimeoutError" in r1.stderr
        # Cell A is over with no final answer captured.
        assert r1.final_answer is None

        # Now release subcall_fn and let it complete. If cell A's
        # answer were globally stored (by old code), it would now be
        # in the broker waiting for the next drain to pick it up.
        barrier.set()
        time.sleep(0.3)

        repl.cell_timeout = 5.0
        r2 = repl.execute_code("print('clean')")

    assert r2.final_answer is None, (
        f"cell B must not inherit cell A's stale final answer; got {r2.final_answer!r}"
    )


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_queued_spawn_returns_handle_before_subcall_admission():
    occupied_entered = threading.Event()
    spawn_entered = threading.Event()
    release_occupied = threading.Event()

    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        if prompt == "occupied":
            occupied_entered.set()
            assert release_occupied.wait(2)
        elif prompt == "spawn":
            spawn_entered.set()
        return RLMChatCompletion(
            root_model="fake",
            prompt=prompt,
            response=prompt.upper(),
            usage_summary=UsageSummary.empty(),
            execution_time=0.0,
        )

    with IPythonREPL(
        kernel_mode="subprocess",
        subcall_fn=subcall,
        max_concurrent_subcalls=1,
    ) as repl:
        first = repl.execute_code("occupied = await rlm.spawn('occupied')")
        assert first.stderr == ""
        assert occupied_entered.wait(2)

        second_results: list[REPLResult] = []
        worker = threading.Thread(
            target=lambda: second_results.append(
                repl.execute_code("queued = await rlm.spawn('spawn')\nprint('handle-created')")
            )
        )
        worker.start()
        try:
            worker.join(1)
            assert not worker.is_alive(), "rlm.spawn waited for subcall admission"
            assert second_results[0].stdout.strip() == "handle-created"
            assert not spawn_entered.is_set()
        finally:
            release_occupied.set()
        gathered = repl.execute_code(
            "print([item['text'] for item in await rlm.gather([occupied, queued])])"
        )

    assert gathered.stderr == ""
    assert gathered.stdout.strip() == "['OCCUPIED', 'SPAWN']"


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_host_fallback_bypasses_occupied_recursive_admission():
    child_entered = threading.Event()
    release_child = threading.Event()
    lm_entered = threading.Event()

    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        if prompt == "occupied":
            child_entered.set()
            assert release_child.wait(2)
        return RLMChatCompletion(
            root_model="fake",
            prompt=prompt,
            response=prompt.upper(),
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    def query_runner(
        kind: str,
        prompts: list[str],
        model: str | None,
        cancel: threading.Event,
    ) -> list[QueryOutcome]:
        assert kind == "rlm"
        assert not cancel.is_set()
        lm_entered.set()
        return [
            QueryOutcome.success(
                RLMChatCompletion(
                    root_model="fake-lm",
                    prompt=prompt,
                    response="LM-FALLBACK",
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                )
            )
            for prompt in prompts
        ]

    with IPythonREPL(
        kernel_mode="subprocess",
        subcall_fn=subcall,
        max_concurrent_subcalls=1,
    ) as repl:
        kernel = repl._kernel
        assert isinstance(kernel, SubprocessKernelSession)
        kernel.host.recursive_queries = False
        kernel.host.query_runner = query_runner
        first = repl.execute_code("h = await rlm.spawn('occupied')")
        assert first.stderr == ""
        assert child_entered.wait(2)

        second_results: list[REPLResult] = []
        worker = threading.Thread(
            target=lambda: second_results.append(repl.execute_code("print(rlm_query('fallback'))"))
        )
        worker.start()
        try:
            assert lm_entered.wait(2), "LM fallback waited for recursive admission"
            worker.join(2)
            assert not worker.is_alive()
            assert second_results[0].stdout.strip() == "LM-FALLBACK"
        finally:
            release_child.set()
        repl.execute_code("await rlm.gather([h])")


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_spawn_and_recursive_query_share_subcall_admission_across_cells():
    spawn_entered = threading.Event()
    query_entered = threading.Event()
    release_spawn = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0

    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if prompt == "spawn":
                spawn_entered.set()
                assert release_spawn.wait(2)
            elif prompt == "query":
                query_entered.set()
            return RLMChatCompletion(
                root_model="fake",
                prompt=prompt,
                response=prompt.upper(),
                usage_summary=UsageSummary.empty(),
                execution_time=0.0,
            )
        finally:
            with lock:
                active -= 1

    with IPythonREPL(
        kernel_mode="subprocess",
        subcall_fn=subcall,
        max_concurrent_subcalls=1,
    ) as repl:
        first = repl.execute_code("h = await rlm.spawn('spawn')")
        assert first.stderr == ""
        assert spawn_entered.wait(2)

        query_results: list[REPLResult] = []
        worker = threading.Thread(
            target=lambda: query_results.append(repl.execute_code("print(rlm_query('query'))"))
        )
        worker.start()
        try:
            time.sleep(0.1)
            assert worker.is_alive()
            assert not query_entered.is_set()
            assert peak == 1
        finally:
            release_spawn.set()
        worker.join(5)
        assert not worker.is_alive()
        assert query_results[0].stdout.strip() == "QUERY"
        gathered = repl.execute_code("print((await rlm.gather([h]))[0]['text'])")

    assert gathered.stderr == ""
    assert gathered.stdout.strip() == "SPAWN"
    assert peak == 1


def test_subcall_concurrency_is_globally_bounded():
    """``max_concurrent_subcalls`` caps simultaneous ``subcall_fn`` calls
    even across multiple ``rlm_query`` / ``rlm_query_batched`` requests."""
    if not _has_subprocess:
        pytest.skip("jupyter_client not installed")

    in_flight = 0
    peak = 0
    cv_lock = threading.Lock()

    def slow_subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        nonlocal in_flight, peak
        with cv_lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with cv_lock:
            in_flight -= 1
        usage = UsageSummary(
            model_usage_summaries={
                "fake-model": ModelUsageSummary(
                    total_calls=1, total_input_tokens=0, total_output_tokens=0
                )
            }
        )
        return RLMChatCompletion(
            root_model="fake-model",
            prompt=prompt,
            response="ok",
            usage_summary=usage,
            execution_time=0.05,
        )

    # cap=2 → no matter how many calls the kernel fans out, only 2 should
    # ever be running at once.
    with IPythonREPL(
        kernel_mode="subprocess",
        subcall_fn=slow_subcall,
        max_concurrent_subcalls=2,
    ) as repl:
        # Two batched-of-8 requests issued from kernel-side threads
        # simulates fan-out far above the cap.
        result = repl.execute_code(
            "import asyncio\n"
            "async def go():\n"
            "    await asyncio.to_thread(\n"
            "        rlm_query_batched, ['p1','p2','p3','p4','p5','p6','p7','p8']\n"
            "    )\n"
            "await asyncio.gather(go(), go())\n"
            "print('done')"
        )
    assert result.stderr == ""
    assert "done" in result.stdout
    assert len(result.rlm_calls) == 16
    assert peak == 2


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_setup_failure_runs_cleanup():
    """If construction fails after the broker/kernel start, cleanup runs.

    A non-JSON-serializable context_payload triggers a TypeError in
    ``add_context`` *after* ``setup()`` has already brought up the kernel
    and broker. The new try/except in ``__init__`` must run cleanup
    before re-raising; otherwise the kernel + broker would be orphaned.
    """

    class _Unserializable:
        pass

    with pytest.raises(TypeError):
        IPythonREPL(
            kernel_mode="subprocess",
            context_payload={"bad": _Unserializable()},
        )
