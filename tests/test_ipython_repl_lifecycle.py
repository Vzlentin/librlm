"""IPythonREPL configuration, lifecycle, and concurrency tests."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("IPython")

from rlm.core.child_execution import ChildOutcome
from rlm.core.types import ModelUsageSummary, REPLResult, RLMChatCompletion, UsageSummary
from rlm.environments.ipython_in_process import (
    InProcessKernelSession,
)
from rlm.environments.ipython_repl import IPythonREPL
from rlm.environments.ipython_sessions import SubprocessKernelSession
from tests.ipython_repl_test_support import (
    BOTH_MODES,
    _FakeSubcall,
)
from tests.ipython_repl_test_support import (
    HAS_SUBPROCESS as _has_subprocess,
)


def test_cleanup_makes_setup_terminal_without_allocating_a_replacement():
    repl = IPythonREPL(kernel_mode="in_process")
    repl.cleanup()

    try:
        with pytest.raises(RuntimeError, match="closed"):
            repl.setup()
        assert repl._kernel is None
    finally:
        replacement = repl._kernel
        if replacement is not None:
            replacement.close()
            repl._kernel = None


def test_in_process_rejects_subprocess_only_child_runner():
    repl = None
    try:
        with pytest.raises(ValueError, match="async_child_runner.*subprocess"):
            repl = IPythonREPL(
                kernel_mode="in_process",
                async_child_runner=lambda _request, _cancel: None,
            )
    finally:
        if repl is not None:
            repl.cleanup()


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_subprocess_timeout_interrupts_cleanly():
    with IPythonREPL(kernel_mode="subprocess", cell_timeout=0.5) as repl:
        start = time.perf_counter()
        result = repl.execute_code("import time; time.sleep(5)")
        elapsed = time.perf_counter() - start

        assert "TimeoutError" in result.stderr
        # Must have actually interrupted, not waited the full 5s
        assert elapsed < 3.0

        # Kernel should still be alive and responsive afterward
        followup = repl.execute_code("print('still-alive')")
    assert "still-alive" in followup.stdout


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_subprocess_no_timeout_by_default():
    """Without cell_timeout, short code runs without issue."""
    with IPythonREPL(kernel_mode="subprocess") as repl:
        result = repl.execute_code("print(sum(range(100)))")
    assert "4950" in result.stdout


def test_invalid_kernel_mode_rejected():
    with pytest.raises(ValueError):
        IPythonREPL(kernel_mode="nonsense")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_invalid_startup_timeout_rejected(bad):
    with pytest.raises(ValueError, match="startup_timeout"):
        IPythonREPL(kernel_mode="in_process", startup_timeout=bad)


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_invalid_subcall_timeout_rejected(bad):
    with pytest.raises(ValueError, match="subcall_timeout"):
        IPythonREPL(kernel_mode="in_process", subcall_timeout=bad)


def test_subcall_timeout_none_is_allowed():
    """``None`` is the documented "no timeout" sentinel, distinct from 0."""
    with IPythonREPL(kernel_mode="in_process", subcall_timeout=None) as repl:
        assert repl.subcall_timeout is None


def test_init_failure_preserves_original_exception_over_cleanup_error(monkeypatch):
    """If ``setup`` fails *and* cleanup would also fail, ``__init__``
    must surface the original cause, not the cleanup error."""

    class _Boom(Exception):
        pass

    # Fail before any IPython state is created so the test leaves no handlers behind.
    def _broken_setup(self):
        raise _Boom("original cause")

    monkeypatch.setattr(IPythonREPL, "setup", _broken_setup)

    # Make cleanup itself raise too, so we can verify the original wins.
    def _broken_cleanup(self):
        raise RuntimeError("cleanup also failed")

    monkeypatch.setattr(IPythonREPL, "cleanup", _broken_cleanup)

    with pytest.raises(_Boom, match="original cause"):
        IPythonREPL(kernel_mode="in_process")


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_subprocess_early_construction_failure_does_not_leak_transport_dir():
    before = set(Path(tempfile.gettempdir()).glob("ipython_transport_*"))
    with pytest.raises(ValueError, match="concurrency"):
        IPythonREPL(kernel_mode="subprocess", max_concurrent_subcalls=0)
    after = set(Path(tempfile.gettempdir()).glob("ipython_transport_*"))
    assert after == before


def test_in_process_partial_construction_drops_user_module(monkeypatch):
    before = {name for name in sys.modules if name.startswith("_rlm_ipython_main_")}

    def fail_scaffold(self, queries):
        raise RuntimeError("scaffold failed")

    monkeypatch.setattr(InProcessKernelSession, "_restore_scaffold", fail_scaffold)
    with pytest.raises(RuntimeError, match="scaffold failed"):
        IPythonREPL(kernel_mode="in_process")

    after = {name for name in sys.modules if name.startswith("_rlm_ipython_main_")}
    assert after == before


@BOTH_MODES
def test_cleanup_is_idempotent(kernel_mode: str):
    repl = IPythonREPL(kernel_mode=kernel_mode)
    repl.execute_code("x = 1")
    repl.cleanup()
    repl.cleanup()  # must not raise


def test_in_process_cleanup_waits_for_an_active_cell_and_removes_owned_workdir():
    kernel_mode = "in_process"
    original_cwd = os.getcwd()
    repl = IPythonREPL(kernel_mode=kernel_mode)
    working_dir = repl.working_dir
    results: list[REPLResult] = []

    execution = threading.Thread(
        target=lambda: results.append(
            repl.execute_code(
                "from pathlib import Path\n"
                "import time\n"
                "Path('started').write_text('yes')\n"
                "while not Path('release').exists():\n"
                "    time.sleep(0.01)\n"
            )
        )
    )
    execution.start()
    deadline = time.monotonic() + 10
    while not os.path.exists(os.path.join(working_dir, "started")):
        assert time.monotonic() < deadline
        time.sleep(0.01)

    cleanup = threading.Thread(target=repl.cleanup)
    cleanup.start()
    time.sleep(0.1)
    assert cleanup.is_alive()
    assert os.path.isdir(working_dir)

    with open(os.path.join(working_dir, "release"), "w") as marker:
        marker.write("yes")
    execution.join(10)
    cleanup.join(10)

    assert not execution.is_alive()
    assert not cleanup.is_alive()
    assert len(results) == 1
    assert repl._kernel is None
    assert not os.path.exists(working_dir)
    assert os.getcwd() == original_cwd


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_subprocess_cleanup_stops_an_active_cell_without_a_cell_timeout():
    repl = IPythonREPL(kernel_mode="subprocess")
    working_dir = repl.working_dir
    kernel = repl._kernel
    assert isinstance(kernel, SubprocessKernelSession)
    manager = kernel.manager
    execution_errors: list[BaseException] = []
    cleanup_errors: list[BaseException] = []

    def execute_forever() -> None:
        try:
            repl.execute_code(
                "from pathlib import Path\n"
                "import time\n"
                "Path('started').write_text('yes')\n"
                "while True:\n"
                "    time.sleep(1)\n"
            )
        except BaseException as error:
            execution_errors.append(error)

    execution = threading.Thread(target=execute_forever)
    execution.start()
    deadline = time.monotonic() + 10
    while not os.path.exists(os.path.join(working_dir, "started")):
        assert time.monotonic() < deadline
        time.sleep(0.01)

    def cleanup() -> None:
        try:
            repl.cleanup()
        except BaseException as error:
            cleanup_errors.append(error)

    cleanup_thread = threading.Thread(target=cleanup)
    cleanup_thread.start()
    cleanup_thread.join(10)
    cleanup_blocked = cleanup_thread.is_alive()
    if cleanup_blocked:
        manager.shutdown_kernel(now=True)
        cleanup_thread.join(5)
    execution.join(5)

    assert not cleanup_blocked, "cleanup remained blocked behind the active cell"
    assert not cleanup_thread.is_alive()
    assert not execution.is_alive()
    assert cleanup_errors == []
    assert not manager.is_alive()
    assert not os.path.exists(working_dir)


def test_subprocess_close_reports_failure_after_attempting_every_phase(tmp_path):
    calls: list[str] = []

    class Host:
        def reset(self) -> None:
            calls.append("reset")

        def stop(self) -> None:
            calls.append("host.stop")

    class Manager:
        def interrupt_kernel(self) -> None:
            calls.append("interrupt")

        def shutdown_kernel(self, *, now: bool) -> None:
            assert now is True
            calls.append("shutdown")
            raise RuntimeError("kernel shutdown failed")

    class Client:
        def stop_channels(self) -> None:
            calls.append("channels")

    transport_dir = tmp_path / "transport"
    transport_dir.mkdir()
    session = object.__new__(SubprocessKernelSession)
    session._lifecycle = threading.Condition()
    session._lifecycle_state = "open"
    session._active_operations = 0
    session.host = Host()
    session.manager = Manager()
    session.client = Client()
    session.transport_dir = transport_dir

    with pytest.raises(RuntimeError, match="kernel shutdown failed"):
        session.close()

    assert calls == ["reset", "interrupt", "shutdown", "channels", "host.stop"]
    assert not transport_dir.exists()
    assert session._lifecycle_state == "closed"
    session.close()


def test_in_process_cleanup_reports_reset_failure_after_releasing_other_resources(
    monkeypatch,
):
    repl = IPythonREPL(kernel_mode="in_process")
    kernel = repl._kernel
    assert isinstance(kernel, InProcessKernelSession)
    working_dir = repl.working_dir

    def fail_reset(*, new_session: bool) -> None:
        assert new_session is False
        raise RuntimeError("shell reset failed")

    monkeypatch.setattr(kernel.shell, "reset", fail_reset)
    with pytest.raises(RuntimeError, match="shell reset failed"):
        repl.cleanup()

    assert repl._kernel is None
    assert not os.path.exists(working_dir)


def test_failed_finalization_drains_persistent_accounting_once(monkeypatch):
    charged = UsageSummary(
        model_usage_summaries={
            "child": ModelUsageSummary(
                total_calls=1,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost=0.25,
            )
        }
    )
    failure = RuntimeError("kernel finalization failed")
    accounting_failure = ValueError("pending accounting failed")

    with IPythonREPL(kernel_mode="in_process", persistent=True) as repl:
        kernel = repl._kernel
        assert kernel is not None
        original_finalize = kernel.finalize_completion
        calls = 0

        def fail_once() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise failure
            original_finalize()

        class FailingChildExecution:
            def finalize(self) -> None:
                raise accounting_failure

        monkeypatch.setattr(kernel, "finalize_completion", fail_once)
        completion = RLMChatCompletion(
            root_model="child",
            prompt="child",
            response="done",
            usage_summary=charged,
            execution_time=0.0,
        )
        repl._settle_custom_child(ChildOutcome.from_completion(completion, charged, elapsed_ms=0))
        repl._child_executions.append(FailingChildExecution())

        first = repl.finalize_completion()
        repl._child_executions.pop()
        second = repl.finalize_completion()

    assert isinstance(first.error, BaseExceptionGroup)
    assert first.error.exceptions == (failure, accounting_failure)
    assert first.usage.total_cost == pytest.approx(0.25)
    assert second.error is None
    assert second.usage.total_cost in (None, 0.0)


def test_get_environment_routes_ipython():
    from rlm.environments import get_environment

    env = get_environment("ipython", {"kernel_mode": "in_process"})
    try:
        assert isinstance(env, IPythonREPL)
        result = env.execute_code("print(2 ** 10)")
        assert "1024" in result.stdout
    finally:
        env.cleanup()


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_get_environment_routes_ipython_subprocess():
    from rlm.environments import get_environment

    env = get_environment("ipython", {"kernel_mode": "subprocess"})
    try:
        assert isinstance(env, IPythonREPL)
        result = env.execute_code("print(2 ** 10)")
        assert "1024" in result.stdout
    finally:
        env.cleanup()


def test_ipython_repl_importable_from_package():
    """``IPythonREPL`` is in ``__all__`` and must be importable lazily."""
    from rlm.environments import IPythonREPL as Imported

    assert Imported is IPythonREPL


@BOTH_MODES
def test_cell_timeout_zero_is_rejected(kernel_mode: str):
    with pytest.raises(ValueError, match="cell_timeout must be positive"):
        IPythonREPL(kernel_mode=kernel_mode, cell_timeout=0)


@BOTH_MODES
def test_subcall_fn_reentry_into_same_repl_raises(kernel_mode: str):
    """If subcall_fn calls execute_code on its parent REPL, the call must
    raise — not deadlock (cross-thread broker case) or silently clobber
    the parent's tracking state (same-thread in-process case)."""

    seen: list[str] = []
    repl_holder: dict[str, IPythonREPL] = {}

    def reentering_subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        try:
            repl_holder["repl"].execute_code("print('inner')")
        except RuntimeError as e:
            seen.append(str(e))
            raise
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
            response="never",
            usage_summary=usage,
            execution_time=0.0,
        )

    repl = IPythonREPL(
        kernel_mode=kernel_mode,
        subcall_fn=reentering_subcall,
    )
    repl_holder["repl"] = repl
    # Run rlm_query → subcall_fn → reentrant execute_code → raises.
    # We bound the call so a regression that deadlocks fails the
    # test instead of hanging the suite forever.
    result_holder: list[REPLResult] = []
    err_holder: list[BaseException] = []

    def runner() -> None:
        try:
            result_holder.append(repl.execute_code('print(rlm_query("p"))'))
        except BaseException as e:
            err_holder.append(e)

    t = threading.Thread(target=runner)
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive(), "execute_code deadlocked instead of raising"
    with pytest.raises(RuntimeError, match="Reentrant execute_code"):
        repl.cleanup()

    # subcall_fn observed the reentry RuntimeError.
    assert seen, "reentering subcall_fn was never reached or never raised"
    assert any("Reentrant" in s for s in seen), seen


@BOTH_MODES
def test_subcall_cleanup_reentry_fails_without_deadlocking(kernel_mode: str):
    repl_holder: dict[str, IPythonREPL] = {}

    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        repl_holder["repl"].cleanup()
        raise AssertionError("cleanup should have raised")

    repl = IPythonREPL(kernel_mode=kernel_mode, subcall_fn=subcall)
    repl_holder["repl"] = repl
    results: list[REPLResult] = []
    worker = threading.Thread(
        target=lambda: results.append(repl.execute_code("print(rlm_query('child'))"))
    )
    worker.start()
    worker.join(10)
    assert not worker.is_alive()
    with pytest.raises(RuntimeError, match="cleanup cannot run"):
        repl.cleanup()

    assert len(results) == 1
    assert "cleanup cannot run from an active IPython cell or child callback" in results[0].stdout


@BOTH_MODES
def test_concurrent_execute_does_not_lose_rlm_calls(kernel_mode: str):
    """Two threads each running an ``rlm_query`` cell must each report
    exactly one call. Prior to the lock-scope fix the in-process path
    raced on ``_pending_llm_calls`` and lost entries."""
    subcall = _FakeSubcall(responses=["x"])
    counts: list[int] = []
    lock = threading.Lock()

    def worker(repl: IPythonREPL) -> None:
        for _ in range(5):
            r = repl.execute_code("rlm_query('p')")
            with lock:
                counts.append(len(r.rlm_calls))

    with IPythonREPL(kernel_mode=kernel_mode, subcall_fn=subcall) as repl:
        threads = [threading.Thread(target=worker, args=(repl,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # 4 threads × 5 cells = 20 cells, each expected to report exactly 1 call.
    assert len(counts) == 20
    assert all(c == 1 for c in counts), f"some cells lost rlm_calls — saw {sorted(counts)}"
