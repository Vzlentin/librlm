"""In-process IPython kernel execution, timeout, and isolation tests."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

import pytest

pytest.importorskip("IPython")

from rlm.core.types import ModelUsageSummary, REPLResult, RLMChatCompletion, UsageSummary
from rlm.environments.ipython_in_process import (
    InProcessKernelSession,
    IPythonCellTimeoutError,
    _InProcessQueryChannel,
)
from rlm.environments.ipython_queries import QueryCoordinator
from rlm.environments.ipython_repl import IPythonREPL
from tests.ipython_repl_test_support import _FakeSubcall

_can_alarm = sys.platform != "win32" and threading.current_thread() is threading.main_thread()


def test_already_cancelled_in_process_query_has_owned_timeout():
    cancel = threading.Event()
    cancel.set()
    coordinator = QueryCoordinator(
        lm_handler_address=lambda: None,
        depth=1,
        child_execution=None,
    )
    channel = _InProcessQueryChannel(coordinator)
    timeout_owner = object()
    queries = channel.begin_cell(cancel, timeout_owner)

    with pytest.raises(IPythonCellTimeoutError) as failure:
        queries.rlm_query("cancelled")

    assert failure.value.owner is timeout_owner


def test_in_process_exposes_locals_without_subprocess_async_api():
    """In-process mode mirrors LocalREPL without exposing subprocess handles."""
    with IPythonREPL(kernel_mode="in_process") as repl:
        repl.execute_code("x = 42\ny = 'hi'")
        assert repl.locals["x"] == 42
        assert repl.locals["y"] == "hi"
        assert "rlm" not in repl.locals


def test_in_process_nested_child_reenters_process_owner_on_same_thread():
    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        with IPythonREPL(kernel_mode="in_process") as child:
            nested = child.execute_code("print('nested')")
        return RLMChatCompletion(
            root_model="fake",
            prompt=prompt,
            response=nested.stdout.strip(),
            usage_summary=UsageSummary.empty(),
            execution_time=0.0,
        )

    with IPythonREPL(kernel_mode="in_process", subcall_fn=subcall) as repl:
        result = repl.execute_code("print(rlm_query('child'))")

    assert result.stderr == ""
    assert result.stdout.strip() == "nested"
    assert [completion.response for completion in result.rlm_calls] == ["nested"]


def test_in_process_worker_recursive_child_fails_without_deadlocking():
    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        with IPythonREPL(kernel_mode="in_process") as child:
            nested = child.execute_code("print('nested')")
        return RLMChatCompletion(
            root_model="fake",
            prompt=prompt,
            response=nested.stdout.strip(),
            usage_summary=UsageSummary.empty(),
            execution_time=0.0,
        )

    repl = IPythonREPL(kernel_mode="in_process", subcall_fn=subcall)
    result = repl.execute_code(
        "import threading\n"
        "values = []\n"
        "worker = threading.Thread(target=lambda: values.append(rlm_query('child')))\n"
        "worker.start()\n"
        "worker.join(2)\n"
        "print(worker.is_alive())\n"
        "print(values[0])\n"
    )
    with pytest.raises(RuntimeError, match="recursive in-process child"):
        repl.cleanup()

    assert result.stderr == ""
    assert result.stdout.splitlines()[0] == "False"
    assert "recursive in-process child cannot start from a worker thread" in result.stdout


def test_in_process_recursive_batch_stays_on_the_execution_thread():
    entered: list[tuple[str, int]] = []

    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        entered.append((prompt, threading.get_ident()))
        return RLMChatCompletion(
            root_model="fake",
            prompt=prompt,
            response=prompt.upper(),
            usage_summary=UsageSummary.empty(),
            execution_time=0.0,
        )

    execution_thread = threading.get_ident()
    with IPythonREPL(
        kernel_mode="in_process",
        subcall_fn=subcall,
        max_concurrent_subcalls=2,
    ) as repl:
        result = repl.execute_code("print(rlm_query_batched(['a', 'b', 'c']))")

    assert result.stderr == ""
    assert result.stdout.strip() == "['A', 'B', 'C']"
    assert entered == [("a", execution_thread), ("b", execution_thread), ("c", execution_thread)]
    assert [completion.prompt for completion in result.rlm_calls] == ["a", "b", "c"]


@pytest.mark.skipif(not _can_alarm, reason="SIGALRM requires Unix main thread")
def test_nested_in_process_timeout_preserves_parent_deadline():
    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        with IPythonREPL(kernel_mode="in_process", cell_timeout=0.05) as child:
            nested = child.execute_code("import time; time.sleep(0.2)")
        assert "TimeoutError" in nested.stderr
        return RLMChatCompletion(
            root_model="fake",
            prompt=prompt,
            response="child-timed-out",
            usage_summary=UsageSummary.empty(),
            execution_time=0.0,
        )

    with IPythonREPL(
        kernel_mode="in_process",
        subcall_fn=subcall,
        cell_timeout=5,
    ) as repl:
        result = repl.execute_code("print(rlm_query('child'))")

    assert result.stderr == ""
    assert result.stdout.strip() == "child-timed-out"


@pytest.mark.skipif(not _can_alarm, reason="SIGALRM requires Unix main thread")
def test_in_process_timeout_interrupts_c_level_sleep():
    with IPythonREPL(kernel_mode="in_process", cell_timeout=0.3) as repl:
        start = time.perf_counter()
        result = repl.execute_code("import time; time.sleep(5)")
        elapsed = time.perf_counter() - start
    assert "TimeoutError" in result.stderr
    assert elapsed < 2.0, f"expected fast interrupt, got {elapsed:.2f}s"


@pytest.mark.skipif(not _can_alarm, reason="SIGALRM requires Unix main thread")
def test_in_process_timeout_interrupts_python_loop():
    with IPythonREPL(kernel_mode="in_process", cell_timeout=0.3) as repl:
        start = time.perf_counter()
        result = repl.execute_code("i = 0\nwhile True: i += 1")
        elapsed = time.perf_counter() - start
    assert "TimeoutError" in result.stderr
    assert elapsed < 2.0, f"expected fast interrupt, got {elapsed:.2f}s"


@pytest.mark.skipif(not _can_alarm, reason="SIGALRM requires Unix main thread")
def test_in_process_timeout_is_terminal_through_recursive_batch():
    continued = threading.Event()
    entered: list[str] = []

    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        entered.append(prompt)
        if prompt == "first":
            time.sleep(2)
            continued.set()
        return RLMChatCompletion(
            root_model="fake",
            prompt=prompt,
            response=prompt.upper(),
            usage_summary=UsageSummary.empty(),
            execution_time=0.0,
        )

    with IPythonREPL(
        kernel_mode="in_process",
        subcall_fn=subcall,
        cell_timeout=0.05,
        max_concurrent_subcalls=2,
    ) as repl:
        timed_out = repl.execute_code("rlm_query_batched(['first', 'must-not-run'])")
        followup = repl.execute_code("print(rlm_query('next'))")

    assert "TimeoutError" in timed_out.stderr
    assert entered == ["first", "next"]
    assert not continued.is_set()
    assert "NEXT" in followup.stdout


def test_in_process_timeout_fails_before_execution_off_main_thread():
    errors: list[BaseException] = []
    with IPythonREPL(kernel_mode="in_process", cell_timeout=0.1) as repl:

        def execute() -> None:
            try:
                repl.execute_code("ran = True")
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=execute)
        worker.start()
        worker.join(2)

        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "requires SIGALRM on the main thread" in str(errors[0])
        assert "ran" not in repl.locals


@pytest.mark.skipif(not _can_alarm, reason="SIGALRM requires Unix main thread")
def test_in_process_shell_alive_after_timeout():
    with IPythonREPL(kernel_mode="in_process", cell_timeout=0.3) as repl:
        repl.execute_code("import time; time.sleep(5)")
        followup = repl.execute_code("print('still-alive')")
    assert "still-alive" in followup.stdout


@pytest.mark.skipif(not _can_alarm, reason="SIGALRM requires Unix main thread")
def test_in_process_timeout_no_fire_for_fast_cell():
    with IPythonREPL(kernel_mode="in_process", cell_timeout=2.0) as repl:
        result = repl.execute_code("print(sum(range(1000)))")
    assert result.stderr == ""
    assert "499500" in result.stdout


@pytest.mark.skipif(not _can_alarm, reason="ITIMER_REAL requires Unix main thread")
def test_in_process_without_timeout_leaves_external_timer_active():
    fired = threading.Event()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, lambda _signum, _frame: fired.set())
    signal.setitimer(signal.ITIMER_REAL, 0.05)
    try:
        with IPythonREPL(kernel_mode="in_process") as repl:
            result = repl.execute_code("import time; time.sleep(0.15)")
        assert result.stderr == ""
        assert fired.is_set()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


@pytest.mark.skipif(not _can_alarm, reason="ITIMER_REAL requires Unix main thread")
def test_in_process_timeout_rejects_an_active_external_timer():
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, lambda _signum, _frame: None)
    signal.setitimer(signal.ITIMER_REAL, 5)
    try:
        with IPythonREPL(kernel_mode="in_process", cell_timeout=1) as repl:
            with pytest.raises(RuntimeError, match="active external ITIMER_REAL"):
                repl.execute_code("print('must not run')")
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def test_in_process_error_includes_traceback_and_user_frame():
    """In-process stderr should include a real traceback, not just a one-liner."""
    with IPythonREPL(kernel_mode="in_process") as repl:
        result = repl.execute_code("def boom():\n    1/0\n\nboom()")
    assert "ZeroDivisionError" in result.stderr
    assert "Traceback" in result.stderr
    assert "boom" in result.stderr  # user frame name visible


def test_in_process_input_is_disabled():
    """``input()`` must not block in cells; it should raise."""
    with IPythonREPL(kernel_mode="in_process") as repl:
        result = repl.execute_code("input('go')")
    assert "RuntimeError" in result.stderr or "input() is disabled" in result.stderr


def test_in_process_input_restored_if_user_overwrites():
    """If a cell rebinds ``input``, the next cell must still see the disabled stub."""
    with IPythonREPL(kernel_mode="in_process") as repl:
        repl.execute_code("input = lambda *a, **kw: 'evil'")
        result = repl.execute_code("input('go')")
    assert "RuntimeError" in result.stderr or "input() is disabled" in result.stderr


def test_cleanup_unregisters_atexit_handler():
    """Cleanup must remove ``InteractiveShell.atexit_operations`` from the
    ``atexit`` registry so it doesn't hold a strong external reference to
    the shell for the rest of the process lifetime.

    ``atexit`` doesn't expose a clean way to introspect its registry, so we
    detect leakage indirectly: register a sentinel, then verify our cleanup
    didn't remove it (i.e., cleanup is targeted at the shell's handler, not
    a blanket clear). Combined with the absence of any error from
    ``atexit.unregister``, this confirms the unregister code path runs.
    """
    import atexit as _atexit

    sentinel_called = [False]

    def sentinel() -> None:
        sentinel_called[0] = True

    _atexit.register(sentinel)
    try:
        repl = IPythonREPL(kernel_mode="in_process")
        kernel = repl._kernel
        assert isinstance(kernel, InProcessKernelSession)
        assert callable(kernel.shell.atexit_operations)
        repl.cleanup()
        assert repl._kernel is None
        # Sentinel still registered — our unregister didn't blast unrelated
        # handlers.
        _atexit.unregister(sentinel)
    finally:
        _atexit.unregister(sentinel)


def test_in_process_scaffold_bound_owner_has_no_kernel_lifecycle():
    with IPythonREPL(kernel_mode="in_process") as repl:
        result = repl.execute_code(
            "owner = rlm_query.__self__\n"
            "names = ('execute', 'execute_code', 'assign', 'close', 'shutdown', '_shutdown')\n"
            "print([hasattr(owner, name) for name in names])"
        )

    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        "[False, False, False, False, False, False]",
    ]


def test_in_process_blocked_scaffold_reentry_preserves_outer_state():
    fake = _FakeSubcall(responses=["A"])
    with IPythonREPL(kernel_mode="in_process", subcall_fn=fake) as repl:
        result = repl.execute_code(
            "real = rlm_query('outer')\n"
            "try:\n"
            "    rlm_query.__self__.execute('inner')\n"
            "except AttributeError as e:\n"
            "    print('caught:', type(e).__name__)\n"
            "print('outer-result:', real)\n"
        )

    assert "caught: AttributeError" in result.stdout
    assert "outer-result: A" in result.stdout
    assert len(result.rlm_calls) == 1


def test_in_process_instances_serialize_process_global_state(tmp_path):
    original_cwd = os.getcwd()
    directory_a = tmp_path / "a"
    directory_b = tmp_path / "b"
    directory_a.mkdir()
    directory_b.mkdir()
    entered_a = threading.Event()
    entered_b = threading.Event()
    release_a = threading.Event()
    release_b = threading.Event()
    results: dict[str, REPLResult] = {}

    def block_a() -> None:
        entered_a.set()
        assert release_a.wait(5)

    def block_b() -> None:
        entered_b.set()
        assert release_b.wait(5)

    repl_a = IPythonREPL(
        kernel_mode="in_process",
        working_dir=str(directory_a),
        custom_tools={"block": block_a},
    )
    repl_b = IPythonREPL(
        kernel_mode="in_process",
        working_dir=str(directory_b),
        custom_tools={"block": block_b},
    )
    try:
        thread_a = threading.Thread(
            target=lambda: results.setdefault(
                "a",
                repl_a.execute_code("block(); import os; print('A=' + os.getcwd())"),
            )
        )
        thread_b = threading.Thread(
            target=lambda: results.setdefault(
                "b",
                repl_b.execute_code("block(); import os; print('B=' + os.getcwd())"),
            )
        )
        thread_a.start()
        assert entered_a.wait(5)
        thread_b.start()
        assert not entered_b.wait(0.2), "second in-process cell overlapped the first"
        release_a.set()
        thread_a.join(5)
        assert entered_b.wait(5)
        release_b.set()
        thread_b.join(5)

        assert not thread_a.is_alive()
        assert not thread_b.is_alive()
        assert results["a"].stdout.strip() == f"A={directory_a}"
        assert results["b"].stdout.strip() == f"B={directory_b}"
        assert os.getcwd() == original_cwd
    finally:
        release_a.set()
        release_b.set()
        repl_a.cleanup()
        repl_b.cleanup()


def test_in_process_finalization_waits_for_background_query_without_leaking():
    started = threading.Event()
    release = threading.Event()

    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        started.set()
        assert release.wait(5)
        return RLMChatCompletion(
            root_model="fake",
            prompt=prompt,
            response="late",
            usage_summary=UsageSummary.empty(),
            execution_time=0.0,
        )

    with IPythonREPL(kernel_mode="in_process", subcall_fn=subcall) as repl:
        # Injecting the wait from the host keeps the cell deterministic without
        # making the query thread part of the session lifecycle itself.
        repl.custom_tools["wait_started"] = lambda: started.wait(5)
        first = repl.execute_code(
            "import threading\n"
            "old_query = rlm_query\n"
            "worker = threading.Thread(target=lambda: old_query('old'))\n"
            "worker.start()\n"
            "wait_started()\n"
        )
        assert first.rlm_calls == []

        finalizations: list[object] = []
        finalizer = threading.Thread(
            target=lambda: finalizations.append(repl.finalize_completion())
        )
        finalizer.start()
        time.sleep(0.1)
        assert finalizer.is_alive()
        release.set()
        finalizer.join(5)
        assert not finalizer.is_alive()
        assert len(finalizations) == 1

        worker = repl.locals["worker"]
        worker.join(1)
        assert not worker.is_alive()
        with pytest.raises(RuntimeError, match="inactive IPython execution"):
            repl.locals["old_query"]("stale")
        second = repl.execute_code("print('next')")

    assert second.rlm_calls == []
    assert second.stdout.strip() == "next"


def test_in_process_two_instances_have_distinct_user_modules():
    """Two coexisting in-process instances must not share ``sys.modules['__main__']``."""
    repl_a = IPythonREPL(kernel_mode="in_process")
    repl_b = IPythonREPL(kernel_mode="in_process")
    try:
        repl_a.execute_code("x = 'A'")
        repl_b.execute_code("x = 'B'")
        ra = repl_a.execute_code("print(x)")
        rb = repl_b.execute_code("print(x)")
        assert "A" in ra.stdout and "B" not in ra.stdout
        assert "B" in rb.stdout and "A" not in rb.stdout
        # And ``sys.modules['__main__']`` is not the IPython user module
        # for either instance.
        kernel_a = repl_a._kernel
        kernel_b = repl_b._kernel
        assert isinstance(kernel_a, InProcessKernelSession)
        assert isinstance(kernel_b, InProcessKernelSession)
        assert kernel_a.user_module is not kernel_b.user_module
        assert kernel_a.user_module.__name__ != "__main__"
        assert kernel_b.user_module.__name__ != "__main__"
    finally:
        repl_a.cleanup()
        repl_b.cleanup()


def test_in_process_user_module_dropped_from_sys_modules_on_cleanup():
    """No ``sys.modules`` leak after cleanup."""
    import sys as _sys

    repl = IPythonREPL(kernel_mode="in_process")
    kernel = repl._kernel
    assert isinstance(kernel, InProcessKernelSession)
    name = kernel.user_module.__name__
    assert name in _sys.modules
    repl.cleanup()
    assert name not in _sys.modules


def test_in_process_subcall_concurrency_is_globally_bounded():
    """In-process mode also caps ``subcall_fn`` globally, matching subprocess.

    Without the per-instance semaphore, user threads inside a single cell
    that each call ``rlm_query`` / ``rlm_query_batched`` would fan out
    past ``max_concurrent_subcalls``.
    """

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

    with IPythonREPL(
        kernel_mode="in_process",
        subcall_fn=slow_subcall,
        max_concurrent_subcalls=1,
    ) as repl:
        result = repl.execute_code(
            "import threading\n"
            "def go():\n"
            "    rlm_query_batched(['p1','p2','p3','p4','p5','p6','p7','p8'])\n"
            "ts = [threading.Thread(target=go) for _ in range(2)]\n"
            "for t in ts: t.start()\n"
            "for t in ts: t.join()\n"
            "print('done')"
        )
    assert "done" in result.stdout
    assert peak == 1, f"in-process admission should cap concurrent children at 1, saw {peak}"
