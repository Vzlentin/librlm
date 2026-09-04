"""IPythonREPL persistence and host-owned state tests."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("IPython")

from rlm.environments.ipython_repl import IPythonREPL
from tests.ipython_repl_test_support import (
    BOTH_MODES,
)


@BOTH_MODES
def test_persistent_env_exposes_protocol(kernel_mode: str):
    from rlm.environments import SupportsPersistence

    with IPythonREPL(kernel_mode=kernel_mode, persistent=True) as repl:
        assert isinstance(repl, SupportsPersistence)


@BOTH_MODES
def test_add_context_versioning(kernel_mode: str):
    with IPythonREPL(kernel_mode=kernel_mode, persistent=True) as repl:
        i0 = repl.add_context("First")
        i1 = repl.add_context("Second")
        assert (i0, i1) == (0, 1)
        assert repl.get_context_count() == 2

        # Visible to executed code
        r = repl.execute_code("print(context_0, '|', context_1, '|', context)")
    assert "First" in r.stdout
    assert "Second" in r.stdout


@BOTH_MODES
def test_add_history_versioning_and_alias(kernel_mode: str):
    with IPythonREPL(kernel_mode=kernel_mode, persistent=True) as repl:
        h1 = [{"role": "user", "content": "Q1"}]
        h2 = [{"role": "user", "content": "Q2"}]
        repl.add_history(h1)
        repl.add_history(h2)
        assert repl.get_history_count() == 2

        r = repl.execute_code(
            "print(history_0[0]['content'], history_1[0]['content'], history[0]['content'])"
        )
    assert "Q1" in r.stdout
    assert "Q2" in r.stdout


@BOTH_MODES
def test_history_is_deep_copied(kernel_mode: str):
    with IPythonREPL(kernel_mode=kernel_mode, persistent=True) as repl:
        h = [{"role": "user", "content": "original"}]
        repl.add_history(h)
        h[0]["content"] = "mutated"
        r = repl.execute_code("print(history_0[0]['content'])")
    assert "original" in r.stdout
    assert "mutated" not in r.stdout


@BOTH_MODES
def test_update_handler_address(kernel_mode: str):
    with IPythonREPL(
        kernel_mode=kernel_mode,
        lm_handler_address=("127.0.0.1", 5000),
        persistent=True,
    ) as repl:
        repl.update_handler_address(("127.0.0.1", 6000))
    assert repl.lm_handler_address == ("127.0.0.1", 6000)


@BOTH_MODES
def test_context_alias_restored_after_user_overwrites_it(kernel_mode: str):
    """If user code reassigns ``context``, the next cell sees it restored."""
    with IPythonREPL(kernel_mode=kernel_mode, persistent=True) as repl:
        repl.add_context("original-text")
        repl.execute_code("context = 'overwritten'")
        r = repl.execute_code("print(context)")
    assert "original-text" in r.stdout


def test_in_process_persistent_multi_turn_variables():
    """Variables defined in one cell are visible in later add_context calls."""
    with IPythonREPL(kernel_mode="in_process", persistent=True) as repl:
        repl.add_context("Sales were $1000")
        repl.execute_code("sales = 1000")
        repl.add_context("Costs were $400")
        r = repl.execute_code("profit = sales - 400\nprint(profit)")
    assert "600" in r.stdout


@BOTH_MODES
def test_concurrent_add_context_indices_are_unique(kernel_mode: str):
    """Two threads calling ``add_context`` concurrently must not collide."""
    with IPythonREPL(kernel_mode=kernel_mode, persistent=True) as repl:
        results: list[int] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            idx = repl.add_context(f"payload-{i}")
            with lock:
                results.append(idx)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # All indices distinct, contiguous, and 0..7 in some order.
    assert sorted(results) == list(range(8))
    assert repl.get_context_count() == 8
