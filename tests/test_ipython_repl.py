"""Shared IPythonREPL behavior across both kernel modes."""

from __future__ import annotations

import pytest

pytest.importorskip("IPython")

from rlm.core.types import RLMChatCompletion, UsageSummary
from rlm.environments.ipython_repl import IPythonREPL
from tests.ipython_repl_test_support import (
    BOTH_MODES,
    _FakeSubcall,
)
from tests.ipython_repl_test_support import (
    HAS_SUBPROCESS as _has_subprocess,
)


@BOTH_MODES
def test_simple_execution(kernel_mode: str):
    with IPythonREPL(kernel_mode=kernel_mode) as repl:
        result = repl.execute_code("x = 1 + 2\nprint(x)")
    assert result.stderr == ""
    assert "3" in result.stdout


@pytest.mark.skipif(not _has_subprocess, reason="jupyter_client not installed")
def test_subprocess_locals_is_only_host_owned_restart_state():
    with IPythonREPL(kernel_mode="subprocess", context_payload={"key": "value"}) as repl:
        repl.execute_code("user_value = 42")
        visible = repl.locals

    assert visible == {
        "context_0": {"key": "value"},
        "context": {"key": "value"},
    }
    assert "user_value" not in visible


@BOTH_MODES
def test_error_captured_in_stderr(kernel_mode: str):
    with IPythonREPL(kernel_mode=kernel_mode) as repl:
        result = repl.execute_code("1 / 0")
    assert "ZeroDivisionError" in result.stderr


@BOTH_MODES
def test_variable_persistence_across_cells(kernel_mode: str):
    with IPythonREPL(kernel_mode=kernel_mode) as repl:
        repl.execute_code("a = 40")
        repl.execute_code("b = a + 2")
        result = repl.execute_code("print(b)")
    assert "42" in result.stdout


@BOTH_MODES
def test_answer_ready_surfaces_final_answer(kernel_mode: str):
    with IPythonREPL(kernel_mode=kernel_mode) as repl:
        result = repl.execute_code('answer["content"] = "forty-two"\nanswer["ready"] = True')
    assert result.final_answer == "forty-two"


@BOTH_MODES
def test_answer_not_ready_returns_none(kernel_mode: str):
    with IPythonREPL(kernel_mode=kernel_mode) as repl:
        result = repl.execute_code('answer["content"] = "wip"')
    assert result.final_answer is None


@BOTH_MODES
def test_answer_content_is_stringified(kernel_mode: str):
    with IPythonREPL(kernel_mode=kernel_mode) as repl:
        result = repl.execute_code('answer["content"] = 123\nanswer["ready"] = True')
    assert result.final_answer == "123"


@BOTH_MODES
def test_load_context_string(kernel_mode: str):
    with IPythonREPL(kernel_mode=kernel_mode, context_payload="hello world") as repl:
        result = repl.execute_code("print(len(context))")
    assert "11" in result.stdout


@BOTH_MODES
def test_load_context_dict(kernel_mode: str):
    payload = {"name": "alice", "count": 7}
    with IPythonREPL(kernel_mode=kernel_mode, context_payload=payload) as repl:
        result = repl.execute_code('print(context["name"], context["count"])')
    assert "alice" in result.stdout
    assert "7" in result.stdout


@BOTH_MODES
def test_custom_tool_callable(kernel_mode: str):
    def greet(name: str) -> str:
        return f"hello, {name}"

    with IPythonREPL(
        kernel_mode=kernel_mode,
        custom_tools={"greet": greet},
    ) as repl:
        result = repl.execute_code('print(greet("world"))')
    assert "hello, world" in result.stdout


@BOTH_MODES
def test_custom_tool_data(kernel_mode: str):
    with IPythonREPL(
        kernel_mode=kernel_mode,
        custom_tools={"MAGIC": 42},
    ) as repl:
        result = repl.execute_code("print(MAGIC * 2)")
    assert "84" in result.stdout


@BOTH_MODES
def test_custom_tool_binding_is_restored_before_the_next_cell(kernel_mode: str):
    with IPythonREPL(kernel_mode=kernel_mode, custom_tools={"tool_value": 1}) as repl:
        repl.execute_code("tool_value = 2")
        result = repl.execute_code("print(tool_value)")
    assert result.stderr == ""
    assert result.stdout.strip() == "1"


@BOTH_MODES
def test_input_is_disabled_again_after_a_cell_rebinds_it(kernel_mode: str):
    with IPythonREPL(kernel_mode=kernel_mode) as repl:
        repl.execute_code("input = lambda *_args, **_kwargs: 'rebound'")
        result = repl.execute_code("input('prompt')")
    assert "RuntimeError" in result.stderr
    assert "input() is disabled" in result.stderr


@BOTH_MODES
def test_custom_tool_odd_name_is_a_namespace_key_not_python_source(kernel_mode: str):
    odd_name = "injected = True\n#"
    with IPythonREPL(
        kernel_mode=kernel_mode,
        custom_tools={odd_name: {"value": 42}},
    ) as repl:
        result = repl.execute_code(
            f"print(globals()[{odd_name!r}]['value'])\nprint(globals().get('injected'))"
        )

    assert result.stderr == ""
    assert result.stdout.splitlines() == ["42", "None"]


@BOTH_MODES
def test_rlm_query_dispatches_to_subcall_fn(kernel_mode: str):
    subcall = _FakeSubcall(responses=["child-answer"])
    with IPythonREPL(kernel_mode=kernel_mode, subcall_fn=subcall) as repl:
        result = repl.execute_code('r = rlm_query("think harder")\nprint(r)')
    assert "child-answer" in result.stdout
    assert len(subcall.calls) == 1
    assert subcall.calls[0][0] == "think harder"
    # Completion recorded on the REPLResult
    assert len(result.rlm_calls) == 1
    assert result.rlm_calls[0].response == "child-answer"


@BOTH_MODES
def test_rlm_query_batched_dispatches(kernel_mode: str):
    calls: list[str] = []
    responses = {"p1": "a", "p2": "b", "p3": "c"}

    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        calls.append(prompt)
        return RLMChatCompletion(
            root_model="fake",
            prompt=prompt,
            response=responses[prompt],
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    with IPythonREPL(kernel_mode=kernel_mode, subcall_fn=subcall) as repl:
        result = repl.execute_code(
            'rs = rlm_query_batched(["p1","p2","p3"])\nprint(\',\'.join(rs))'
        )
    assert "a,b,c" in result.stdout
    assert sorted(calls) == ["p1", "p2", "p3"]
    assert len(result.rlm_calls) == 3


@BOTH_MODES
def test_rlm_query_preserves_returned_errors_and_isolates_raised_errors(kernel_mode: str):
    def subcall(prompt: str, model: str | None = None) -> RLMChatCompletion:
        if prompt == "raises":
            raise RuntimeError
        error = "returned failure" if prompt == "returned" else None
        return RLMChatCompletion(
            root_model="fake",
            prompt=prompt,
            response="Error: returned failure" if error else "GOOD",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
            error=error,
        )

    repl = IPythonREPL(kernel_mode=kernel_mode, subcall_fn=subcall)
    result = repl.execute_code(
        'single = rlm_query("returned")\n'
        'batch = rlm_query_batched(["good", "returned", "raises"])\n'
        "print(single)\n"
        'print("|".join(batch))'
    )
    with pytest.raises(RuntimeError):
        repl.cleanup()

    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        "Error: returned failure",
        "GOOD|Error: returned failure|Error: RLM query failed - RuntimeError",
    ]
    assert [
        (completion.prompt, completion.response, completion.error)
        for completion in result.rlm_calls
    ] == [
        ("returned", "Error: returned failure", "returned failure"),
        ("good", "GOOD", None),
        ("returned", "Error: returned failure", "returned failure"),
    ]


@BOTH_MODES
def test_rlm_query_falls_back_when_no_subcall_fn(kernel_mode: str):
    """Without subcall_fn, rlm_query falls through to llm_query (which errors
    cleanly when no LM handler is configured)."""
    with IPythonREPL(kernel_mode=kernel_mode, subcall_fn=None) as repl:
        result = repl.execute_code('print(rlm_query("hi"))')
    # Error either about missing subcall_fn, handler, or kernel address —
    # the contract is: it doesn't crash and surfaces a clear error string.
    assert "Error" in result.stdout


def test_missing_lm_handler_has_identical_query_output_in_both_modes():
    modes = ["in_process"]
    if _has_subprocess:
        modes.append("subprocess")

    outputs = []
    for mode in modes:
        with IPythonREPL(kernel_mode=mode, lm_handler_address=None) as repl:
            outputs.append(repl.execute_code("print(llm_query('x'))").stdout.strip())

    assert outputs == ["Error: LM query failed - No LM handler configured"] * len(modes)


@BOTH_MODES
def test_rlm_calls_do_not_leak_across_cells(kernel_mode: str):
    """Each REPLResult.rlm_calls must reflect only the current cell's calls."""
    subcall = _FakeSubcall(responses=["x"])
    with IPythonREPL(kernel_mode=kernel_mode, subcall_fn=subcall) as repl:
        r1 = repl.execute_code('rlm_query("a")')
        r2 = repl.execute_code('rlm_query("b")')
        r3 = repl.execute_code('rlm_query("c")')
        r4 = repl.execute_code('print("no rlm here")')
    assert len(r1.rlm_calls) == 1
    assert len(r2.rlm_calls) == 1
    assert len(r3.rlm_calls) == 1
    assert len(r4.rlm_calls) == 0
