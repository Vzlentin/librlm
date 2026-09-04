"""Kernel-side installation and Jupyter output normalization."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from rlm.environments.ipython_client import RLMClient

_CUSTOM_TOOLS_KEY = "_RLM_CUSTOM_TOOLS"


class ScaffoldChannel(Protocol):
    """Only the synchronous query capabilities exposed by scaffold helpers."""

    def llm_query(self, prompt: str, model: str | None = None) -> str: ...

    def llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]: ...

    def rlm_query(self, prompt: str, model: str | None = None) -> str: ...

    def rlm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]: ...


def show_vars(namespace: dict[str, Any]) -> str:
    """Describe user-visible names in one kernel namespace."""
    skip = {"In", "Out", "exit", "quit", "get_ipython", "answer", "rlm"}
    available = {
        name: type(value).__name__
        for name, value in namespace.items()
        if not name.startswith("_") and name not in skip
    }
    if not available:
        return "No variables created yet. Use ```repl``` blocks to create variables."
    return f"Available variables: {available}"


def disabled_input(*_args: Any, **_kwargs: Any) -> str:
    """Reject stdin consistently in every IPython kernel mode."""
    raise RuntimeError("input() is disabled in IPythonREPL: cells cannot prompt for stdin")


def install_scaffold(
    shell: Any,
    channel: ScaffoldChannel,
    *,
    answer: Any,
    async_api: Any | None = None,
    custom_tools: Mapping[str, Any] | None = None,
) -> None:
    """Restore every host-owned capability before a cell runs.

    User variables persist, but rebinding a scaffold name or custom tool lasts
    only for the current cell. Tool objects themselves are not deep-copied.
    """
    namespace = shell.user_ns
    namespace["llm_query"] = channel.llm_query
    namespace["llm_query_batched"] = channel.llm_query_batched
    namespace["rlm_query"] = channel.rlm_query
    namespace["rlm_query_batched"] = channel.rlm_query_batched
    namespace["SHOW_VARS"] = lambda: show_vars(namespace)
    namespace["answer"] = answer
    if async_api is None:
        namespace.pop("rlm", None)
    else:
        namespace["rlm"] = async_api
    if custom_tools is not None:
        namespace.update(custom_tools)
    namespace["input"] = disabled_input
    if "context_0" in namespace:
        namespace["context"] = namespace["context_0"]
    if "history_0" in namespace:
        namespace["history"] = namespace["history_0"]


class IPythonOutputStream:
    """Normalize incremental Jupyter output messages and clear semantics."""

    def __init__(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.callback = callback
        self._pending_clear = False

    def feed(self, msg_type: str, content: dict[str, Any]) -> None:
        if msg_type == "clear_output":
            if bool(content.get("wait")):
                self._pending_clear = True
            else:
                self._clear()
            return

        event: dict[str, Any] | None = None
        if msg_type == "stream":
            event = {
                "type": "text",
                "kind": str(content.get("name", "stdout")),
                "text": str(content.get("text", "")),
            }
        elif msg_type in ("execute_result", "display_data", "update_display_data"):
            value = content.get("data", {}).get("text/plain")
            if isinstance(value, list):
                value = "".join(str(part) for part in value)
            if isinstance(value, str):
                event = {
                    "type": "text",
                    "kind": "result" if msg_type == "execute_result" else "display",
                    "text": value,
                }
        elif msg_type == "error":
            trace = content.get("traceback")
            if isinstance(trace, list):
                text = "\n".join(str(line) for line in trace)
            else:
                name = str(content.get("ename", "Error"))
                value = str(content.get("evalue", ""))
                text = f"{name}: {value}".rstrip()
            event = {"type": "text", "kind": "error", "text": text}

        if event is not None:
            if self._pending_clear:
                self._clear()
            if event["text"]:
                self.callback(event)

    def _clear(self) -> None:
        self._pending_clear = False
        self.callback({"type": "clear"})


def install_kernel_runtime(
    shell: Any,
    address: tuple[str, int] | list[Any],
    auth_token: str,
    *,
    timeout: float | None = None,
    execution_id: str | None = None,
    working_dir: str | None = None,
    client_key: str = "_RLM_CLIENT",
) -> RLMClient:
    """Install or restore one execution-scoped RLM scaffold in an IPython shell."""
    namespace = shell.user_ns
    client = namespace.get(client_key)
    if not isinstance(client, RLMClient):
        host, port = address
        if not isinstance(host, str) or type(port) is not int:
            raise TypeError("kernel RLM address must contain a string host and integer port")
        client = RLMClient((host, port), auth_token, timeout=timeout)
        namespace[client_key] = client
    client.install_ipython(shell)
    if working_dir is not None:
        os.chdir(working_dir)
    if execution_id is not None:
        client.prepare_execution(execution_id)
        client.activate_execution()
    custom_tools = namespace.get(_CUSTOM_TOOLS_KEY, {})
    if not isinstance(custom_tools, dict):
        raise RuntimeError("kernel custom-tool scaffold state is corrupted")
    install_scaffold(
        shell,
        client,
        answer=client.answer,
        async_api=client,
        custom_tools=custom_tools,
    )
    return client


def kernel_bootstrap_code(
    address: tuple[str, int],
    auth_token: str,
    timeout: float | None,
    *,
    execution_id: str | None = None,
    working_dir: str | None = None,
    client_key: str = "_RLM_CLIENT",
) -> str:
    """Return the small import-and-install cell used by a subprocess kernel."""
    validation = f"\n{client_key}.validate_execution()" if execution_id is not None else ""
    return (
        "from rlm.environments.ipython_kernel import "
        "install_kernel_runtime as _RLM_INSTALL_KERNEL\n"
        f"{client_key} = _RLM_INSTALL_KERNEL(\n"
        "    get_ipython(),\n"
        f"    {address!r},\n"
        f"    {auth_token!r},\n"
        f"    timeout={timeout!r},\n"
        f"    execution_id={execution_id!r},\n"
        f"    working_dir={working_dir!r},\n"
        f"    client_key={client_key!r},\n"
        ")"
        f"{validation}"
    )
