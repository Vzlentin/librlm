"""Harness-independent instructions for IPython consumers."""

import json
from importlib.resources import files


def load_ipython_prompt() -> dict[str, str]:
    prompt = json.loads(files(__package__).joinpath("ipython.json").read_text())
    if prompt.get("schema") != "librlm.ipython-prompt.v1" or not all(
        isinstance(prompt.get(key), str) and prompt[key]
        for key in ("api", "guidance", "promptSnippet")
    ):
        raise ValueError("Invalid shared IPython prompt")
    return prompt


def ipython_description(guidance: str | None = None) -> str:
    prompt = load_ipython_prompt()
    return prompt["api"] + "\n\n" + (prompt["guidance"] if guidance is None else guidance)
