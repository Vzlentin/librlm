# Hermes IPython/RLM plugin

Native Hermes consumer of the standalone librlm Jupyter bridge. Tool descriptions load `rlm/prompts/ipython.json`. The personal shared Pi/Hermes `rlm` skill lives directly in `~/.agents/skills/rlm`. Both harnesses discover that directory; no skill links or prompt aliases are needed. The skill is user configuration rather than part of this plugin's source package.

## Installation

From the librlm checkout, provision the shared interpreter with `uv sync --extra ipython --group test`. Symlink `integrations/hermes/ipython-rlm` into the active profile's `plugins/ipython-rlm`, preserving any existing plugin directory as a backup outside the discovery directory. Enable with `hermes plugins enable ipython-rlm --no-allow-tool-override`. The owning Hermes process must rediscover plugins to receive changes; starting a new chat in an existing desktop backend does not force discovery. No Hermes core edits or extra Hermes interpreter packages are needed.

The runtime checkout defaults to `~/Dev/librlm`: `rlm/bridge.py` plus `.venv/bin/python`. The default remains stable when Hermes copies the plugin into a temporary validation directory. Override with `hermes config set plugins.entries.ipython-rlm.settings.runtime_repo /absolute/checkout`. The adapter requires bridge protocol 6 / host protocol 2 and fails on mismatches.

## Tool

`ipython(action='execute', code='...', cwd='/absolute/initial/directory', timeout=120, max_child_calls=0)`

- True Jupyter/IPython state per host chat identity; top-level `await` and native magics.
- `status` does not start a kernel; `reset` explicitly destroys state.
- Child calls default to zero. Explicitly allow up to four per cell, using `h = await rlm.spawn(task, context=text)` then `reports = await rlm.gather([h])`. Handles survive successful cells; a later gather retains its originating budget, deadline, and host context.
- Children are tool-free completions through Hermes `ctx.llm.complete`, not Pi agent sessions. No model/provider/profile override or parent conversation is passed.
- Up to four open chat kernels per owner process. State does not survive host restarts. Ordinary cell exceptions retain namespace; timeout discards it and requires reset before further work.
- Initial cwd is reinstated for each cell; `%cd` is cell-local in this adapter. Reset for a different initial cwd.
- Full cell events and admitted child requests/results stay in the active profile's `cache/ipython-rlm/`. Stdout returned to chat is capped at 12,000 characters and the full tool response at 16,000; `audit_path` identifies the full cell log. Raw evidence can be private. Actual child results survive audit write failures in memory; `uncheckpointed_children` exposes request IDs, and later cells retry only their checkpoints.

## Safety and limitations

POSIX/local-only, NOT a filesystem/network sandbox. The handler refuses a configured remote/sandbox terminal backend and isolated task override rather than executing on the wrong machine. Keep child budget zero for trace mining; this budget does not disable arbitrary Python network access. Follow the user's scope and preserve original sources.

No automatic retry or replay. Stopping a local kernel does not guarantee revocation of an already submitted provider request; the host/provider timeout remains relevant. Returned provider costs can be unknown, not zero.

## Model-free verification

```text
~/Dev/librlm/.venv/bin/python -m pytest ~/.hermes/plugins/ipython-rlm/tests/test_ipython.py -q
~/.hermes/hermes-agent/venv/bin/python ~/.hermes/plugins/ipython-rlm/tests/verify_native.py
hermes plugins doctor ipython-rlm --ci
```

Adjust paths for a non-default profile. The native probe exercises actual Hermes discovery/dispatch and explicitly mocks the provider method; it performs no live provider calls. Live-model acceptance is separate and has not been run. Keep this distinction in reports.
