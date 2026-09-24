# Shared RLM runtime and trace-driven instructions

The independent librlm checkout owns the Python API, async IPython primitives,
Jupyter stdio bridge, canonical instructions, and Hermes integration. The Pi
extension is a consumer. Its interpreter policy, model routing and UI stay in its
own repository.

## Extraction provenance

The Pi subtree was split at `41187970d0a7bebcaeb2db403fd2f6c3e061a11f` with its
history preserved. The split's missing Git object was recovered from the local
former submodule object database. Core files were unchanged during extraction.
`upstream` points to `https://github.com/alexzhang13/rlm.git`; the `extraction-source`
remote recorded the local source repository during extraction. The Pi repository retains its historical
subtree commits and now records deletion of the embedded copy.

## Consumers

`rlm/ipython_extension.py` is the shared kernel-side code. Loaded in a kernel,
it runs `AsyncRLMHost` there, binds the client as `rlm`, and forwards admitted
children to the harness socket (host protocol 2). Each non-silent cell is one
host execution, begun on `pre_run_cell` and ended with the cell's success on
`post_run_cell`. Handles survive successful cells; a failed cell discards its
children, which closes their harness connections. `execution_report()` returns
the last cell's summary and the executions released since the previous report.
A harness may set `next_execution_id` to name the next cell's execution.

Pi: `pi-ipython` owns the kernel, the generic `ipython` tool and its Python 3.12
runtime; it does not import librlm. `pi-rlm` hooks kernel start to add the socket
variables and load the extension from `RLM_LIBRLM_ROOT` or a managed clone of
this repository's `main`. It completes children through Pi's model registry and
appends `rlmApi` and `rlmGuidance` to the system prompt.

Hermes loads `integrations/hermes/ipython-rlm` from this checkout. Its runtime is
this checkout's `.venv`; it has no dependency on Pi files. The installed plugin
is a symlink to the canonical adapter. The adapter uses Hermes model routing for
tool-free child completions. `rlm/bridge.py` loads the extension in its kernel,
names each cell's execution after the adapter's request, and reads the report
through Jupyter `user_expressions`, so its `release` and `result` messages are
unchanged. The bridge protocol remains 6, host protocol 2 and internal async
protocol 5. These are separate protocols, not conflicting values.

`rlm/prompts/ipython.json` is the common model-facing contract. `api` describes
fixed runtime semantics and `guidance` describes how to apply them; `rlmApi` and
`rlmGuidance` are their RLM-specific tails. Each harness adds its own wrapper
parameters and actual runtime limits.

## Acceptance evidence

pi-ipython's `npm test` checks the kernel's state, reset, cleanup and startup
hook without other extensions. pi-rlm's `npm test` checks host cancellation,
provider-neutral child completion, librlm discovery and cross-cell handles.
Protocol compatibility is verified by executing the bridge, not matching source
strings or repeating implementation constants. Its model-backed acceptance
suite exercises state, usage, gather ownership, cancellation and cleanup. The
optional large-context benchmark is separate.

Core `pytest` checks the underlying runtime and the extension in a real kernel. Hermes adapter tests use actual
kernels with deterministic provider callbacks; native verification exercises
Hermes discovery and tool dispatch. Real Herdr Pi and Hermes sessions additionally
verify the imported source path, retained state and a real child completion.
