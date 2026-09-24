# Shared RLM runtime and trace-driven instructions

The independent librlm checkout owns the Python API, async IPython primitives,
Jupyter stdio bridge, canonical instructions, and Hermes integration. The Pi
extension is a consumer. Its interpreter policy, model routing and UI stay in its
own repository.

## Extraction provenance

The Pi subtree was split at `41187970d0a7bebcaeb2db403fd2f6c3e061a11f` with its
history preserved. The split's missing Git object was recovered from the local
former submodule object database. Core files were unchanged during extraction.
`upstream` points to `https://github.com/alexzhang13/rlm.git`; `extraction-source`
records the local source repository. The Pi repository retains its historical
subtree commits and now records deletion of the embedded copy.

## Consumers

Pi resolves `~/Dev/librlm` or an explicit absolute `RLM_LIBRLM_ROOT` (`~/...`
is accepted). The default is independent of the Pi package install location;
there is no sibling or bundled-copy fallback.
`extensions/ipython.py` retains the extension-owned Python 3.12 kernelspec and
delegates to `rlm.bridge.main`. Both bridge and kernel import the independent
checkout. The installed Pi package source points to the development extension;
the old Git cache was retained so existing sessions can finish before `/reload`.

Hermes loads `integrations/hermes/ipython-rlm` from this checkout. Its runtime is
this checkout's `.venv`; it has no dependency on Pi files. The installed plugin
is a symlink to the canonical adapter. The adapter uses Hermes model routing for
tool-free child completions. The bridge protocol remains 6, host protocol 2 and
internal async protocol 5. These are separate protocols, not conflicting values.

`rlm/prompts/ipython.json` is the common model-facing contract. `api` describes
fixed runtime semantics and `guidance` describes how to apply them. Each harness
adds its own wrapper parameters and actual runtime limits.

## Acceptance evidence

Pi's `npm test` checks TypeScript, host cancellation, provider-neutral child
completion, relocated librlm discovery, and actual kernel state/reset/cleanup.
Protocol compatibility is verified by executing the bridge, not matching source
strings or repeating implementation constants. Its model-backed acceptance
suite exercises state, usage, gather ownership, cancellation and cleanup. The
optional large-context benchmark is separate.

Core `pytest` checks the underlying runtime. Hermes adapter tests use actual
kernels with deterministic provider callbacks; native verification exercises
Hermes discovery and tool dispatch. Real Herdr Pi and Hermes sessions additionally
verify the imported source path, retained state and a real child completion.
