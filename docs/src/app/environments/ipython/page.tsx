import { CodeBlock } from "@/components/CodeBlock";
import { Table } from "@/components/Table";

export default function IPythonPage() {
  return (
    <div>
      <h1 className="text-3xl font-bold mb-4">IPythonREPL</h1>

      <p className="text-xl text-muted-foreground mb-6 leading-relaxed">
        <strong className="text-foreground">IPythonREPL</strong> executes code inside a real{" "}
        <a href="https://ipython.org/" className="text-primary hover:underline font-medium">IPython</a>{" "}
        session instead of plain <code className="px-1.5 py-0.5 rounded bg-muted text-foreground text-sm font-semibold">exec()</code>.
        It supports two kernel modes — an <strong className="text-foreground">in-process</strong> shell that
        runs in the same Python process as the RLM (the default, fastest), and a{" "}
        <strong className="text-foreground">subprocess</strong> kernel that runs a real{" "}
        <code className="px-1.5 py-0.5 rounded bg-muted text-foreground text-sm font-semibold">ipykernel</code> in a separate
        Python process for hard cell timeouts and full namespace isolation from the RLM host.
        Both modes give the LM access to IPython&apos;s full surface (cell magics, rich
        repr, line tracebacks).
      </p>

      <p className="text-muted-foreground mb-4">
        <strong>Prerequisite:</strong> install the optional extra:
      </p>
      <CodeBlock language="bash" code={`pip install 'rlms[ipython]'
# or with uv:
# uv pip install -e ".[ipython]"`} />

      <CodeBlock code={`from rlm import RLM

# In-process (default kernel_mode): same process, fast.
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-mini"},
    environment="ipython",
    environment_kwargs={
        "kernel_mode": "in_process",
        "cell_timeout": 30,         # SIGALRM-based; Unix main thread only
    },
)

# Subprocess: separate Python process, hard timeouts, full isolation.
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-mini"},
    environment="ipython",
    environment_kwargs={
        "kernel_mode": "subprocess",
        "cell_timeout": 30,         # Hard guarantee via interrupt_kernel
        "startup_timeout": 60,
        "max_concurrent_subcalls": 4,
    },
)`} />

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">Arguments</h2>
      <Table
        headers={["Argument", "Type", "Default", "Description"]}
        rows={[
          [<code key="1">kernel_mode</code>, <code key="2">&quot;in_process&quot; | &quot;subprocess&quot;</code>, <code key="3">&quot;in_process&quot;</code>, "Where the IPython session runs"],
          [<code key="4">cell_timeout</code>, <code key="5">float | None</code>, <code key="6">None</code>, "Per-cell timeout in seconds. In-process mode requires the Unix main thread and no active external ITIMER_REAL"],
          [<code key="7">startup_timeout</code>, <code key="8">float</code>, <code key="9">60.0</code>, "Subprocess kernel boot timeout"],
          [<code key="10">subcall_timeout</code>, <code key="11">float | None</code>, <code key="12">None</code>, "Per-request kernel→host socket timeout (subprocess)"],
          [<code key="13">max_concurrent_subcalls</code>, <code key="14">int</code>, <code key="15">4</code>, "Global cap on concurrent subcall_fn invocations"],
          [<code key="16">setup_code</code>, <code key="17">str</code>, <code key="18">None</code>, "Code to run at initialization"],
          [<code key="19">custom_tools</code>, <code key="20">dict</code>, <code key="21">None</code>, "Functions / values injected into the namespace"],
          [<code key="22">working_dir</code>, <code key="23">str | None</code>, <code key="24">None</code>, "Kernel cwd only; context and history transport use private REPL storage"],
        ]}
      />

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">In-process vs. subprocess</h2>
      <Table
        headers={["", "in_process", "subprocess"]}
        rows={[
          ["Process", "Same as host", "Separate Python via ipykernel"],
          ["Subcall path", "Direct Python call", "Authenticated TCP host (4-byte length-prefixed JSON)"],
          [<><code key="t1">cell_timeout</code></>, "Terminal SIGALRM interruption on the Unix main thread", <>Interrupt, drain to idle, then restart if needed</>],
          ["Recursive batches", "Sequential on the cell thread", "Concurrent, including when cell_timeout is set"],
          ["Cell magics (%%timeit, …)", "Yes", "Yes"],
          [<><code key="t3">input()</code></>, "Disabled and restored before each cell", "Disabled and restored before each cell"],
          ["Isolation from host", "Shares stdout/stderr/cwd/SIGALRM", "Full process isolation"],
          ["Custom tool injection", "Direct namespace inject, restored before each cell", <>Pickled with <code key="t5">dill</code> over ZMQ, restored before each cell</>],
        ]}
      />

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">How It Works</h2>
      <h3 className="text-lg font-medium mt-4 mb-2">In-process</h3>
      <ol className="list-decimal list-inside text-muted-foreground space-y-1 mb-4">
        <li>Creates a fresh <code>InteractiveShell</code> with a per-instance user module (so multiple in-process REPLs don&apos;t share <code>sys.modules[&apos;__main__&apos;]</code>) and IPython&apos;s history database disabled.</li>
        <li>Injects synchronous scaffold helpers (<code>llm_query</code>, <code>rlm_query</code>, <code>answer</code>, <code>SHOW_VARS</code>) and a stubbed <code>input()</code> into <code>user_ns</code>. The async <code>rlm</code> handle API is subprocess-only.</li>
        <li><code>execute_code</code> runs each cell via <code>shell.run_cell</code> under a process-wide reentrant owner because cwd, stdout, stderr, and signal timers are process globals. Cleanup waits for the active cell before detaching its kernel.</li>
        <li><code>rlm_query</code> calls <code>subcall_fn</code> directly, gated by a per-instance semaphore. Recursive batches stay on the cell thread and run sequentially so nested in-process children can reenter the process-wide owner without worker-thread deadlock.</li>
      </ol>

      <h3 className="text-lg font-medium mt-4 mb-2">Subprocess</h3>
      <ol className="list-decimal list-inside text-muted-foreground space-y-1 mb-4">
        <li>Starts an authenticated <code>AsyncRLMHost</code> on <code>127.0.0.1:0</code> (ephemeral port).</li>
        <li>Launches an <code>ipykernel</code> subprocess pinned to the host&apos;s <code>sys.executable</code> (so it inherits the same site-packages — important for <code>dill</code>, custom imports, etc.).</li>
        <li>Bootstraps one <code>RLMClient</code> that routes compatibility queries, async child handles, and final values to the host over the canonical 4-byte-prefixed JSON protocol.</li>
        <li>Each user cell gets a unique execution id. Internal bootstrap, context, history, and custom-tool assignments use control cells that do not create user executions. The client activates the user id in the cell&apos;s <code>ContextVar</code> and attaches it to every request.</li>
        <li><code>cell_timeout</code> interrupts the kernel, drains messages until the kernel reports idle, and restarts the kernel if it ignores interruption. Ending a failed cell cancels queued query work and cooperatively signals active child and query runners.</li>
      </ol>

      <pre className="text-sm">{`┌────────────────────────────────────────────┐
│ Host (RLM process)                         │
│  ┌────────────────┐    ┌────────────────┐  │
│  │ AsyncRLMHost   │───►│ LM / RLM      │  │
│  │ lifecycle,     │    │ runners        │  │
│  │ handles, final │    └────────────────┘  │
│  └───────▲────────┘                        │
└──────────┼─────────────────────────────────┘
           │ authenticated framed TCP
┌──────────┼─────────────────────────────────┐
│ ipykernel subprocess                       │
│  ┌───────┴────────┐                        │
│  │ RLMClient     │ query / spawn / gather │
│  │ + IPython     │ release / final         │
│  │ ContextVar    │                         │
│  └────────────────┘                        │
└────────────────────────────────────────────┘`}</pre>

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">Notable behavior</h2>
      <ul className="list-disc list-inside text-muted-foreground space-y-2">
        <li><strong className="text-foreground">Lifecycle and process-global serialization.</strong> Each instance serializes execution and finalization. Cleanup rejects new operations immediately. Subprocess cleanup interrupts an active cell even when <code>cell_timeout</code> is disabled, while in-process cleanup waits for the cell because Python cannot safely stop that thread. A process-wide reentrant owner also prevents different in-process instances from overlapping cwd and stream redirection.</li>
        <li><strong className="text-foreground">Namespace restoration.</strong> Scaffold helpers, <code>input</code>, context and history aliases, and custom tool bindings are restored before every cell in both modes. Rebinding one of these names affects only the current cell. Mutating an object exposed through a custom tool still mutates that object. <code>IPythonREPL.locals</code> is a compatibility view: in-process mode returns a namespace snapshot, while subprocess mode returns only host-assigned state retained for kernel restart.</li>
        <li><strong className="text-foreground">Global subcall cap.</strong> <code>max_concurrent_subcalls</code> bounds all in-flight <code>subcall_fn</code> invocations on the instance, including recursive compatibility queries and async children. Plain LM batch queries use separately tracked host operations. <code>rlm.spawn()</code> still returns a live handle immediately and queues child execution when every subcall slot is busy. <code>max_budget</code> remains postpaid: children receive the latest observed remainder, but concurrent calls do not reserve spend and can exceed the limit before the next check.</li>
        <li><strong className="text-foreground">Reentry guard.</strong> If <code>subcall_fn</code> calls <code>execute_code</code> back on the parent REPL (or a cell traverses <code>rlm_query.__self__.execute_code(…)</code> in in-process mode), the call raises <code>RuntimeError</code> instead of deadlocking the cell lock or corrupting the in-flight cell&apos;s tracking. <code>subcall_fn</code> should spawn a child REPL.</li>
        <li><strong className="text-foreground">Execution attribution.</strong> Async tasks retain the originating execution context, so late work is rejected after that execution ends instead of adopting a later cell. Raw <code>threading.Thread</code> workers do not inherit a cell binding and are rejected. Use <code>asyncio.to_thread()</code> when a cell needs thread-backed work with the active execution context.</li>
        <li><strong className="text-foreground">Typed child results.</strong> <code>rlm.gather()</code> returns validated <code>ChildResult</code> objects. Each result supports attributes such as <code>result.text</code> and mapping access such as <code>result[&quot;text&quot;]</code>. <code>ChildExecution</code> owns admission, cancellation, teardown, and settlement before the host delivers a result. Standalone adapters return <code>ChildOutcome.external(...)</code> and may use any string-keyed canonical JSON object as the usage shape. A subprocess-only <code>async_child_runner</code> attached to <code>IPythonREPL</code> returns one <code>RLMChatCompletion</code>; the host derives wire usage from that completion.</li>
        <li><strong className="text-foreground">Single delivery.</strong> A successful gather consumes its handles. Lost gather and cleanup responses remain recoverable without attributing usage twice. The client caches delivered values until cleanup succeeds.</li>
        <li><strong className="text-foreground">In-process is not isolated.</strong> Two in-process instances each get a unique <code>__main__</code> substitute, but they still share the host&apos;s stdout/stderr/cwd/SIGALRM. Their cells are serialized to prevent contamination. Use <code>subprocess</code> for true isolation or parallel cells.</li>
      </ul>

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">When to use which mode</h2>
      <ul className="list-disc list-inside text-muted-foreground space-y-1.5">
        <li><strong className="text-foreground">in_process</strong> is the fastest path for trusted code, development, and short-lived cells. <code>cell_timeout</code> is terminal on the Unix main thread, rejects an active external timer, and recursive batches always run sequentially.</li>
        <li><strong className="text-foreground">subprocess</strong> is the choice for a hard <code>cell_timeout</code> guarantee, concurrent recursive batches under that timeout, or full namespace, signal, and cwd isolation between the LM&apos;s code and the RLM host.</li>
      </ul>
    </div>
  );
}
