import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from rlm.clients import BaseLM, get_client
from rlm.core.accounting import CompletionTransaction, UsageLedger
from rlm.core.child_execution import (
    Cancellation,
    ChildExecution,
    ChildOutcome,
    ChildRequest,
)
from rlm.core.child_execution import (
    PreparedChild as _PreparedChild,
)
from rlm.core.lm_handler import LMHandler
from rlm.core.types import (
    ClientBackend,
    CodeBlock,
    EnvironmentType,
    FinalValue,
    JSONValue,
    ModelUsageSummary,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
    UsageSummary,
)
from rlm.environments import (
    BaseEnv,
    SupportsCompaction,
    SupportsCustomTools,
    SupportsPersistence,
    get_environment,
    get_environment_capabilities,
)
from rlm.logger.rlm_logger import RLMLogger
from rlm.utils.exceptions import (
    BudgetExceededError,
    CancellationError,
    ErrorThresholdExceededError,
    TimeoutExceededError,
    TokenLimitExceededError,
)
from rlm.utils.parsing import (
    find_code_blocks,
    format_iteration,
)
from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)
from rlm.utils.rlm_utils import filter_sensitive_keys
from rlm.utils.token_utils import count_tokens, get_context_limit


def _completion_response(value: JSONValue) -> str:
    """Render a structured final without widening the text completion interface."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


@dataclass(slots=True)
class _CompletionBody:
    final: FinalValue
    response: str
    iteration_count: int
    message_history: list[dict[str, Any]]


class RLM:
    """
    Recursive Language Model class that the user instantiates and runs on their tasks.

    Each completion() call spawns its own environment and LM handler, which are
    cleaned up when the call completes.
    """

    def __init__(
        self,
        backend: ClientBackend = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        environment: EnvironmentType = "local",
        environment_kwargs: dict[str, Any] | None = None,
        depth: int = 0,
        max_depth: int = 1,
        max_iterations: int = 30,
        max_budget: float | None = None,
        max_timeout: float | None = None,
        max_tokens: int | None = None,
        max_errors: int | None = None,
        custom_system_prompt: str | None = None,
        other_backends: list[ClientBackend] | None = None,
        other_backend_kwargs: list[dict[str, Any]] | None = None,
        logger: RLMLogger | None = None,
        verbose: bool = False,
        persistent: bool = False,
        custom_tools: dict[str, Any] | None = None,
        custom_sub_tools: dict[str, Any] | None = None,
        compaction: bool = False,
        compaction_threshold_pct: float = 0.85,
        max_concurrent_subcalls: int = 4,
        on_subcall_start: Callable[[int, str, str], None] | None = None,
        on_subcall_complete: Callable[[int, str, float, str | None], None] | None = None,
        on_iteration_start: Callable[[int, int], None] | None = None,
        on_iteration_complete: Callable[[int, int, float], None] | None = None,
        sampling_args: dict[str, Any] | None = None,
        sub_sampling_args: dict[str, Any] | None = None,
        orchestrator: bool = True,
        user_prologue: str | None = None,
    ):
        """
        Args:
            backend: The backend to use for the RLM.
            backend_kwargs: The kwargs to pass to the backend.
            environment: The environment to use for the RLM.
            environment_kwargs: The kwargs to pass to the environment.
            depth: The current depth of the RLM (0-indexed).
            max_depth: The maximum depth of recursion. When depth >= max_depth, falls back to plain LM completion.
            max_iterations: The maximum number of iterations of the RLM.
            max_budget: Postpaid budget in USD. Execution stops after observed usage exceeds it; concurrent calls do not reserve spend. Requires a cost-tracking backend (e.g., OpenRouter).
            max_timeout: Maximum execution time in seconds. Execution stops if exceeded, returning best answer if available.
            max_tokens: Maximum total tokens (input + output). Execution stops if exceeded, returning best answer if available.
            max_errors: Maximum consecutive errors before stopping. Execution stops if exceeded, returning best answer if available.
            custom_system_prompt: The custom system prompt to use for the RLM.
            other_backends: A list of other client backends that the environments can use to make sub-calls.
            other_backend_kwargs: The kwargs to pass to the other client backends (ordered to match other_backends).
            logger: The logger to use for the RLM.
            verbose: Whether to print verbose output in rich to console.
            persistent: If True, reuse the environment across completion() calls for multi-turn conversations.
            custom_tools: Dict of custom functions/tools available in the REPL. Keys are function names,
                values are callable functions. These are injected into the REPL globals.
            custom_sub_tools: Dict of custom tools for child RLMs (rlm_query calls). If None, inherits
                from custom_tools. Pass an empty dict {} to disable tools for sub-agents.
            compaction: If True, keep full root model history in REPL variable `history` and compact
                when root context reaches compaction_threshold_pct of the model's context limit.
            compaction_threshold_pct: When compaction is on, trigger summarization when root
                message token count reaches this fraction of the model context limit (default 0.85).
            max_concurrent_subcalls: Maximum number of parallel threads for rlm_query_batched subcalls.
                Each child RLM runs in its own thread. Default 4.
            on_subcall_start: Callback fired when a child RLM starts. Args: (depth, model, prompt_preview).
            on_subcall_complete: Callback fired when a child RLM completes. Args: (depth, model, duration, error_or_none).
            on_iteration_start: Callback fired when an iteration starts. Args: (depth, iteration_num).
            on_iteration_complete: Callback fired when an iteration completes. Args: (depth, iteration_num, duration).
        """
        # Sampling args plumbed into backend_kwargs / other_backend_kwargs
        # before the clients are constructed, so they reach the chat-completions
        # call (e.g. temperature, top_p, max_tokens, seed). ``sampling_args``
        # applies to the root model (depth=0); ``sub_sampling_args`` to
        # depth=1 sub-LLM calls. If ``sub_sampling_args`` is set without an
        # ``other_backends``, we mirror the root backend so depth=1 routes
        # through a separate client with its own sampling args.
        if sampling_args is not None:
            backend_kwargs = dict(backend_kwargs or {})
            existing = dict(backend_kwargs.get("sampling_args") or {})
            existing.update(sampling_args)
            backend_kwargs["sampling_args"] = existing
        if sub_sampling_args is not None:
            if other_backends is None:
                other_backends = [backend]
                other_backend_kwargs = [dict(backend_kwargs or {})]
            else:
                other_backend_kwargs = [dict(kw or {}) for kw in (other_backend_kwargs or [{}])]
            first = dict(other_backend_kwargs[0])
            existing = dict(first.get("sampling_args") or {})
            existing.update(sub_sampling_args)
            first["sampling_args"] = existing
            other_backend_kwargs[0] = first

        # Store config for spawning per-completion
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.environment_type = environment
        self.environment_kwargs = (
            environment_kwargs.copy() if environment_kwargs is not None else {}
        )
        self._environment_capabilities = get_environment_capabilities(environment)
        # Validate other_backends: currently only support one additional backend
        if other_backends is not None:
            if len(other_backends) != 1:
                raise ValueError(
                    "We currently only support one additional backend for the recursive sub-calls! "
                    "This model will be the model used for recursive sub-calls, but this will change in the future"
                )

        self.other_backends = other_backends
        self.other_backend_kwargs = other_backend_kwargs

        # Custom tools: functions available in the REPL environment
        self.custom_tools = custom_tools
        # Sub-tools: if None, inherit from custom_tools; if {}, no tools for sub-agents
        self.custom_sub_tools = custom_sub_tools if custom_sub_tools is not None else custom_tools

        self.compaction = compaction
        self.compaction_threshold_pct = compaction_threshold_pct
        self.max_concurrent_subcalls = max_concurrent_subcalls

        self.depth = depth
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.max_budget = max_budget
        self.max_timeout = max_timeout
        self.max_tokens = max_tokens
        self.max_errors = max_errors
        self.system_prompt = custom_system_prompt if custom_system_prompt else RLM_SYSTEM_PROMPT
        self.orchestrator = orchestrator
        # Optional user-prologue message inserted between the metadata user
        # message and the iter-0 turn prompt. Mirrors RLMTrainEnv's
        # ``user_prologue`` so canonical inference can match envs that
        # depend on a task-specific tips message (e.g. BC+).
        self.user_prologue = user_prologue
        self.logger = logger
        from rlm.logger.verbose import VerbosePrinter

        self.verbose = VerbosePrinter(enabled=verbose)

        # Event callbacks for live tree display
        self.on_subcall_start = on_subcall_start
        self.on_subcall_complete = on_subcall_complete
        self.on_iteration_start = on_iteration_start
        self.on_iteration_complete = on_iteration_complete

        # Tracking for the active completion.
        self._usage_ledger = UsageLedger(max_budget)
        self._child_execution = ChildExecution(
            self._prepare_child,
            settle=self._settle_child_outcome,
            max_concurrent=max_concurrent_subcalls,
        )
        self._consecutive_errors: int = 0
        self._last_error: str | None = None
        self._best_partial_answer: str | None = None
        self._completion_start_time: float | None = None  # Set when completion() starts

        # Persistence support
        self.persistent = persistent
        self._persistent_env: BaseEnv | None = None
        self._validate_environment_capabilities()

        # Log metadata if logger is provided
        if self.logger or verbose:
            metadata = RLMMetadata(
                root_model=backend_kwargs.get("model_name", "unknown")
                if backend_kwargs
                else "unknown",
                max_depth=max_depth,
                max_iterations=max_iterations,
                backend=backend,
                backend_kwargs=filter_sensitive_keys(backend_kwargs) if backend_kwargs else {},
                environment_type=environment,
                environment_kwargs=filter_sensitive_keys(environment_kwargs)
                if environment_kwargs
                else {},
                other_backends=other_backends,
            )
            if self.logger:
                self.logger.log_metadata(metadata)
            self.verbose.print_metadata(metadata)

    def _spawn_completion_resources(
        self,
        prompt: str | dict[str, Any],
    ) -> tuple[LMHandler, BaseEnv]:
        """Create one completion's resources, rolling back partial startup."""
        client: BaseLM = get_client(self.backend, self.backend_kwargs)
        other_backend_client: BaseLM | None = None
        if self.other_backends and self.other_backend_kwargs:
            other_backend_client = get_client(self.other_backends[0], self.other_backend_kwargs[0])

        lm_handler = LMHandler(client, other_backend_client=other_backend_client)
        if other_backend_client is not None:
            lm_handler.register_client(other_backend_client.model_name, other_backend_client)
            for backend, kwargs in zip(
                self.other_backends[1:],
                self.other_backend_kwargs[1:],
                strict=True,
            ):
                other_client = get_client(backend, kwargs)
                lm_handler.register_client(other_client.model_name, other_client)

        environment: BaseEnv | None = None
        try:
            lm_handler.start()
            self._usage_ledger.bind_root(lm_handler.get_usage_summary)
            if self.persistent and self._persistent_env is not None:
                environment = self._persistent_env
                persistent_environment = cast(SupportsPersistence, environment)
                persistent_environment.update_handler_address(lm_handler.address)
                persistent_environment.add_context(prompt)
            else:
                env_kwargs = self.environment_kwargs.copy()
                env_kwargs["lm_handler_address"] = lm_handler.address
                env_kwargs["context_payload"] = prompt
                env_kwargs["depth"] = self.depth + 1
                if self._environment_capabilities.recursive_subcalls and self.max_depth > 1:
                    env_kwargs["subcall_fn"] = self._child_execution
                if self.custom_tools is not None:
                    env_kwargs["custom_tools"] = self.custom_tools
                if self.compaction:
                    env_kwargs["compaction"] = True
                if self._environment_capabilities.recursive_subcalls:
                    env_kwargs["max_concurrent_subcalls"] = self.max_concurrent_subcalls
                environment = get_environment(self.environment_type, env_kwargs)
                self._validate_environment_instance(environment)
                if self.persistent:
                    self._persistent_env = environment
            return lm_handler, environment
        except BaseException as startup_error:
            rollback_errors: list[BaseException] = []
            try:
                lm_handler.stop()
            except BaseException as error:
                rollback_errors.append(error)
            if environment is not None:
                try:
                    environment.cleanup()
                except BaseException as error:
                    rollback_errors.append(error)
                else:
                    if environment is self._persistent_env:
                        self._persistent_env = None
            for error in rollback_errors:
                startup_error.add_note(f"resource rollback also failed: {error!r}")
            raise

    def _cleanup_completion_environment(self, environment: BaseEnv) -> None:
        """Clean a non-persistent completion environment."""
        if not self.persistent:
            environment.cleanup()

    def _setup_prompt(
        self,
        prompt: str | dict[str, Any],
        root_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Setup the system prompt for the RLM. Also include metadata about the prompt and build
        up the initial message history.
        """
        metadata = QueryMetadata(prompt)
        message_history = build_rlm_system_prompt(
            system_prompt=self.system_prompt,
            query_metadata=metadata,
            custom_tools=self.custom_tools,
            root_prompt=root_prompt,
            orchestrator=self.orchestrator,
        )
        if self.user_prologue:
            message_history.append({"role": "user", "content": self.user_prologue})
        if self.compaction:
            message_history[0]["content"] += (
                "\n\nThe full conversation history (trajectory segments and any summaries) "
                "is available in the REPL variable `history` as a list."
            )
        return message_history

    def completion(
        self, prompt: str | dict[str, Any], root_prompt: str | None = None
    ) -> RLMChatCompletion:
        """Run one RLM completion with atomic accounting and finalization."""
        time_start = time.perf_counter()
        self._completion_start_time = time_start
        self._consecutive_errors = 0
        self._last_error = None
        self._best_partial_answer = None
        self._usage_ledger.reset(self.max_budget)

        if self.depth >= self.max_depth:
            return self._fallback_answer(prompt)

        if self.logger:
            self.logger.clear_iterations()

        lm_handler, environment = self._spawn_completion_resources(prompt)
        transaction = CompletionTransaction(
            finalize_environment=environment.finalize_completion,
            stop_handler=lm_handler.stop,
            cleanup_environment=(lambda: self._cleanup_completion_environment(environment)),
            ledger=self._usage_ledger,
            check_limits=self._check_final_usage_limits,
        )
        body, usage = transaction.execute(
            lambda: self._run_completion_body(
                prompt,
                root_prompt,
                lm_handler,
                environment,
                time_start,
            )
        )

        if self.persistent:
            cast(SupportsPersistence, environment).add_history(body.message_history)

        time_end = time.perf_counter()
        self.verbose.print_final_answer(
            body.final.value if body.final.is_present else body.response
        )
        self.verbose.print_summary(
            body.iteration_count,
            time_end - time_start,
            usage.to_dict(),
        )
        return RLMChatCompletion(
            root_model=self.backend_kwargs.get("model_name", "unknown")
            if self.backend_kwargs
            else "unknown",
            prompt=prompt,
            response=body.response,
            usage_summary=usage,
            execution_time=time_end - time_start,
            metadata=self.logger.get_trajectory() if self.logger else None,
            final=body.final,
        )

    def _run_completion_body(
        self,
        prompt: str | dict[str, Any],
        root_prompt: str | None,
        lm_handler: LMHandler,
        environment: BaseEnv,
        time_start: float,
    ) -> _CompletionBody:
        """Run the linear iteration body owned by a completion transaction."""
        message_history = self._setup_prompt(prompt, root_prompt=root_prompt)
        completion_final = FinalValue.absent()
        response = ""
        iteration_count = self.max_iterations
        compaction_count = 0
        persistent_environment = cast(SupportsPersistence, environment) if self.persistent else None
        compaction_environment = cast(SupportsCompaction, environment) if self.compaction else None

        try:
            for i in range(self.max_iterations):
                self._check_timeout(i, time_start)

                if compaction_environment is not None:
                    current_tokens, threshold_tokens, max_tokens = self._get_compaction_status(
                        message_history
                    )
                    self.verbose.print_compaction_status(
                        current_tokens,
                        threshold_tokens,
                        max_tokens,
                    )
                    if current_tokens >= threshold_tokens:
                        compaction_count += 1
                        self.verbose.print_compaction()
                        message_history = self._compact_history(
                            lm_handler,
                            compaction_environment,
                            message_history,
                            compaction_count,
                        )

                context_count = (
                    persistent_environment.get_context_count()
                    if persistent_environment is not None
                    else 1
                )
                history_count = (
                    persistent_environment.get_history_count()
                    if persistent_environment is not None
                    else 0
                )
                message_history.append(
                    build_user_prompt(
                        iteration=i,
                        context_count=context_count,
                        history_count=history_count,
                        max_iterations=self.max_iterations,
                    )
                )

                iteration_number = i + 1
                if self.on_iteration_start is not None:
                    try:
                        self.on_iteration_start(self.depth, iteration_number)
                    except Exception:
                        pass
                iteration_started = time.perf_counter()
                try:
                    iteration = self._completion_turn(
                        prompt=message_history,
                        lm_handler=lm_handler,
                        environment=environment,
                    )
                finally:
                    if self.on_iteration_complete is not None:
                        try:
                            self.on_iteration_complete(
                                self.depth,
                                iteration_number,
                                time.perf_counter() - iteration_started,
                            )
                        except Exception:
                            pass
                self._check_iteration_limits(iteration, i)

                final = next(
                    (
                        block.result.final
                        for block in iteration.code_blocks
                        if block.result.final.is_present
                    ),
                    FinalValue.absent(),
                )
                iteration.final = final

                if iteration.response and iteration.response.strip():
                    self._best_partial_answer = iteration.response
                if self.logger:
                    self.logger.log(iteration)
                self.verbose.print_iteration(iteration, i + 1)

                if final.is_present:
                    completion_final = final
                    response = _completion_response(final.value)
                    iteration_count = i + 1
                    break

                new_messages = format_iteration(iteration)
                message_history.extend(new_messages)
                if compaction_environment is not None:
                    compaction_environment.append_compaction_entry(new_messages)
        except KeyboardInterrupt:
            self.verbose.print_limit_exceeded("cancelled", "User interrupted execution")
            raise CancellationError(
                partial_answer=self._best_partial_answer,
                message="Execution cancelled by user (Ctrl+C)",
            ) from None

        if not completion_final.is_present:
            response = self._default_answer(message_history, lm_handler)

        return _CompletionBody(
            completion_final,
            response,
            iteration_count,
            message_history,
        )

    def _check_final_usage_limits(self, usage: UsageSummary) -> None:
        """Check usage settled while the environment was being finalized."""
        current_cost = usage.total_cost or 0.0
        if self.max_budget is not None and current_cost > self.max_budget:
            self.verbose.print_budget_exceeded(current_cost, self.max_budget)
            raise BudgetExceededError(
                spent=current_cost,
                budget=self.max_budget,
                message=(
                    f"Budget exceeded after completion: spent ${current_cost:.6f} "
                    f"of ${self.max_budget:.6f} budget"
                ),
            )
        total_tokens = usage.total_input_tokens + usage.total_output_tokens
        if self.max_tokens is not None and total_tokens > self.max_tokens:
            self.verbose.print_limit_exceeded(
                "tokens", f"{total_tokens:,} of {self.max_tokens:,} tokens"
            )
            raise TokenLimitExceededError(
                tokens_used=total_tokens,
                token_limit=self.max_tokens,
                partial_answer=self._best_partial_answer,
                message=(
                    f"Token limit exceeded after completion: {total_tokens:,} "
                    f"of {self.max_tokens:,} tokens"
                ),
            )

    def _check_timeout(self, iteration: int, time_start: float) -> None:
        """Raise TimeoutExceededError if the timeout has been exceeded."""
        if self.max_timeout is None:
            return
        elapsed = time.perf_counter() - time_start
        if elapsed > self.max_timeout:
            self.verbose.print_limit_exceeded(
                "timeout",
                f"{elapsed:.1f}s of {self.max_timeout:.1f}s",
            )
            raise TimeoutExceededError(
                elapsed=elapsed,
                timeout=self.max_timeout,
                partial_answer=self._best_partial_answer,
                message=(
                    f"Timeout exceeded after iteration {iteration}: "
                    f"{elapsed:.1f}s of {self.max_timeout:.1f}s limit"
                ),
            )

    def _check_iteration_limits(self, iteration: RLMIteration, iteration_num: int) -> None:
        """Check error tracking, budget, and token limits after an iteration.

        Raises ErrorThresholdExceededError, BudgetExceededError, or TokenLimitExceededError
        if the respective limits are exceeded.
        """
        # Track errors from code execution (check stderr for errors)
        iteration_had_error = False
        for code_block in iteration.code_blocks:
            if code_block.result and code_block.result.stderr:
                iteration_had_error = True
                self._last_error = code_block.result.stderr
                break

        if iteration_had_error:
            self._consecutive_errors += 1
        else:
            self._consecutive_errors = 0  # Reset on success

        # Check error threshold
        if self.max_errors is not None and self._consecutive_errors >= self.max_errors:
            self.verbose.print_limit_exceeded(
                "errors",
                f"{self._consecutive_errors} consecutive errors (limit: {self.max_errors})",
            )
            raise ErrorThresholdExceededError(
                error_count=self._consecutive_errors,
                threshold=self.max_errors,
                last_error=self._last_error,
                partial_answer=self._best_partial_answer,
                message=(
                    "Error threshold exceeded: "
                    f"{self._consecutive_errors} consecutive errors "
                    f"(limit: {self.max_errors})"
                ),
            )

        current_usage = self._usage_ledger.summary()

        # Check budget
        if self.max_budget is not None:
            current_cost = current_usage.total_cost or 0.0
            if current_cost > self.max_budget:
                self.verbose.print_budget_exceeded(current_cost, self.max_budget)
                raise BudgetExceededError(
                    spent=current_cost,
                    budget=self.max_budget,
                    message=(
                        f"Budget exceeded after iteration {iteration_num + 1}: "
                        f"spent ${current_cost:.6f} of ${self.max_budget:.6f} budget"
                    ),
                )

        # Check token limit
        if self.max_tokens is not None:
            total_tokens = current_usage.total_input_tokens + current_usage.total_output_tokens
            if total_tokens > self.max_tokens:
                self.verbose.print_limit_exceeded(
                    "tokens",
                    f"{total_tokens:,} of {self.max_tokens:,} tokens",
                )
                raise TokenLimitExceededError(
                    tokens_used=total_tokens,
                    token_limit=self.max_tokens,
                    partial_answer=self._best_partial_answer,
                    message=(
                        f"Token limit exceeded after iteration {iteration_num + 1}: "
                        f"{total_tokens:,} of {self.max_tokens:,} tokens"
                    ),
                )

    def _get_compaction_status(self, message_history: list[dict[str, Any]]) -> tuple[int, int, int]:
        """Return (current_tokens, threshold_tokens, max_tokens) for compaction."""
        model_name = (
            self.backend_kwargs.get("model_name", "unknown") if self.backend_kwargs else "unknown"
        )
        max_tokens = get_context_limit(model_name)
        current_tokens = count_tokens(message_history, model_name)
        threshold_tokens = int(self.compaction_threshold_pct * max_tokens)
        return current_tokens, threshold_tokens, max_tokens

    def _compact_history(
        self,
        lm_handler: LMHandler,
        environment: SupportsCompaction,
        message_history: list[dict[str, Any]],
        compaction_count: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Summarize current trajectory, append summary to REPL history, and return
        a short message_history with the summary as the new starting point.
        """
        summary_prompt = message_history + [
            {
                "role": "user",
                "content": (
                    "Summarize your progress so far. Include:\n"
                    "1. Which steps/sub-tasks you have completed and which remain.\n"
                    "2. Any concrete intermediate results (numbers, values, variable names) "
                    "you computed — preserve these exactly.\n"
                    "3. What your next action should be.\n"
                    "Be concise (1–3 paragraphs) but preserve all key results and your "
                    "current position in the task."
                ),
            }
        ]
        summary = lm_handler.completion(summary_prompt)
        environment.append_compaction_entry({"type": "summary", "content": summary})
        # Keep system + initial assistant (metadata), then summary + continue
        new_history = message_history[:2] + [
            {"role": "assistant", "content": summary},
            {
                "role": "user",
                "content": (
                    f"Your conversation has been compacted {compaction_count} time(s). "
                    "Continue from the above summary. Do NOT repeat work you have already "
                    "completed. Use SHOW_VARS() to check which REPL variables exist, "
                    "and check `history` for full context. "
                    "Your next action:"
                ),
            },
        ]
        return new_history

    def _completion_turn(
        self,
        prompt: str | dict[str, Any],
        lm_handler: LMHandler,
        environment: BaseEnv,
    ) -> RLMIteration:
        """
        Perform a single iteration of the RLM, including prompting the model
        and code execution + tool execution.
        """
        iter_start = time.perf_counter()
        response = lm_handler.completion(prompt)
        code_block_strs = find_code_blocks(response)
        code_blocks = []

        for code_block_str in code_block_strs:
            code_result: REPLResult = environment.execute_code(code_block_str)
            code_blocks.append(CodeBlock(code=code_block_str, result=code_result))

        iteration_time = time.perf_counter() - iter_start
        return RLMIteration(
            prompt=prompt,
            response=response,
            code_blocks=code_blocks,
            iteration_time=iteration_time,
        )

    def _default_answer(self, message_history: list[dict[str, Any]], lm_handler: LMHandler) -> str:
        """
        Default behavior if the RLM runs out of iterations and does not find a final answer.
        It will take the message history, and try to generate a final answer from it.
        """
        current_prompt = message_history + [
            {
                "role": "assistant",
                "content": "Please provide a final answer to the user's question based on the information provided.",
            }
        ]
        response = lm_handler.completion(current_prompt)

        if self.logger:
            self.logger.log(
                RLMIteration(
                    prompt=current_prompt,
                    response=response,
                    final_answer=response,
                    code_blocks=[],
                )
            )

        return response

    def _fallback_answer(self, message: str | dict[str, Any]) -> str:
        """
        Fallback behavior if the RLM is actually at max depth, and should be treated as an LM.
        """
        client: BaseLM = get_client(self.backend, self.backend_kwargs)
        response = client.completion(message)
        return response

    def subcall(self, prompt: str, model: str | None = None) -> RLMChatCompletion:
        """Execute one recursive child through the canonical child module."""
        request = ChildRequest(task=prompt, model=model)
        return self._child_execution.run(request).unwrap_completion()

    def _settle_child_outcome(self, outcome: ChildOutcome) -> None:
        if not isinstance(outcome.usage, UsageSummary):
            raise TypeError("integrated child execution must report a UsageSummary")
        self._usage_ledger.settle(outcome.usage)

    @staticmethod
    def _completed_child(completion: RLMChatCompletion) -> _PreparedChild:
        return _PreparedChild(
            execute_child=lambda _cancellation: completion,
            usage_source=lambda: completion.usage_summary,
            teardown_child=lambda: None,
        )

    def _prepare_child(
        self,
        request: ChildRequest,
        _cancellation: Cancellation,
    ) -> _PreparedChild:
        """Resolve routing and open one fresh runtime for child execution."""
        prompt = request.prompt
        model = request.model
        next_depth = self.depth + 1
        if model is not None:
            child_backend_kwargs = (self.backend_kwargs or {}).copy()
            child_backend_kwargs["model_name"] = model
        else:
            child_backend_kwargs = self.backend_kwargs
        resolved_model = model or (child_backend_kwargs or {}).get("model_name", "unknown")

        try:
            child_budget = self._usage_ledger.remaining_budget()
        except BudgetExceededError as error:
            message = f"Budget exhausted (spent ${error.spent:.6f} of ${error.budget:.6f})"
            return self._completed_child(
                RLMChatCompletion(
                    root_model=resolved_model,
                    prompt=prompt,
                    response=f"Error: {message}",
                    usage_summary=UsageSummary.empty(),
                    execution_time=0.0,
                    error=message,
                )
            )

        if next_depth >= self.max_depth:
            if self.other_backends and self.other_backend_kwargs:
                client = get_client(self.other_backends[0], self.other_backend_kwargs[0])
            else:
                client = get_client(self.backend, child_backend_kwargs or {})
            root_model = model or client.model_name
            latest_usage: UsageSummary | None = None

            def execute_leaf(_cancel: Cancellation) -> RLMChatCompletion:
                nonlocal latest_usage
                started = time.perf_counter()
                try:
                    response = client.completion(prompt)
                except Exception as error:
                    message = f"LM query failed at max depth - {error}"
                    latest_usage = client.get_usage_summary()
                    return RLMChatCompletion(
                        root_model=root_model,
                        prompt=prompt,
                        response=f"Error: {message}",
                        usage_summary=latest_usage,
                        execution_time=time.perf_counter() - started,
                        error=message,
                    )

                model_usage = client.get_last_usage()
                if not isinstance(model_usage, ModelUsageSummary):
                    raise TypeError("LM client get_last_usage must return ModelUsageSummary")
                latest_usage = UsageSummary(model_usage_summaries={root_model: model_usage})
                return RLMChatCompletion(
                    root_model=root_model,
                    prompt=prompt,
                    response=response,
                    usage_summary=latest_usage,
                    execution_time=time.perf_counter() - started,
                )

            return _PreparedChild(
                execute_child=execute_leaf,
                usage_source=lambda: (
                    latest_usage if latest_usage is not None else client.get_usage_summary()
                ),
                teardown_child=lambda: None,
            )

        remaining_timeout = None
        if self.max_timeout is not None and self._completion_start_time is not None:
            elapsed = time.perf_counter() - self._completion_start_time
            remaining_timeout = self.max_timeout - elapsed
            if remaining_timeout <= 0:
                message = f"Timeout exhausted ({elapsed:.1f}s of {self.max_timeout:.1f}s)"
                return self._completed_child(
                    RLMChatCompletion(
                        root_model=resolved_model,
                        prompt=prompt,
                        response=f"Error: {message}",
                        usage_summary=UsageSummary.empty(),
                        execution_time=0.0,
                        error=message,
                    )
                )

        prompt_preview = prompt[:80] if len(prompt) > 80 else prompt
        if self.on_subcall_start:
            try:
                self.on_subcall_start(next_depth, str(resolved_model), prompt_preview)
            except Exception:
                pass

        subcall_start = time.perf_counter()
        error_msg: str | None = None
        latest_completion: RLMChatCompletion | None = None
        child = RLM(
            backend=self.backend,
            backend_kwargs=child_backend_kwargs,
            environment=self.environment_type,
            environment_kwargs=self.environment_kwargs,
            depth=next_depth,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            max_budget=child_budget,
            max_timeout=remaining_timeout,
            max_tokens=self.max_tokens,
            max_errors=self.max_errors,
            custom_system_prompt=self.system_prompt,
            other_backends=self.other_backends,
            other_backend_kwargs=self.other_backend_kwargs,
            logger=RLMLogger() if self.logger else None,
            verbose=False,
            custom_tools=self.custom_sub_tools,
            custom_sub_tools=self.custom_sub_tools,
            max_concurrent_subcalls=self.max_concurrent_subcalls,
            on_subcall_start=self.on_subcall_start,
            on_subcall_complete=self.on_subcall_complete,
            on_iteration_start=self.on_iteration_start,
            on_iteration_complete=self.on_iteration_complete,
        )

        def execute_recursive(_cancel: Cancellation) -> RLMChatCompletion:
            nonlocal error_msg, latest_completion
            try:
                latest_completion = child.completion(prompt, root_prompt=None)
                return latest_completion
            except BudgetExceededError as error:
                error_msg = f"Child RLM budget exceeded - {error}"
            except Exception as error:
                error_msg = f"Child RLM completion failed - {error}"
            assert error_msg is not None
            latest_completion = RLMChatCompletion(
                root_model=resolved_model,
                prompt=prompt,
                response=f"Error: {error_msg}",
                usage_summary=child._usage_ledger.summary(),
                execution_time=time.perf_counter() - subcall_start,
                error=error_msg,
            )
            return latest_completion

        def recursive_usage() -> UsageSummary:
            if latest_completion is not None:
                return latest_completion.usage_summary
            return child._usage_ledger.summary()

        def teardown_recursive() -> None:
            try:
                child.close()
            finally:
                if self.on_subcall_complete:
                    try:
                        duration = time.perf_counter() - subcall_start
                        self.on_subcall_complete(
                            next_depth,
                            str(resolved_model),
                            duration,
                            error_msg,
                        )
                    except Exception:
                        pass

        return _PreparedChild(
            execute_child=execute_recursive,
            usage_source=recursive_usage,
            teardown_child=teardown_recursive,
        )

    def _validate_environment_capabilities(self) -> None:
        capabilities = self._environment_capabilities
        requested = (
            (self.persistent, capabilities.persistence, "persistent=True"),
            (self.compaction, capabilities.compaction, "compaction=True"),
            (self.custom_tools is not None, capabilities.custom_tools, "custom_tools"),
        )
        for enabled, supported, option in requested:
            if enabled and not supported:
                raise ValueError(
                    f"{option} is not supported for environment type {self.environment_type!r}"
                )

    def _validate_environment_instance(self, environment: BaseEnv) -> None:
        required = (
            (self.persistent, SupportsPersistence, "persistence"),
            (self.compaction, SupportsCompaction, "compaction"),
            (self.custom_tools is not None, SupportsCustomTools, "custom tools"),
        )
        for enabled, protocol, capability in required:
            if enabled and not isinstance(environment, protocol):
                raise RuntimeError(
                    f"Environment {type(environment).__name__} declares {capability} support "
                    "but does not implement its interface"
                )

    def close(self) -> None:
        """Clean up persistent environment and child-execution resources."""
        errors: list[BaseException] = []
        if self._persistent_env is not None:
            try:
                self._persistent_env.cleanup()
            except BaseException as error:
                errors.append(error)
            else:
                self._persistent_env = None
        try:
            self._child_execution.close()
        except BaseException as error:
            errors.append(error)

        if len(errors) == 1:
            error = errors[0]
            raise error.with_traceback(error.__traceback__)
        if errors:
            raise BaseExceptionGroup("RLM cleanup failed", errors)

    def __enter__(self) -> "RLM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
