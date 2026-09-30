"""Evaluate LLMs on generated benchmark instances.

For each validated instance, sends the question plus explicitly configured
table context to each requested model and records the response in data/results/.
The default is question-only; prompt flags and tools opt into table access.

Provider is auto-detected from the model name via the MODEL_FAMILIES
registry — a single run can mix Anthropic, OpenAI, DeepSeek, and Bedrock
models. API clients are instantiated lazily; only the providers
actually needed get initialized.

Supports tool-use evaluation via --tools with any combination of:
  - load_data:             return the instance's CSV as text
  - run_python:            run code in a persistent subprocess sandbox

Filter flags (--dataset, --injector) restrict which instances are
evaluated; --models picks which LLMs to run (required).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from io_utils import (
    JsonResultLifecycle,
    load_json,
    validate_atomic_output_target,
)
from shared.cli import configure_cli_streams
from shared.manifests import resolve_manifest_paths
from shared.path_utils import (
    ensure_portable_child_namespace,
    normalize_path_text,
    safe_path_component,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INSTANCES_DIR = Path("data/instances")
DEFAULT_OUTPUT = Path("data/results/eval_results.json")
SUPPORTED_TOOLS = ("load_data", "run_python")
LOAD_DATA_TOOL, RUN_PYTHON_TOOL = SUPPORTED_TOOLS


def _nonnegative_finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return value

# These access restrictions are prompt-level guidance.
# Dataset context may be supplied inline in the user message or exposed through
# declared tools; run_python provides the instance table as a preloaded `df`.

SYSTEM_PROMPT = (
    "You are a data analyst. Answer the question about the dataset concisely. "
    "Return ONLY the answer in the exact format requested with no extra text. "
    "You may analyze only the preloaded pandas DataFrame `df` and outputs returned by the declared tools. "
    "Do not inspect or access the filesystem, directories, environment variables, network, subprocesses, "
    "source code, manifests, metadata artifacts, or answer files. "
    "Do not use open(), pathlib, os, glob, subprocess, socket, or shell commands."
)
DEEPSEEK_JSON_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + '\nReturn a valid JSON object with exactly this shape: {"answer": "your answer"}.'
)

# Shared JSON schema used by all providers for structured output.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

# Per-(QA, model) hard defaults for tool-enabled evaluation. The model-call
# budget includes the optional final, tool-free salvage request, so ordinary
# tool rounds reserve one call for that final answer. Token usage is reported
# only after a provider response, so the token limit is enforced at turn
# boundaries (a single response may overshoot it once).
MAX_TOOL_ITER = 25
MAX_TOOL_CALLS = 64
MAX_TOOL_TOTAL_TOKENS = 500_000
MAX_TOOL_RESULT_BYTES = 2_000_000
MAX_TOOL_WALL_TIME_S = 600.0
MAX_TOOL_PROVIDER_CALL_S = 300.0
MAX_PROVIDER_ERROR_MESSAGE_CHARS = 1_000

# One model-call slot inside MAX_TOOL_ITER is reserved for a tool-free salvage
# request. It is used when the iteration limit is reached, the tool-call limit
# prevents further execution, or a provider emits a no-text final turn. Token,
# deadline, and tool-output limits can stop the loop without salvage. A
# successful non-empty final/salvage answer is retained even if that response
# brings reported token/time usage to its limit. Each provider disables further
# tools using its compatible request shape. Most final answers are short, so the
# default cap is small; DeepSeek V4 keeps thinking enabled and therefore needs a
# larger provider-specific allowance.
SALVAGE_MAX_TOKENS = 1024
DEEPSEEK_SALVAGE_MAX_TOKENS = 16_000
_FINAL_ANSWER_NUDGE = (
    "You have gathered enough information. Based on your analysis above, state your "
    "final answer now — concise and direct, and do not request any more tools."
)


class _ToolLoopBudget:
    """Track the aggregate cost of one tool-enabled (QA, model) session."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.deadline_at = self.started_at + MAX_TOOL_WALL_TIME_S
        self.model_calls = 0
        self.tool_calls = 0
        self.tool_result_bytes = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason: str | None = None
        self.provider_error: dict[str, str] | None = None
        self.salvage_error: dict[str, str] | None = None
        self.usage_incomplete = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def remaining_tokens(self) -> int:
        return max(0, MAX_TOOL_TOTAL_TOKENS - self.total_tokens)

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_at - time.monotonic())

    def hard_stop_reason(self) -> str | None:
        if self.total_tokens >= MAX_TOOL_TOTAL_TOKENS:
            return "tokens"
        if self.remaining_seconds() <= 0:
            return "deadline"
        return None

    def begin_model_call(self, *, reserve_salvage: bool) -> str | None:
        """Consume one provider call, optionally reserving the last for salvage."""
        reason = self.hard_stop_reason()
        if reason is not None:
            return reason
        limit = MAX_TOOL_ITER - (1 if reserve_salvage else 0)
        if self.model_calls >= max(0, limit):
            return "iterations"
        self.model_calls += 1
        return None

    def begin_tool_call(self) -> str | None:
        reason = self.hard_stop_reason()
        if reason is not None:
            return reason
        if self.tool_result_bytes >= MAX_TOOL_RESULT_BYTES:
            return "tool_output"
        if self.tool_calls >= MAX_TOOL_CALLS:
            return "tool_calls"
        self.tool_calls += 1
        return None

    def add_usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.input_tokens += max(0, int(input_tokens or 0))
        self.output_tokens += max(0, int(output_tokens or 0))

    def cap_tool_result(self, result: str) -> tuple[str, bool]:
        """Fit one result into the aggregate tool-result byte budget."""
        encoded = result.encode("utf-8", errors="replace")
        remaining = max(0, MAX_TOOL_RESULT_BYTES - self.tool_result_bytes)
        if len(encoded) <= remaining:
            self.tool_result_bytes += len(encoded)
            return result, False

        marker = (
            f"\n[tool result truncated: aggregate limit "
            f"{MAX_TOOL_RESULT_BYTES} bytes]"
        ).encode("utf-8")
        if remaining <= 0:
            captured = b""
        elif remaining <= len(marker):
            captured = marker[:remaining]
        else:
            captured = encoded[: remaining - len(marker)] + marker
        self.tool_result_bytes += len(captured)
        return captured.decode("utf-8", errors="ignore"), True

    def mark_stopped(self, reason: str) -> None:
        self.stop_reason = reason

    def record_request_exception(
        self,
        stage: str,
        exc: BaseException,
    ) -> dict[str, str]:
        message = " ".join(str(exc).splitlines()).strip() or "(no message)"
        if len(message) > MAX_PROVIDER_ERROR_MESSAGE_CHARS:
            message = message[: MAX_PROVIDER_ERROR_MESSAGE_CHARS - 3] + "..."
        detail = {"type": type(exc).__name__, "message": message}
        if stage == "provider":
            self.provider_error = detail
        elif stage == "salvage":
            self.salvage_error = detail
        else:
            raise ValueError(f"unknown request error stage: {stage}")
        # A failed request can consume provider tokens without returning usage.
        self.usage_incomplete = True
        return detail

    def record_empty_salvage(self) -> dict[str, str]:
        detail = {
            "type": "empty_response",
            "message": "provider returned no answer text",
        }
        self.salvage_error = detail
        return detail


def _tool_budget_error(reason: str) -> str:
    labels = {
        "iterations": f"model-call limit {MAX_TOOL_ITER}",
        "tool_calls": f"tool-call limit {MAX_TOOL_CALLS}",
        "tool_output": f"tool-result limit {MAX_TOOL_RESULT_BYTES} bytes",
        "tokens": f"token limit {MAX_TOOL_TOTAL_TOKENS}",
        "deadline": f"wall-time limit {MAX_TOOL_WALL_TIME_S:g}s",
    }
    return f"ERROR: tool loop budget exhausted ({labels.get(reason, reason)})"


def _tool_budget_metrics(
    budget: _ToolLoopBudget,
    **extra: object,
) -> dict:
    # ``tool_iterations`` is a historical field name: it counts provider/model
    # calls, including a salvage call, rather than executed tool calls.
    metrics: dict = {
        "input_tokens": budget.input_tokens,
        "output_tokens": budget.output_tokens,
        "ttft_s": None,
        "total_latency_s": max(0.0, time.monotonic() - budget.started_at),
        "tool_iterations": budget.model_calls,
        "tool_calls": budget.tool_calls,
        "tool_result_bytes": budget.tool_result_bytes,
    }
    if budget.stop_reason is not None:
        metrics["budget_stop_reason"] = budget.stop_reason
    if budget.provider_error is not None:
        metrics["provider_error"] = budget.provider_error
    if budget.salvage_error is not None:
        metrics["salvage_error"] = budget.salvage_error
    if budget.usage_incomplete:
        metrics["usage_incomplete"] = True
    metrics.update(extra)
    return metrics


def _budgeted_api_client(client, budget: _ToolLoopBudget):
    """When the SDK supports it, disable retries and clamp one request timeout."""
    with_options = getattr(client, "with_options", None)
    if not callable(with_options):
        return client
    timeout = max(
        0.001,
        min(MAX_TOOL_PROVIDER_CALL_S, budget.remaining_seconds()),
    )
    return with_options(timeout=timeout, max_retries=0)


def _request_error_answer(kind: str, detail: dict[str, str]) -> str:
    return f"ERROR: {kind}: {detail['type']}: {detail['message']}"


class _BedrockDeadlineGuardError(RuntimeError):
    """Raised when a Bedrock client cannot honor the remaining loop deadline."""


@contextmanager
def _budgeted_bedrock_client(
    client,
    budget: _ToolLoopBudget,
    client_factory: Callable[[float], object] | None,
):
    """Yield a Bedrock client whose configured timeouts fit the current window.

    The base client is reused for a full provider-call window. A shorter window
    requires a temporary client from ``client_factory``; the client is closed
    afterward. If no factory exists, or creating it consumes the deadline, fail
    closed rather than start a request that cannot honor the remaining budget.
    """
    timeout = max(
        0.001,
        min(MAX_TOOL_PROVIDER_CALL_S, budget.remaining_seconds()),
    )
    if timeout >= MAX_TOOL_PROVIDER_CALL_S:
        yield client
        return
    if client_factory is None:
        # boto3 has no public per-operation timeout override. Refuse to start a
        # fixed-timeout request that could substantially overrun the deadline.
        raise _BedrockDeadlineGuardError(
            "Bedrock call cannot be clamped to the remaining tool-loop deadline"
        )

    call_client = client_factory(timeout)
    try:
        if budget.hard_stop_reason() == "deadline":
            raise _BedrockDeadlineGuardError(
                "Bedrock client creation exhausted the tool-loop deadline"
            )
        yield call_client
    finally:
        close = getattr(call_client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # Closing a temporary connection pool must not replace a
                # successful response or obscure the original request error.
                pass


_ToolQueryResult = tuple[str, Optional[str], list[dict], dict]
_SalvageCallResult = tuple[str, int, int, dict[str, object]]


class _ToolLoopState:
    """Cross-provider state transitions shared by the tool-loop adapters."""

    def __init__(
        self,
        thinking_parts: list[str],
        *,
        metrics_extra: Callable[[], dict[str, object]] | None = None,
    ) -> None:
        self.budget = _ToolLoopBudget()
        self.thinking_parts = thinking_parts
        self.tool_use_log: list[dict] = []
        self._metrics_extra = metrics_extra

    def finish(
        self,
        answer: str,
        **metrics_override: object,
    ) -> _ToolQueryResult:
        extra = dict(self._metrics_extra() if self._metrics_extra else {})
        extra.update(metrics_override)
        thinking = "\n\n".join(self.thinking_parts) or None
        return (
            answer,
            thinking,
            self.tool_use_log,
            _tool_budget_metrics(self.budget, **extra),
        )

    def stop(self, reason: str) -> _ToolQueryResult:
        self.budget.mark_stopped(reason)
        return self.finish(_tool_budget_error(reason))

    def provider_failure(self, exc: Exception) -> _ToolQueryResult:
        if isinstance(exc, _BedrockDeadlineGuardError):
            return self.stop("deadline")
        detail = self.budget.record_request_exception("provider", exc)
        reason = self.budget.hard_stop_reason()
        if reason is not None:
            return self.stop(reason)
        return self.finish(_request_error_answer("provider_error", detail))

    def begin_round(
        self,
        salvage: Callable[[str | None], _ToolQueryResult],
    ) -> _ToolQueryResult | None:
        reason = self.budget.begin_model_call(reserve_salvage=True)
        if reason == "iterations":
            return salvage(reason)
        if reason is not None:
            return self.stop(reason)
        return None

    def salvage(
        self,
        request: Callable[[], _SalvageCallResult],
        trigger_reason: str | None = None,
    ) -> _ToolQueryResult:
        """Run the reserved no-tools call and apply terminal-state priorities.

        A non-empty answer wins even if its reported usage reaches a hard limit.
        For an empty response or exception, the priority is hard-limit reason,
        then the reason that triggered salvage, then a salvage diagnostic. A
        successful triggered salvage retains the answer and records the trigger
        as ``budget_stop_reason``.
        """
        reason = self.budget.begin_model_call(reserve_salvage=False)
        if reason is not None:
            return self.stop(reason)
        try:
            answer, input_tokens, output_tokens, finish_metrics = request()
        except Exception as exc:
            if isinstance(exc, _BedrockDeadlineGuardError):
                return self.stop("deadline")
            detail = self.budget.record_request_exception("salvage", exc)
            hard_reason = self.budget.hard_stop_reason()
            if hard_reason is not None:
                return self.stop(hard_reason)
            if trigger_reason is not None:
                return self.stop(trigger_reason)
            return self.finish(_request_error_answer("salvage_error", detail))

        self.budget.add_usage(input_tokens, output_tokens)
        hard_reason = self.budget.hard_stop_reason()
        if not answer:
            detail = self.budget.record_empty_salvage()
            if hard_reason is not None:
                return self.stop(hard_reason)
            if trigger_reason is not None:
                return self.stop(trigger_reason)
            return self.finish(_request_error_answer("salvage_error", detail))
        if trigger_reason is not None:
            self.budget.mark_stopped(trigger_reason)
        return self.finish(answer, **finish_metrics)


# ---------------------------------------------------------------------------
# Model-family registry
# ---------------------------------------------------------------------------
# Single source of truth for prefix-based model routing. Order matters:
# first matching prefix wins, so longer / more specific prefixes go first
# (e.g. "gpt-5" before "gpt-" so gpt-5 gets max_completion_tokens).

MODEL_FAMILIES: list[tuple[str, dict]] = [
    # Anthropic — adaptive support is version-specific. Fable 5 is always in
    # adaptive mode, so its request omits the optional `thinking` switch while
    # still following the thinking-compatible tool-choice rules.
    ("claude-fable-5",    {"provider": "anthropic", "tokens_kwarg": "max_tokens", "thinking_mode": "always_on"}),
    ("claude-opus-4-8",   {"provider": "anthropic", "tokens_kwarg": "max_tokens", "thinking_mode": "adaptive"}),
    ("claude-opus-4-7",   {"provider": "anthropic", "tokens_kwarg": "max_tokens", "thinking_mode": "adaptive"}),
    ("claude-opus-4-6",   {"provider": "anthropic", "tokens_kwarg": "max_tokens", "thinking_mode": "adaptive"}),
    ("claude-sonnet-5",   {"provider": "anthropic", "tokens_kwarg": "max_tokens", "thinking_mode": "adaptive"}),
    ("claude-sonnet-4-6", {"provider": "anthropic", "tokens_kwarg": "max_tokens", "thinking_mode": "adaptive"}),
    # Older/unknown Claude versions are still routed, but adaptive thinking is
    # not enabled unless the exact compatible prefix appears above.
    ("claude-haiku",  {"provider": "anthropic", "tokens_kwarg": "max_tokens"}),
    ("claude-opus",   {"provider": "anthropic", "tokens_kwarg": "max_tokens"}),
    ("claude-sonnet", {"provider": "anthropic", "tokens_kwarg": "max_tokens"}),
    ("claude",        {"provider": "anthropic", "tokens_kwarg": "max_tokens"}),
    # OpenAI
    ("gpt-5",         {"provider": "openai",    "tokens_kwarg": "max_completion_tokens"}),
    ("gpt-",          {"provider": "openai",    "tokens_kwarg": "max_tokens"}),
    ("o1",            {"provider": "openai",    "tokens_kwarg": "max_completion_tokens"}),
    ("o3",            {"provider": "openai",    "tokens_kwarg": "max_completion_tokens"}),
    ("o4",            {"provider": "openai",    "tokens_kwarg": "max_completion_tokens"}),
    # DeepSeek (OpenAI-compatible API). Thinking with tools is version-specific;
    # unknown/older versions keep the standard Chat Completions request shape.
    ("deepseek-v4-",   {"provider": "deepseek", "tokens_kwarg": "max_tokens", "supports_thinking_tools": True}),
    ("deepseek-",      {"provider": "deepseek", "tokens_kwarg": "max_tokens"}),
    # Amazon Bedrock DeepSeek (Converse API). V3.2 supports on-demand
    # single-region invocation (R1 requires a cross-region inference profile).
    # Use `bedrock:<model-id>` to pass a Bedrock model ID or inference
    # profile ID directly.
    ("bedrock-deepseek-v3.2", {"provider": "bedrock", "bedrock_model_id": "deepseek.v3.2", "max_tokens": 8_192}),
    ("bedrock:",              {"provider": "bedrock", "max_tokens": 8_192}),
]


def _family(model: str) -> dict:
    """Return per-family metadata for the given model name."""
    for prefix, meta in MODEL_FAMILIES:
        if model.startswith(prefix):
            return meta
    prefixes = ", ".join(p for p, _ in MODEL_FAMILIES)
    raise ValueError(
        f"Cannot infer model family for '{model}'. Known prefixes: {prefixes}."
    )


def _openai_tokens_kwarg(model: str, n: int) -> dict:
    """Return the token-limit kwarg for an OpenAI-compatible model."""
    return {_family(model)["tokens_kwarg"]: n}


def detect_provider(model: str) -> str:
    """Infer the provider from the model name."""
    return _family(model)["provider"]


def supports_thinking(model: str) -> bool:
    """True when this request uses Anthropic adaptive/always-on thinking."""
    return bool(_family(model).get("thinking_mode"))


def _anthropic_thinking_kwargs(model: str) -> dict:
    """Return the explicit thinking switch required by this Claude model.

    Fable 5 is always in adaptive mode and does not need a request switch.
    Sonnet/Opus adaptive models require an explicit switch on versions where
    thinking is otherwise off, so use the same explicit payload for all of them.
    """
    mode = _family(model).get("thinking_mode")
    return {"thinking": {"type": "adaptive"}} if mode == "adaptive" else {}


def _uses_responses_api(model: str) -> bool:
    """True for OpenAI reasoning models (gpt-5*, o1/o3/o4).

    Chat Completions rejects function tools while reasoning is on
    (400: "Function tools with reasoning_effort are not supported ... in
    /v1/chat/completions"), so these must go through the Responses API.
    Non-reasoning OpenAI models (gpt-4o etc.) and DeepSeek stay on Chat
    Completions — they use the max_tokens kwarg, reasoning models use
    max_completion_tokens, which is exactly the discriminator here.
    """
    fam = _family(model)
    return fam["provider"] == "openai" and fam["tokens_kwarg"] == "max_completion_tokens"


def _bedrock_model_id(model: str) -> str:
    """Return the Bedrock model ID / inference profile ID for a routed model name."""
    if model.startswith("bedrock:"):
        return model.removeprefix("bedrock:")
    override = os.environ.get("BEDROCK_DEEPSEEK_MODEL_ID") if model.startswith("bedrock-deepseek") else None
    return override or _family(model)["bedrock_model_id"]


def _bedrock_max_tokens(model: str) -> int:
    """Return the Bedrock Converse maxTokens setting for the model."""
    raw = os.environ.get("BEDROCK_MAX_TOKENS")
    if not raw:
        return int(_family(model).get("max_tokens", 8_192))
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"BEDROCK_MAX_TOKENS={raw!r} is not a valid integer"
        ) from None


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
# Static tool schemas passed to the provider APIs via the `tools=` kwarg.


def _build_tool_specs() -> dict:
    """Return the static schemas for the supported evaluation tools."""
    specs = {
        RUN_PYTHON_TOOL: {
            "description": (
                "Execute Python code only against the preloaded pandas DataFrame `df`. "
                "Filesystem, directory, environment, network, shell, and subprocess access are prohibited. "
                "Use print() to produce output."
            ),
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        LOAD_DATA_TOOL: {
            "description": "Load the full dataset as a CSV string. Call this to inspect column names, data types, and raw values.",
            "properties": {},
            "required": [],
        },
    }
    return {name: specs[name] for name in SUPPORTED_TOOLS}


_TOOL_SPECS = _build_tool_specs()


def _openai_tool(name: str) -> dict:
    spec = _TOOL_SPECS[name]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": {"type": "object", "properties": spec["properties"],
                           "required": spec["required"], "additionalProperties": False},
        },
    }


def _openai_responses_tool(name: str) -> dict:
    """Function-tool schema for the Responses API (flat, no `function` wrapper)."""
    spec = _TOOL_SPECS[name]
    return {
        "type": "function",
        "name": name,
        "description": spec["description"],
        "parameters": {"type": "object", "properties": spec["properties"],
                       "required": spec["required"], "additionalProperties": False},
    }


def _anthropic_tool(name: str) -> dict:
    spec = _TOOL_SPECS[name]
    return {
        "name": name,
        "description": spec["description"],
        "input_schema": {"type": "object", "properties": spec["properties"],
                         "required": spec["required"], "additionalProperties": False},
    }


def _bedrock_tool(name: str) -> dict:
    spec = _TOOL_SPECS[name]
    return {
        "toolSpec": {
            "name": name,
            "description": spec["description"],
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": spec["properties"],
                    "required": spec["required"],
                    "additionalProperties": False,
                }
            },
        }
    }


def _provider_tools(
    enabled_tools: set,
    build_tool: Callable[[str], dict],
) -> list[dict]:
    """Build enabled provider tools in the stable public tool order."""
    return [
        build_tool(name)
        for name in SUPPORTED_TOOLS
        if name in enabled_tools
    ]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
def _build_system_prompt_tools(enabled_tools: set) -> str:
    """Build the common system prompt for tool-enabled requests."""
    return SYSTEM_PROMPT + "\nRespond with text only when you have the final answer."


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------


def _print_request(system: str, user_content: str, tools: list[dict]) -> None:
    """Print the full initial request payload for debugging."""
    sep = "─" * 72
    print(f"\n{sep}")
    print("DEBUG: INITIAL REQUEST")
    print(sep)
    print("SYSTEM PROMPT:")
    print(system)
    print(sep)
    print("USER MESSAGE:")
    print(user_content)
    if tools:
        print(sep)
        print("TOOLS:")
        print(json.dumps(tools, indent=2, ensure_ascii=False))
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Query functions  (client, model, csv_text, question) -> (answer, thinking, metrics)
# ---------------------------------------------------------------------------


def _build_user_content(csv_text: str, question: str) -> str:
    """Assemble user message from dataset and question."""
    parts = []
    if csv_text:
        parts.append(f"## Dataset\n{csv_text}")
    parts.append(f"## Question\n{question}")
    return "\n\n".join(parts)


def _parse_answer_payload(raw: str) -> str:
    """Extract a string ``answer`` from provider JSON, else keep raw text.

    Providers can occasionally ignore the requested object schema and return
    another valid JSON value such as ``null`` or ``[]``. Those payloads, and
    objects whose ``answer`` is not a string, are retained as raw text rather
    than raising or leaking a non-string into the result schema.
    """
    fallback = raw.strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    answer = payload.get("answer")
    return answer if isinstance(answer, str) else fallback


def query_anthropic(
    client,  # anthropic.Anthropic
    model: str,
    csv_text: str,
    question: str,
) -> tuple[str, str | None, dict]:
    """Query an Anthropic model using tool-use for structured output."""
    user_content = _build_user_content(csv_text, question)
    use_thinking = supports_thinking(model)
    kwargs: dict = dict(
        model=model,
        max_tokens=16_000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        tools=[
            {
                "name": "submit_answer",
                "description": "Submit the answer to the question.",
                "input_schema": ANSWER_SCHEMA,
            }
        ],
        # Forced tool_choice is incompatible with extended thinking; use "auto" instead.
        tool_choice={"type": "auto"} if use_thinking else {"type": "tool", "name": "submit_answer"},
    )
    kwargs.update(_anthropic_thinking_kwargs(model))

    start = time.time()
    ttft: float | None = None
    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            if ttft is None and event.type == "content_block_delta":
                ttft = time.time() - start
        msg = stream.get_final_message()
    total_latency = time.time() - start

    metrics = {
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "ttft_s": ttft,
        "total_latency_s": total_latency,
    }

    thinking_parts: list[str] = []
    text_parts: list[str] = []
    answer = ""
    for block in msg.content:
        if block.type == "thinking":
            thinking_text = (block.thinking or "").strip()
            if thinking_text:
                thinking_parts.append(thinking_text)
        elif block.type == "tool_use" and block.name == "submit_answer":
            answer = block.input.get("answer", "")
        elif block.type == "text":
            text_parts.append(block.text)

    # `tool_choice=auto` is required when thinking is active, and auto allows a
    # valid direct-text answer. Prefer the structured tool result when present,
    # otherwise keep that text instead of silently recording an empty answer.
    if not answer:
        answer = " ".join(text_parts).strip()
    thinking = "\n\n".join(thinking_parts) or None
    return answer, thinking, metrics


def query_openai(
    client,  # openai.OpenAI
    model: str,
    csv_text: str,
    question: str,
) -> tuple[str, str | None, dict]:
    """Query an OpenAI model with JSON schema structured output."""
    user_content = _build_user_content(csv_text, question)

    start = time.time()
    ttft: float | None = None
    chunks: list[str] = []
    usage = None
    stream = client.chat.completions.create(
        model=model,
        **_openai_tokens_kwarg(model, 1024),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": ANSWER_SCHEMA,
            },
        },
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            ttft = time.time() - start
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
        if chunk.usage:
            usage = chunk.usage
    raw = "".join(chunks)
    total_latency = time.time() - start

    metrics = {
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "ttft_s": ttft,
        "total_latency_s": total_latency,
    }

    answer = _parse_answer_payload(raw)
    return answer, None, metrics


def _reasoning_content(message) -> str | None:
    """Extract reasoning content returned by OpenAI-compatible providers."""
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        return reasoning
    model_extra = getattr(message, "model_extra", None) or {}
    return model_extra.get("reasoning_content")


def _bedrock_message_text_and_reasoning(message: dict) -> tuple[str, str | None]:
    """Extract visible text and reasoning text from a Bedrock Converse message."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for block in message.get("content", []):
        if "text" in block:
            text_parts.append(block["text"])
        reasoning = block.get("reasoningContent")
        if not reasoning:
            continue
        reasoning_text = reasoning.get("reasoningText")
        if isinstance(reasoning_text, dict) and reasoning_text.get("text"):
            reasoning_parts.append(reasoning_text["text"])
        elif isinstance(reasoning_text, str):
            reasoning_parts.append(reasoning_text)
    return "".join(text_parts).strip(), ("\n\n".join(reasoning_parts) or None)


def _bedrock_metrics(response: dict, total_latency: float) -> dict:
    usage = response.get("usage") or {}
    metrics = response.get("metrics") or {}
    return {
        "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"),
        "ttft_s": None,
        "total_latency_s": total_latency,
        "bedrock_latency_ms": metrics.get("latencyMs"),
    }


def _step_thinking_text(
    visible_text: str | None,
    reasoning_text: str | None,
) -> str | None:
    """Return this turn's best provider-supplied explanation for a tool call.

    Prefer visible narration. If the provider emitted no narration on its tool
    turn, fall back to reasoning content/summary from that same turn. Never
    synthesize text or reuse reasoning from an earlier turn.
    """
    visible = (visible_text or "").strip()
    if visible:
        return visible
    reasoning = (reasoning_text or "").strip()
    return reasoning or None


def _tool_log_entry(tool: str, input_, step_thinking: str | None) -> dict:
    """Build a tool log with the best explanation emitted on the same turn.

    ``step_thinking`` prefers visible narration and otherwise contains the
    provider-exposed reasoning content/summary for that turn. It is never raw
    hidden chain-of-thought and may still be ``None`` if the provider returns
    neither form.
    """
    entry: dict = {"tool": tool}
    if input_ is not None:
        entry["input"] = input_
    entry["step_thinking"] = step_thinking or None
    return entry


def _parse_json_tool_input(raw_input: object) -> dict:
    """Parse provider JSON arguments, accepting only an object payload."""
    if not isinstance(raw_input, str):
        return {}
    try:
        parsed = json.loads(raw_input or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _use_structured_tool_input(raw_input: object) -> dict:
    """Accept an already-decoded provider input only when it is an object."""
    return raw_input if isinstance(raw_input, dict) else {}


def _run_eval_tool(
    tool_name: str,
    tool_input: dict | None,
    sandbox: "PythonSandbox | None",
    csv_path: Path,
    tool_use_log: list[dict],
    step_thinking: str | None = None,
    *,
    deadline: float | None = None,
) -> str:
    """Execute one eval tool call and append the existing tool-use log format."""
    if not isinstance(tool_input, dict):
        tool_input = {}

    if tool_name == LOAD_DATA_TOOL:
        tool_use_log.append(
            _tool_log_entry(LOAD_DATA_TOOL, None, step_thinking)
        )
        return csv_path.read_text(encoding="utf-8")
    if tool_name == RUN_PYTHON_TOOL:
        code = tool_input.get("code")
        tool_use_log.append(
            _tool_log_entry(RUN_PYTHON_TOOL, code, step_thinking)
        )
        if not isinstance(code, str):
            return "[run_python failed: code must be a string]"
        if sandbox is None:
            return "[run_python unavailable: sandbox is not initialized]"
        if deadline is None:
            return sandbox.run(code, csv_path)
        return sandbox.run(code, csv_path, deadline=deadline)

    tool_use_log.append(_tool_log_entry(tool_name, tool_input, step_thinking))
    return f"[unknown tool: {tool_name or '<empty>'}]"


def _run_budgeted_eval_tool(
    tool_name: str,
    tool_input: dict | None,
    sandbox: "PythonSandbox | None",
    csv_path: Path,
    tool_use_log: list[dict],
    step_thinking: str | None,
    budget: _ToolLoopBudget,
) -> str:
    """Run one tool under the session's absolute wall-time deadline."""
    return _run_eval_tool(
        tool_name,
        tool_input,
        sandbox,
        csv_path,
        tool_use_log,
        step_thinking,
        deadline=budget.deadline_at,
    )


def _execute_tool_batch(
    calls: list[tuple[str, str, object]],
    sandbox: "PythonSandbox | None",
    csv_path: Path,
    tool_use_log: list[dict],
    step_thinking: str | None,
    budget: _ToolLoopBudget,
    decode_input: Callable[[object], dict],
    emit_result: Callable[[str, str], None],
) -> tuple[str | None, str | None]:
    """Execute normalized tool calls under shared count/time/output policies.

    Returns ``(stop_reason, salvage_reason)``. A tool-call count limit yields
    synthetic results for the rest of the provider batch and requests a final
    no-tools salvage call. After each executed result is capped and emitted,
    token/deadline limits are checked before another call; an intervening hard
    limit takes precedence over salvage.
    """
    salvage_reason: str | None = None
    for call_id, tool_name, raw_input in calls:
        if salvage_reason is not None:
            result = _tool_budget_error(salvage_reason)
        else:
            reason = budget.begin_tool_call()
            if reason is not None:
                if reason != "tool_calls":
                    return reason, None
                salvage_reason = reason
                result = _tool_budget_error(reason)
            else:
                result = _run_budgeted_eval_tool(
                    tool_name,
                    decode_input(raw_input),
                    sandbox,
                    csv_path,
                    tool_use_log,
                    step_thinking,
                    budget,
                )

        result, output_limited = budget.cap_tool_result(result)
        emit_result(call_id, result)
        if output_limited:
            return "tool_output", salvage_reason
        reason = budget.hard_stop_reason()
        if reason is not None:
            return reason, salvage_reason

    return None, salvage_reason


def query_deepseek(
    client,  # openai.OpenAI configured for DeepSeek
    model: str,
    csv_text: str,
    question: str,
) -> tuple[str, str | None, dict]:
    """Query DeepSeek using its JSON Output mode."""
    user_content = _build_user_content(csv_text, question)

    start = time.time()
    request_kwargs: dict = dict(
        model=model,
        **_openai_tokens_kwarg(model, 16_000),
        messages=[
            {"role": "system", "content": DEEPSEEK_JSON_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )
    total_input = total_output = 0
    saw_usage = False
    raw = ""
    # DeepSeek documents an occasional empty content response in JSON mode.
    # Retry that exact transient once rather than recording a false empty answer.
    for _ in range(2):
        resp = client.chat.completions.create(**request_kwargs)
        usage = resp.usage
        if usage:
            saw_usage = True
            total_input += usage.prompt_tokens or 0
            total_output += usage.completion_tokens or 0
        msg = resp.choices[0].message
        raw = msg.content or ""
        if raw.strip():
            break
    total_latency = time.time() - start

    metrics = {
        "input_tokens": total_input if saw_usage else None,
        "output_tokens": total_output if saw_usage else None,
        "ttft_s": None,
        "total_latency_s": total_latency,
    }

    answer = _parse_answer_payload(raw)
    return answer, _reasoning_content(msg), metrics


def query_bedrock_deepseek(
    client,  # boto3 bedrock-runtime client
    model: str,
    csv_text: str,
    question: str,
) -> tuple[str, str | None, dict]:
    """Query a DeepSeek model (default deepseek.v3.2, or any bedrock:<model-id>) through Amazon Bedrock Converse."""
    user_content = _build_user_content(csv_text, question)

    start = time.time()
    resp = client.converse(
        modelId=_bedrock_model_id(model),
        system=[{"text": DEEPSEEK_JSON_SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": user_content}]}],
        inferenceConfig={"maxTokens": _bedrock_max_tokens(model)},
    )
    total_latency = time.time() - start

    raw, thinking = _bedrock_message_text_and_reasoning(resp["output"]["message"])
    metrics = _bedrock_metrics(resp, total_latency)
    answer = _parse_answer_payload(raw)
    return answer, thinking, metrics


def query_openai_style_tools(
    client,        # openai.OpenAI
    model: str,
    csv_text: str,
    question: str,
    sandbox: "PythonSandbox | None",
    csv_path: Path,
    enabled_tools: set = frozenset({RUN_PYTHON_TOOL}),
    *,
    thinking_mode: bool = False,
    max_tokens_per_turn: int = 2048,
) -> tuple[str, str | None, list[dict], dict]:
    """Tool-calling query via OpenAI API (Chat Completions).

    OpenAI reasoning models (gpt-5*, o-series) can't use function tools on Chat
    Completions while reasoning is on, so they are routed to the Responses API.
    gpt-4o and DeepSeek stay on the Chat Completions path below.
    """
    if _uses_responses_api(model):
        return query_openai_responses_tools(
            client, model, csv_text, question, sandbox, csv_path,
            enabled_tools=enabled_tools,
        )
    system = _build_system_prompt_tools(enabled_tools)
    tools_list = _provider_tools(enabled_tools, _openai_tool)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": _build_user_content(csv_text, question)},
    ]
    accumulated_reasoning: list[str] = []
    state = _ToolLoopState(accumulated_reasoning)
    budget = state.budget

    def _salvage(
        trigger_reason: str | None = None,
    ) -> tuple[str, str | None, list[dict], dict]:
        default_cap = (
            DEEPSEEK_SALVAGE_MAX_TOKENS
            if thinking_mode
            else SALVAGE_MAX_TOKENS
        )

        def _request() -> _SalvageCallResult:
            answer, salv_input, salv_output = _force_final_answer_openai_chat(
                _budgeted_api_client(client, budget),
                model,
                messages,
                thinking_mode=thinking_mode,
                max_tokens=max(1, min(default_cap, budget.remaining_tokens())),
            )
            return answer, salv_input, salv_output, {}

        return state.salvage(_request, trigger_reason)

    while True:
        terminal = state.begin_round(_salvage)
        if terminal is not None:
            return terminal
        request_kwargs: dict = dict(
            model=model,
            **_openai_tokens_kwarg(
                model,
                max(1, min(max_tokens_per_turn, budget.remaining_tokens())),
            ),
            messages=messages,
            tools=tools_list,
        )
        if thinking_mode:
            # DeepSeek V4 thinking mode rejects the tool_choice parameter. With
            # tools present, omitting it already means automatic tool choice.
            request_kwargs["reasoning_effort"] = "high"
            request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            request_kwargs["tool_choice"] = "auto"
        try:
            resp = _budgeted_api_client(client, budget).chat.completions.create(
                **request_kwargs
            )
        except Exception as exc:
            return state.provider_failure(exc)
        if resp.usage:
            budget.add_usage(
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
            )
        msg = resp.choices[0].message
        reasoning = _reasoning_content(msg)
        if reasoning:
            accumulated_reasoning.append(reasoning)
        if thinking_mode:
            # DeepSeek V4 requires both a non-null assistant content field and
            # the complete reasoning_content on every subsequent tool round.
            assistant_message = {
                "role": "assistant",
                "content": msg.content or "",
                "reasoning_content": reasoning or "",
            }
            if msg.tool_calls:
                assistant_message["tool_calls"] = [
                    tc.model_dump(exclude_none=True) for tc in msg.tool_calls
                ]
        else:
            assistant_message = msg.model_dump(exclude_unset=True)
        messages.append(assistant_message)

        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            if not answer:
                return _salvage()
            return state.finish(answer)

        reason = budget.hard_stop_reason()
        if reason is not None:
            return state.stop(reason)

        step_text = _step_thinking_text(msg.content, reasoning)
        calls = [
            (
                tc.id,
                tc.function.name,
                tc.function.arguments,
            )
            for tc in msg.tool_calls
        ]
        stop_reason, salvage_reason = _execute_tool_batch(
            calls,
            sandbox,
            csv_path,
            state.tool_use_log,
            step_text,
            budget,
            _parse_json_tool_input,
            lambda call_id, result: messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": result}
            ),
        )
        if stop_reason is not None:
            return state.stop(stop_reason)
        if salvage_reason is not None:
            return _salvage(salvage_reason)


# Reasoning effort for OpenAI reasoning models on the Responses API. Fixed here
# so all model tiers are compared at the same setting; reasoning tokens count
# against max_output_tokens, so keep the cap generous.
OPENAI_REASONING_EFFORT = "medium"
OPENAI_RESPONSES_MAX_OUTPUT = 16_000


def query_openai_responses_tools(
    client,        # openai.OpenAI
    model: str,
    csv_text: str,
    question: str,
    sandbox: "PythonSandbox | None",
    csv_path: Path,
    enabled_tools: set = frozenset({RUN_PYTHON_TOOL}),
) -> tuple[str, str | None, list[dict], dict]:
    """Tool-calling query via the OpenAI Responses API (/v1/responses).

    Required for reasoning models (gpt-5*, o-series): Chat Completions rejects
    function tools while reasoning is on. Reasoning stays enabled via
    reasoning.effort. Same tool set and dispatch as query_openai_style_tools.
    """
    system = _build_system_prompt_tools(enabled_tools)
    tools_list = _provider_tools(enabled_tools, _openai_responses_tool)
    # Responses API grows an `input` list: model output items (incl. reasoning,
    # which reasoning models require replayed) then function_call_output items.
    input_items: list = [
        {"role": "user", "content": _build_user_content(csv_text, question)},
    ]
    accumulated_reasoning: list[str] = []
    state = _ToolLoopState(accumulated_reasoning)
    budget = state.budget

    def _salvage(
        trigger_reason: str | None = None,
        history: list | None = None,
    ) -> tuple[str, str | None, list[dict], dict]:
        def _request() -> _SalvageCallResult:
            answer, salv_input, salv_output = _force_final_answer_openai_responses(
                _budgeted_api_client(client, budget),
                model,
                system,
                input_items if history is None else history,
                max_tokens=max(
                    1,
                    min(SALVAGE_MAX_TOKENS, budget.remaining_tokens()),
                ),
            )
            return answer, salv_input, salv_output, {}

        return state.salvage(_request, trigger_reason)

    while True:
        terminal = state.begin_round(_salvage)
        if terminal is not None:
            return terminal
        try:
            resp = _budgeted_api_client(client, budget).responses.create(
                model=model,
                instructions=system,
                input=input_items,
                tools=tools_list,
                tool_choice="auto",
                reasoning={"effort": OPENAI_REASONING_EFFORT, "summary": "auto"},
                max_output_tokens=max(
                    1,
                    min(OPENAI_RESPONSES_MAX_OUTPUT, budget.remaining_tokens()),
                ),
            )
        except Exception as exc:
            return state.provider_failure(exc)
        if resp.usage:
            budget.add_usage(
                resp.usage.input_tokens,
                resp.usage.output_tokens,
            )

        # Capture summaries both for this tool turn and for the top-level
        # aggregate. Raw hidden reasoning tokens are not exposed by the API.
        turn_reasoning: list[str] = []
        for item in resp.output:
            if getattr(item, "type", None) == "reasoning":
                for s in (getattr(item, "summary", None) or []):
                    text = getattr(s, "text", None)
                    if text:
                        turn_reasoning.append(text)
                        accumulated_reasoning.append(text)

        function_calls = [it for it in resp.output if getattr(it, "type", None) == "function_call"]
        if not function_calls:
            answer = (resp.output_text or "").strip()
            if not answer:
                return _salvage(history=[*input_items, *resp.output])
            return state.finish(answer)

        reason = budget.hard_stop_reason()
        if reason is not None:
            return state.stop(reason)

        # Replay the model's output items (reasoning + calls) before the results.
        input_items += resp.output
        step_text = _step_thinking_text(
            resp.output_text,
            "\n\n".join(turn_reasoning),
        )
        calls = [
            (
                fc.call_id,
                fc.name,
                fc.arguments,
            )
            for fc in function_calls
        ]

        def _emit_result(call_id: str, result: str) -> None:
            input_items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            })

        stop_reason, salvage_reason = _execute_tool_batch(
            calls,
            sandbox,
            csv_path,
            state.tool_use_log,
            step_text,
            budget,
            _parse_json_tool_input,
            _emit_result,
        )
        if stop_reason is not None:
            return state.stop(stop_reason)
        if salvage_reason is not None:
            return _salvage(salvage_reason)


def query_deepseek_tools(
    client,        # openai.OpenAI configured for DeepSeek
    model: str,
    csv_text: str,
    question: str,
    sandbox: "PythonSandbox | None",
    csv_path: Path,
    enabled_tools: set = frozenset({RUN_PYTHON_TOOL}),
) -> tuple[str, str | None, list[dict], dict]:
    """Tool-calling query via DeepSeek's OpenAI-compatible API."""
    return query_openai_style_tools(
        client,
        model,
        csv_text,
        question,
        sandbox,
        csv_path,
        enabled_tools=enabled_tools,
        thinking_mode=bool(_family(model).get("supports_thinking_tools", False)),
        max_tokens_per_turn=16_000,
    )


def query_bedrock_deepseek_tools(
    client,        # boto3 bedrock-runtime client
    model: str,
    csv_text: str,
    question: str,
    sandbox: "PythonSandbox | None",
    csv_path: Path,
    enabled_tools: set = frozenset({RUN_PYTHON_TOOL}),
    *,
    client_factory: Callable[[float], object] | None = None,
) -> tuple[str, str | None, list[dict], dict]:
    """Tool-calling query via Amazon Bedrock Converse.

    ``client_factory`` creates deadline-clamped temporary clients when the
    remaining session window is shorter than the base client's timeout.
    """
    system = _build_system_prompt_tools(enabled_tools)
    tool_config = {
        "tools": _provider_tools(enabled_tools, _bedrock_tool)
    }
    messages: list[dict] = [
        {"role": "user", "content": [{"text": _build_user_content(csv_text, question)}]},
    ]
    accumulated_reasoning: list[str] = []
    last_latency_ms: int | None = None
    state = _ToolLoopState(
        accumulated_reasoning,
        metrics_extra=lambda: {"bedrock_latency_ms": last_latency_ms},
    )
    budget = state.budget

    def _salvage(
        trigger_reason: str | None = None,
    ) -> tuple[str, str | None, list[dict], dict]:
        def _request() -> _SalvageCallResult:
            with _budgeted_bedrock_client(
                client,
                budget,
                client_factory,
            ) as call_client:
                answer, salv_input, salv_output, salvage_latency_ms = (
                    _force_final_answer_bedrock(
                        call_client,
                        model,
                        system,
                        messages,
                        max_tokens=max(
                            1,
                            min(SALVAGE_MAX_TOKENS, budget.remaining_tokens()),
                        ),
                    )
                )
            finish_metrics: dict[str, object] = {}
            if salvage_latency_ms is not None:
                finish_metrics["bedrock_latency_ms"] = salvage_latency_ms
            return answer, salv_input, salv_output, finish_metrics

        return state.salvage(_request, trigger_reason)

    while True:
        terminal = state.begin_round(_salvage)
        if terminal is not None:
            return terminal
        try:
            with _budgeted_bedrock_client(
                client,
                budget,
                client_factory,
            ) as call_client:
                resp = call_client.converse(
                    modelId=_bedrock_model_id(model),
                    system=[{"text": system}],
                    messages=messages,
                    toolConfig=tool_config,
                    inferenceConfig={
                        "maxTokens": max(
                            1,
                            min(
                                _bedrock_max_tokens(model),
                                budget.remaining_tokens(),
                            ),
                        )
                    },
                )
        except Exception as exc:
            return state.provider_failure(exc)
        usage = resp.get("usage") or {}
        budget.add_usage(
            usage.get("inputTokens"),
            usage.get("outputTokens"),
        )
        last_latency_ms = (resp.get("metrics") or {}).get("latencyMs")

        output_message = resp["output"]["message"]
        messages.append(output_message)
        text, reasoning = _bedrock_message_text_and_reasoning(output_message)
        if reasoning:
            accumulated_reasoning.append(reasoning)
        step_text = _step_thinking_text(text, reasoning)
        tool_requests = [
            block["toolUse"]
            for block in output_message.get("content", [])
            if block.get("toolUse")
        ]

        if resp.get("stopReason") != "tool_use" or not tool_requests:
            answer = text.strip()
            if not answer:
                return _salvage()
            return state.finish(answer)

        reason = budget.hard_stop_reason()
        if reason is not None:
            return state.stop(reason)

        tool_results: list[dict] = []
        calls = [
            (
                tool["toolUseId"],
                tool.get("name", ""),
                tool.get("input") or {},
            )
            for tool in tool_requests
        ]

        def _emit_result(call_id: str, result: str) -> None:
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": call_id,
                        "content": [{"text": result}],
                    }
                }
            )

        stop_reason, salvage_reason = _execute_tool_batch(
            calls,
            sandbox,
            csv_path,
            state.tool_use_log,
            step_text,
            budget,
            _use_structured_tool_input,
            _emit_result,
        )
        if stop_reason is not None:
            return state.stop(stop_reason)

        messages.append({"role": "user", "content": tool_results})
        if salvage_reason is not None:
            return _salvage(salvage_reason)


def _force_final_answer_openai_chat(
    client,
    model: str,
    messages: list[dict],
    *,
    thinking_mode: bool = False,
    max_tokens: int | None = None,
) -> tuple[str, int, int]:
    """Force a final answer through OpenAI-compatible Chat Completions."""
    convo: list[dict] = []
    for message in messages:
        cleaned = dict(message)
        if thinking_mode:
            # DeepSeek V4 defaults to thinking even when the switch is omitted.
            # Preserve tool-turn reasoning and normalize content instead of
            # creating an invalid replay during this final, tool-free request.
            if cleaned.get("role") == "assistant" and cleaned.get("content") is None:
                cleaned["content"] = ""
        else:
            cleaned.pop("reasoning_content", None)
            if (
                cleaned.get("role") == "assistant"
                and not cleaned.get("content")
                and not cleaned.get("tool_calls")
            ):
                cleaned["content"] = "(prior empty response omitted)"
        convo.append(cleaned)
    convo.append({"role": "user", "content": _FINAL_ANSWER_NUDGE})

    request_kwargs: dict = dict(
        model=model,
        **_openai_tokens_kwarg(
            model,
            max_tokens
            if max_tokens is not None
            else (
                DEEPSEEK_SALVAGE_MAX_TOKENS
                if thinking_mode
                else SALVAGE_MAX_TOKENS
            ),
        ),
        messages=convo,
    )
    if thinking_mode:
        request_kwargs["reasoning_effort"] = "high"
        request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    resp = client.chat.completions.create(**request_kwargs)

    answer = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    return (
        answer,
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _force_final_answer_openai_responses(
    client,
    model: str,
    instructions: str,
    input_items: list,
    *,
    max_tokens: int = SALVAGE_MAX_TOKENS,
) -> tuple[str, int, int]:
    """Force a final answer through the OpenAI Responses API."""
    salvage_input = [*input_items, {"role": "user", "content": _FINAL_ANSWER_NUDGE}]
    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=salvage_input,
        max_output_tokens=max_tokens,
    )

    usage = resp.usage
    return (
        (resp.output_text or "").strip(),
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def _force_final_answer_bedrock(
    client,
    model: str,
    system: str,
    messages: list[dict],
    *,
    max_tokens: int = SALVAGE_MAX_TOKENS,
) -> tuple[str, int, int, int | None]:
    """Force a final answer through Bedrock Converse without toolConfig."""
    convo = list(messages)
    nudge = {"text": _FINAL_ANSWER_NUDGE}
    if convo and convo[-1].get("role") == "user":
        last = dict(convo[-1])
        last["content"] = [*(last.get("content") or []), nudge]
        convo[-1] = last
    else:
        convo.append({"role": "user", "content": [nudge]})

    resp = client.converse(
        modelId=_bedrock_model_id(model),
        system=[{"text": system}],
        messages=convo,
        inferenceConfig={"maxTokens": max_tokens},
    )

    answer, _ = _bedrock_message_text_and_reasoning(resp["output"]["message"])
    usage = resp.get("usage") or {}
    return (
        answer,
        int(usage.get("inputTokens") or 0),
        int(usage.get("outputTokens") or 0),
        (resp.get("metrics") or {}).get("latencyMs"),
    )


def _force_final_answer_anthropic(
    client,
    model: str,
    system: str,
    messages: list[dict],
    tools: list[dict],
    *,
    max_tokens: int = SALVAGE_MAX_TOKENS,
) -> tuple[str, int, int]:
    """Request an Anthropic final answer without permitting another tool call.

    Thinking blocks are replayed unchanged because adaptive/always-on models
    require them during tool loops. ``tool_choice=none`` remains compatible
    with thinking while preventing additional calls.
    """
    convo = [dict(message) for message in messages]
    if convo and convo[-1]["role"] == "assistant":
        convo.append({"role": "user", "content": _FINAL_ANSWER_NUDGE})
    elif convo:
        last = dict(convo[-1])
        content = last["content"]
        if isinstance(content, list):
            content = [*content, {"type": "text", "text": _FINAL_ANSWER_NUDGE}]
        else:
            content = f"{content}\n\n{_FINAL_ANSWER_NUDGE}"
        last["content"] = content
        convo[-1] = last
    request_kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=convo,
        tools=tools,
        tool_choice={"type": "none"},
    )
    request_kwargs.update(_anthropic_thinking_kwargs(model))
    if supports_thinking(model):
        # Keep the small salvage budget focused on the short final answer.
        request_kwargs["output_config"] = {"effort": "low"}
    msg = client.messages.create(**request_kwargs)
    answer = " ".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
    return answer, int(msg.usage.input_tokens or 0), int(msg.usage.output_tokens or 0)


def query_anthropic_tools(
    client,        # anthropic.Anthropic
    model: str,
    csv_text: str,
    question: str,
    sandbox: "PythonSandbox | None",
    csv_path: Path,
    enabled_tools: set = frozenset({RUN_PYTHON_TOOL}),
) -> tuple[str, str | None, list[dict], dict]:
    """Tool-calling query via Anthropic API."""
    system = _build_system_prompt_tools(enabled_tools)
    tools_list = _provider_tools(enabled_tools, _anthropic_tool)
    kwargs: dict = dict(
        model=model, max_tokens=16_000, system=system,
        tools=tools_list,
        tool_choice={"type": "auto"},
    )
    kwargs.update(_anthropic_thinking_kwargs(model))

    messages: list[dict] = [
        {"role": "user", "content": _build_user_content(csv_text, question)},
    ]
    accumulated_thinking: list[str] = []
    state = _ToolLoopState(accumulated_thinking)
    budget = state.budget

    def _salvage(
        trigger_reason: str | None = None,
    ) -> tuple[str, str | None, list[dict], dict]:
        def _request() -> _SalvageCallResult:
            answer, salv_input, salv_output = _force_final_answer_anthropic(
                _budgeted_api_client(client, budget),
                model,
                system,
                messages,
                tools_list,
                max_tokens=max(
                    1,
                    min(SALVAGE_MAX_TOKENS, budget.remaining_tokens()),
                ),
            )
            return answer, salv_input, salv_output, {}

        return state.salvage(_request, trigger_reason)

    while True:
        terminal = state.begin_round(_salvage)
        if terminal is not None:
            return terminal
        kwargs["max_tokens"] = max(
            1,
            min(16_000, budget.remaining_tokens()),
        )
        kwargs["messages"] = messages
        try:
            msg = _budgeted_api_client(client, budget).messages.create(**kwargs)
        except Exception as exc:
            return state.provider_failure(exc)
        budget.add_usage(
            msg.usage.input_tokens,
            msg.usage.output_tokens,
        )

        tool_use_blocks, text_blocks, turn_thinking = [], [], []
        for block in msg.content:
            if block.type == "thinking":
                thinking_text = (block.thinking or "").strip()
                if thinking_text:
                    turn_thinking.append(thinking_text)
                    accumulated_thinking.append(thinking_text)
            elif block.type == "tool_use":
                tool_use_blocks.append(block)
            elif block.type == "text":
                text_blocks.append(block.text)

        messages.append({"role": "assistant", "content": msg.content})

        if not tool_use_blocks:
            answer = " ".join(text_blocks).strip()
            if not answer:
                # Thinking-only / no-text final turn: force a concise answer with a
                # capped, no-more-tools call instead of recording an empty (=INCORRECT) one.
                return _salvage()
            return state.finish(answer)

        reason = budget.hard_stop_reason()
        if reason is not None:
            return state.stop(reason)

        step_text = _step_thinking_text(
            " ".join(text_blocks),
            "\n\n".join(turn_thinking),
        )
        tool_results = []
        calls = [
            (block.id, block.name, block.input)
            for block in tool_use_blocks
        ]

        def _emit_result(call_id: str, result: str) -> None:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": result,
            })

        stop_reason, salvage_reason = _execute_tool_batch(
            calls,
            sandbox,
            csv_path,
            state.tool_use_log,
            step_text,
            budget,
            _use_structured_tool_input,
            _emit_result,
        )
        if stop_reason is not None:
            return state.stop(stop_reason)

        messages.append({"role": "user", "content": tool_results})
        if salvage_reason is not None:
            return _salvage(salvage_reason)


# ---------------------------------------------------------------------------
# Provider query registry
# ---------------------------------------------------------------------------

QUERY_FN: dict[str, Callable] = {
    "anthropic": query_anthropic,
    "openai": query_openai,
    "deepseek": query_deepseek,
    "bedrock": query_bedrock_deepseek,
}

QUERY_FN_TOOLS: dict[str, Callable] = {
    "anthropic": query_anthropic_tools,
    "openai": query_openai_style_tools,
    "deepseek": query_deepseek_tools,
    "bedrock": query_bedrock_deepseek_tools,
}


# ---------------------------------------------------------------------------
# Instance discovery
# ---------------------------------------------------------------------------


def discover_instances(
    instances_dir: Path,
    dataset: str | None = None,
    injector: str | list[str] | None = None,
    manifest_paths: list[Path] | None = None,
) -> list[Path]:
    """Return manifests sorted by CSV size (smallest datasets first).

    Args:
        dataset:  Suffix match against the dataset folder name. A full name
                  ('bike_sharing_100') selects one dataset; a size suffix
                  ('_1000') selects that size across every dataset while
                  excluding '_100'/'_10000'.
        injector: Substring match (or list of substrings) against the injector folder name.
    """
    manifests = resolve_manifest_paths(instances_dir, manifest_paths)
    if dataset:
        dataset = normalize_path_text(dataset)
        manifests = [
            m
            for m in manifests
            if normalize_path_text(m.parts[-4]).endswith(dataset)
        ]
    if injector:
        if isinstance(injector, str):
            injector = [injector]
        manifests = [m for m in manifests if any(inj in m.parts[-2] for inj in injector)]
    return sorted(manifests, key=lambda m: (m.parent / "table.csv").stat().st_size)


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def run_eval(
    models: list[str],
    instances_dir: Path,
    output_path: Path,
    dataset: str | None = None,
    injector: str | list[str] | None = None,
    template_ids: str | list[str] | None = None,
    enabled_tools: set = frozenset(),
    columns_only: bool = False,
    table_in_prompt: bool = False,
    debug: bool = False,
    manifest_paths: list[Path] | None = None,
    run_id: str | None = None,
    injection_delay_seconds: float = 0.0,
) -> list[dict]:
    injection_delay_seconds = float(injection_delay_seconds)
    if (
        not math.isfinite(injection_delay_seconds)
        or injection_delay_seconds < 0
    ):
        raise ValueError("injection_delay_seconds must be a finite nonnegative number")

    seen_models: set[str] = set()
    duplicate_models: list[str] = []
    for model in models:
        if model in seen_models and model not in duplicate_models:
            duplicate_models.append(model)
        seen_models.add(model)
    if duplicate_models:
        raise ValueError(
            "models must be unique; duplicate model(s): "
            + ", ".join(duplicate_models)
        )

    validate_atomic_output_target(output_path)
    manifests = discover_instances(
        instances_dir,
        dataset=dataset,
        injector=injector,
        manifest_paths=manifest_paths,
    )
    if isinstance(template_ids, str):
        template_ids = [template_ids]
    allowed_template_ids = set(template_ids) if template_ids else None
    # The columns-only baseline historically targets the 100-row variants when
    # the caller does not provide an explicit dataset scope.
    if columns_only and dataset is None:
        manifests = [m for m in manifests if m.parts[-4].endswith("_100")]

    # Only evaluate instances that passed validation and have no warning.
    valid_manifests = []
    for m in manifests:
        with open(m, encoding="utf-8") as f:
            manifest_data = json.load(f)
        validation = manifest_data.get("validation")
        if not validation or not validation.get("passed", False):
            reason = "not validated" if validation is None else "validation failed"
            print(f"[skip] {m.parent.relative_to(instances_dir)}: {reason}")
        elif "validation_warning" in manifest_data:
            print(f"[skip] {m.parent.relative_to(instances_dir)}: {manifest_data['validation_warning']}")
        else:
            valid_manifests.append(m)
    manifests = valid_manifests

    if not manifests:
        print(f"No validated manifests found under {instances_dir}")
        sys.exit(1)

    # Detect providers for each model up front
    model_providers: dict[str, str] = {}
    for model in models:
        try:
            model_providers[model] = detect_provider(model)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    # Probe interpreter launchability once for diagnostics. This does not test
    # the REPL protocol, scientific imports, or CSV setup, and it does not
    # disable run_python on failure. Every (QA, model) pair gets its own object.
    from sandbox import PythonSandbox
    if RUN_PYTHON_TOOL in enabled_tools:
        sandbox_probe = PythonSandbox()
        try:
            if sandbox_probe.ping():
                print(f"[sandbox] ready ({sandbox_probe._python})")
            else:
                print(f"[sandbox] WARNING: could not launch Python at {sandbox_probe._python}")
        finally:
            sandbox_probe.close()

    # Lazy client initialization — only for providers actually needed
    from dotenv import load_dotenv
    load_dotenv()

    clients: dict[str, object] = {}
    bedrock_tool_client_factory: Callable[[float], object] | None = None

    if any(p == "anthropic" for p in model_providers.values()):
        import anthropic as _anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set in .env")
            sys.exit(1)
        clients["anthropic"] = _anthropic.Anthropic(api_key=api_key)

    if any(p == "openai" for p in model_providers.values()):
        import openai as _openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set in .env")
            sys.exit(1)
        # Explicit per-attempt timeout: reasoning models (gpt-5*) can run long,
        # but an unbounded hang stalls the whole eval batch. 5 min per attempt
        # x (1+2 retries) bounds a single call to ~15 min worst case.
        clients["openai"] = _openai.OpenAI(api_key=api_key, timeout=300.0)

    if any(p == "deepseek" for p in model_providers.values()):
        import openai as _openai
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("ERROR: DEEPSEEK_API_KEY not set in .env")
            sys.exit(1)
        clients["deepseek"] = _openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    if any(p == "bedrock" for p in model_providers.values()):
        import boto3
        region = (
            os.environ.get("BEDROCK_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        bedrock_session = boto3.Session()
        if enabled_tools:
            from botocore.config import Config

            def _new_bedrock_tool_client(timeout_s: float):
                # Botocore exposes timeouts only when constructing a client.
                # Split the remaining request window across connection and read
                # phases so their configured maxima do not obviously add beyond it.
                window = max(0.001, float(timeout_s))
                connect_timeout = max(0.001, min(10.0, window / 4))
                read_timeout = max(0.001, window - connect_timeout)
                return bedrock_session.client(
                    "bedrock-runtime",
                    region_name=region,
                    config=Config(
                        connect_timeout=connect_timeout,
                        read_timeout=read_timeout,
                        retries={"total_max_attempts": 1, "mode": "standard"},
                    ),
                )

            bedrock_tool_client_factory = _new_bedrock_tool_client
            clients["bedrock"] = _new_bedrock_tool_client(
                min(MAX_TOOL_PROVIDER_CALL_S, MAX_TOOL_WALL_TIME_S)
            )
        else:
            clients["bedrock"] = bedrock_session.client(
                "bedrock-runtime",
                region_name=region,
            )

    results: list[dict] = []
    output_lifecycle = JsonResultLifecycle(
        output_path,
        kind="eval",
        records_key="results",
    )
    # Keep an explicit, recoverable checkpoint while the batch is running.
    # The formal output remains untouched until every manifest/model pair has
    # completed, including the valid zero-result case.
    output_lifecycle.checkpoint(results)
    _debug_printed = False

    evaluated_an_injection = False
    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        instance_dir = manifest_path.parent
        try:
            instance_path = instance_dir.resolve().relative_to(instances_dir.resolve()).as_posix()
        except ValueError:
            instance_path = str(instance_dir.resolve())
        # Prompt-data precedence is load_data > columns_only > table_in_prompt.
        if LOAD_DATA_TOOL in enabled_tools:
            csv_text = ""
        elif columns_only:
            import pandas as pd
            cols = pd.read_csv(instance_dir / "table.csv", nrows=0).columns.tolist()
            csv_text = "Columns: " + ", ".join(cols)
        elif table_in_prompt:
            csv_text = (instance_dir / "table.csv").read_text(encoding="utf-8")
        else:
            csv_text = ""

        dataset = manifest["dataset_name"]
        injector = manifest["phenomenon"]["injector_type"]

        delay_before_first_request = evaluated_an_injection
        injection_was_evaluated = False
        for qa in manifest["qa_pairs"]:
            if allowed_template_ids and qa["template_id"] not in allowed_template_ids:
                continue
            if qa["answer"] is None:
                continue

            question = qa["question"]
            expected = qa["answer"]
            template_id = qa["template_id"]
            answer_format = qa["answer_format"]

            for model in models:
                if delay_before_first_request and injection_delay_seconds:
                    print(
                        f"[throttle] waiting {injection_delay_seconds:g}s "
                        "before next injection",
                        flush=True,
                    )
                    time.sleep(injection_delay_seconds)
                    delay_before_first_request = False
                injection_was_evaluated = True
                provider = model_providers[model]
                use_tools = bool(enabled_tools) and provider in QUERY_FN_TOOLS

                print(f"[{model}] {dataset}/{injector} → ", end="", flush=True)

                query_fn = QUERY_FN_TOOLS[provider] if use_tools else QUERY_FN[provider]
                client = clients[provider]

                thinking: str | None = None
                tool_use_log: list[dict] = []
                metrics: dict = {}
                if debug and not _debug_printed:
                    if use_tools:
                        system = _build_system_prompt_tools(enabled_tools)
                    elif provider in {"deepseek", "bedrock"}:
                        system = DEEPSEEK_JSON_SYSTEM_PROMPT
                    else:
                        system = SYSTEM_PROMPT
                    user_content = _build_user_content(csv_text, question)
                    tools_list = (
                        _provider_tools(enabled_tools, _anthropic_tool)
                        if provider == "anthropic" else
                        _provider_tools(enabled_tools, _bedrock_tool)
                        if provider == "bedrock" else
                        _provider_tools(enabled_tools, _openai_tool)
                    ) if use_tools else []
                    _print_request(system, user_content, tools_list)
                    _debug_printed = True
                # Each (QA, model) pair owns one sandbox. The same object remains
                # alive for that answer's multi-turn tool calls; close() is
                # attempted in finally even if the model/API/tool loop raises.
                sandbox = None
                try:
                    if RUN_PYTHON_TOOL in enabled_tools:
                        sandbox = PythonSandbox()
                    if use_tools:
                        tool_query_kwargs: dict = {
                            "enabled_tools": enabled_tools,
                        }
                        if provider == "bedrock":
                            tool_query_kwargs["client_factory"] = (
                                bedrock_tool_client_factory
                            )
                        model_answer, thinking, tool_use_log, metrics = query_fn(
                            client, model, csv_text, question,
                            sandbox, instance_dir / "table.csv",
                            **tool_query_kwargs,
                        )
                    else:
                        model_answer, thinking, metrics = query_fn(
                            client, model, csv_text, question,
                        )
                except Exception as exc:
                    model_answer = f"ERROR: {exc}"
                finally:
                    if sandbox is not None:
                        sandbox.close()

                print(f"expected: {expected} | got: {model_answer}")

                result = {
                    "run_id": run_id or manifest.get("run_id"),
                    "seed": manifest.get("seed"),
                    "instance_path": instance_path,
                    "dataset": dataset,
                    "injector": injector,
                    "template_id": template_id,
                    "model": model,
                    "question": question,
                    "expected_answer": expected,
                    "model_answer": model_answer,
                    "answer_format": answer_format,
                }
                result["metrics"] = metrics
                if enabled_tools:
                    result["tools_enabled"] = sorted(enabled_tools)
                    result["tool_use"] = tool_use_log
                if thinking:
                    result["thinking"] = thinking
                results.append(result)
                output_lifecycle.checkpoint(results)

        if injection_was_evaluated:
            evaluated_an_injection = True

    for initialized_client in clients.values():
        close = getattr(initialized_client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    # Publish only a fully completed batch. The same-directory temporary file
    # makes replacement atomic, so readers see either the old complete output
    # or this complete output, never an in-progress list.
    output_lifecycle.publish(results)

    print(f"\nResults saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    if not results:
        return

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        counts[r["model"]][r["template_id"]] += 1

    templates = sorted({r["template_id"] for r in results})
    models = sorted(counts.keys())

    col_w = max(len(t) for t in templates) + 2
    model_w = max(len(m) for m in models) + 2
    header = "model".ljust(model_w) + "".join(t.ljust(col_w) for t in templates)
    print(f"\n{'=' * len(header)}")
    print("Summary (model × template counts)")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for m in models:
        row = m.ljust(model_w) + "".join(
            str(counts[m].get(t, 0)).ljust(col_w) for t in templates
        )
        print(row)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _filename_token(value: object, max_length: int = 64) -> str:
    """Return a portable, bounded label for a generated result filename."""
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_-")
    return (token or "value")[:max_length]


def _default_eval_output_path(
    *,
    models: list[str],
    instances_dir: Path,
    tools: list[str] | None = None,
    columns_only: bool = False,
    table_in_prompt: bool = False,
    run_id: str | None = None,
    dataset: str | None = None,
    injectors: list[str] | None = None,
    templates: list[str] | None = None,
) -> Path:
    """Build a collision-resistant auto output path for one eval condition."""
    signature = {
        "models": sorted(models),
        "instances_dir": Path(instances_dir).as_posix(),
        "tools": sorted(set(tools or [])),
        "columns_only": columns_only,
        "table_in_prompt": table_in_prompt,
        "run_id": (
            normalize_path_text(run_id) if run_id is not None else None
        ),
        "dataset": (
            normalize_path_text(dataset) if dataset is not None else None
        ),
        "injectors": sorted(set(injectors or [])),
        "templates": sorted(set(templates or [])),
    }
    digest = hashlib.sha256(
        json.dumps(
            signature,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]

    if len(models) == 1:
        model_label = _filename_token(models[0])
    else:
        model_label = f"{len(models)}models"
    base = DEFAULT_OUTPUT.parent / f"eval_results_{model_label}_{digest}.json"

    if run_id:
        run_scope = safe_path_component(run_id, "run_id")
        base = base.parent / "runs" / run_scope / base.name
    return base


def main() -> None:
    configure_cli_streams()
    parser = argparse.ArgumentParser(
        description=(
            "Run QA eval against Anthropic, OpenAI, DeepSeek, and Bedrock models.\n"
            "Provider is auto-detected from the model name:\n"
            "  - Anthropic: name starts with 'claude'\n"
            "  - OpenAI:    name starts with 'gpt-', 'o1', 'o3', or 'o4'\n"
            "  - DeepSeek:  name starts with 'deepseek-'\n"
            "  - Bedrock:   name is 'bedrock-deepseek-v3.2' or starts with 'bedrock:'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to evaluate (provider auto-detected). Required.",
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=DEFAULT_INSTANCES_DIR,
        help="Root directory containing instance manifests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for the JSON results file. When omitted, a portable model "
            "label and a hash of the full eval condition form the filename."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Select a run-scoped instance batch and scope auto-named output",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Filter instances by dataset name suffix (e.g. 'bike_sharing_100' or '_100')",
    )
    parser.add_argument(
        "--injector",
        nargs="+",
        default=None,
        help=(
            "Filter instances by injector name substring(s) "
            "(e.g. --injector fc_nonmonotone_peak fc_interaction_dominant)"
        ),
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        default=None,
        help="Evaluate only QA pairs whose template_id is in this list.",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=SUPPORTED_TOOLS,
        default=None,
        metavar="TOOL",
        help=(
            "Tools to expose to the model: "
            + ", ".join(SUPPORTED_TOOLS)
            + ", or any combination."
        ),
    )
    parser.add_argument(
        "--columns-only",
        action="store_true",
        default=False,
        help=(
            "Include only column names in the prompt; without --dataset, "
            "also restrict discovery to dataset names ending in '_100'"
        ),
    )
    parser.add_argument(
        "--table-in-prompt",
        action="store_true",
        default=False,
        help=(
            "Include the full CSV in the prompt (ignored when load_data or "
            "--columns-only takes precedence)"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print the full initial request (system prompt, user message, tools) before the first API call.",
    )
    parser.add_argument(
        "--injection-delay-seconds",
        type=_nonnegative_finite_float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Wait this many seconds between evaluated injections "
            "(default: 0)."
        ),
    )
    args = parser.parse_args()

    if args.run_id:
        args.run_id = safe_path_component(args.run_id, "run_id")
        ensure_portable_child_namespace(
            args.instances_dir / "runs",
            args.run_id,
            "run_id",
        )
        if args.output is None:
            ensure_portable_child_namespace(
                DEFAULT_OUTPUT.parent / "runs",
                args.run_id,
                "run_id",
            )

    if args.output is None:
        args.output = _default_eval_output_path(
            models=args.models,
            instances_dir=args.instances_dir,
            tools=args.tools,
            columns_only=args.columns_only,
            table_in_prompt=args.table_in_prompt,
            run_id=args.run_id,
            dataset=args.dataset,
            injectors=args.injector,
            templates=args.templates,
        )

    enabled_tools = set(args.tools) if args.tools else set()
    manifest_paths = None
    if args.run_id:
        run_instances_dir = args.instances_dir / "runs" / args.run_id
        manifest_paths = sorted(run_instances_dir.glob("**/manifest.json"))
    results = run_eval(
        args.models, args.instances_dir, args.output,
        dataset=args.dataset, injector=args.injector,
        template_ids=args.templates,
        enabled_tools=enabled_tools,
        columns_only=args.columns_only,
        table_in_prompt=args.table_in_prompt,
        debug=args.debug,
        manifest_paths=manifest_paths,
        run_id=args.run_id,
        injection_delay_seconds=args.injection_delay_seconds,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
