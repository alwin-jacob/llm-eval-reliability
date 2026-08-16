"""Claude Code CLI transport for real model evaluation."""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
from dataclasses import dataclass
from typing import Any

from evalreliability.candidates import Candidate
from evalreliability.errors import CandidateError
from evalreliability.models import CandidateRequest, CandidateResponse, FailureOrigin, Usage

DEFAULT_SYSTEM_PROMPT = "You are a model under evaluation. Answer only the user request."

_AUTHENTICATION_MARKERS = (
    "authentication failed",
    "authentication_error",
    "invalid api key",
    "login required",
    "not logged in",
    "oauth token has expired",
    "please run /login",
    "unauthorized",
)

_TRANSIENT_MARKERS = (
    "connection reset",
    "internal server error",
    "network error",
    "overloaded",
    "rate limit",
    "rate_limit",
    "temporarily unavailable",
    "timed out",
    "timeout",
)


class _MalformedEnvelope(ValueError):
    """The CLI completed but did not return its documented JSON result envelope."""


@dataclass(frozen=True)
class _NormalizedUsage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    cost_usd: float | None
    model_ids: tuple[str, ...]


class ClaudeCliCandidate(Candidate):
    """Invoke one model response through an installed Claude Code CLI.

    Authentication is intentionally delegated to the user's existing CLI session. The
    adapter never reads, stores, or accepts credentials.
    """

    def __init__(
        self,
        *,
        model: str = "sonnet",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout_seconds: float = 120.0,
        max_turns: int = 1,
        executable: str = "claude",
    ) -> None:
        if not isinstance(model, str) or not model:
            raise ValueError("Claude CLI model must be a non-empty string")
        if not isinstance(system_prompt, str) or not system_prompt:
            raise ValueError("Claude CLI system_prompt must be a non-empty string")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise ValueError("Claude CLI timeout_seconds must be positive")
        if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1:
            raise ValueError("Claude CLI max_turns must be a positive integer")
        if not isinstance(executable, str) or not executable:
            raise ValueError("Claude CLI executable must be a non-empty string")
        self.model = model
        self.system_prompt = system_prompt
        self.timeout_seconds = float(timeout_seconds)
        self.max_turns = max_turns
        self.executable = executable

    @property
    def identifier(self) -> str:
        return f"claude-cli:{self.model}"

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "claude_cli",
            "executable": self.executable,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "timeout_seconds": self.timeout_seconds,
            "max_turns": self.max_turns,
            "output_format": "json",
            "tools_disabled": True,
            "dynamic_system_prompt_sections_excluded": True,
        }

    async def generate(self, request: CandidateRequest) -> CandidateResponse:
        arguments = self._arguments(request.input)
        with tempfile.TemporaryDirectory(prefix="evalrel-claude-") as working_directory:
            try:
                process = await asyncio.create_subprocess_exec(
                    *arguments,
                    cwd=working_directory,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                raise CandidateError(
                    f"Claude CLI executable not found: {self.executable!r}",
                    code="claude_cli_executable_missing",
                    retryable=False,
                    origin=FailureOrigin.INFRASTRUCTURE,
                ) from exc
            except OSError as exc:
                raise CandidateError(
                    f"Claude CLI process could not start: {exc}",
                    code="claude_cli_process_start_failed",
                    retryable=False,
                    origin=FailureOrigin.INFRASTRUCTURE,
                ) from exc

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout_seconds
                )
            except asyncio.CancelledError:
                await _stop_process(process)
                raise
            except TimeoutError as exc:
                await _stop_process(process)
                raise CandidateError(
                    f"Claude CLI exceeded its {self.timeout_seconds:g}s process timeout",
                    code="claude_cli_timeout",
                    retryable=True,
                    origin=FailureOrigin.INFRASTRUCTURE,
                ) from exc
            except OSError as exc:
                await _stop_process(process)
                raise CandidateError(
                    f"Claude CLI process communication failed: {exc}",
                    code="claude_cli_process_failed",
                    retryable=_is_transient(str(exc)),
                    origin=FailureOrigin.INFRASTRUCTURE,
                ) from exc

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        return self._response_from_process(process.returncode, stdout_text, stderr_text)

    def _arguments(self, prompt: str) -> list[str]:
        return [
            self.executable,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model,
            "--max-turns",
            str(self.max_turns),
            "--tools",
            "",
            "--exclude-dynamic-system-prompt-sections",
            "--system-prompt",
            self.system_prompt,
        ]

    def _response_from_process(
        self, returncode: int | None, stdout: str, stderr: str
    ) -> CandidateResponse:
        envelope: dict[str, Any] | None = None
        envelope_error: _MalformedEnvelope | None = None
        if stdout:
            try:
                envelope = _load_envelope(stdout)
            except _MalformedEnvelope as exc:
                envelope_error = exc

        combined_error = _error_text(envelope, stderr)
        if returncode != 0:
            if _is_authentication_failure(combined_error):
                raise CandidateError(
                    _bounded_message("Claude CLI authentication failed", combined_error),
                    code="claude_cli_authentication_failed",
                    retryable=False,
                    origin=FailureOrigin.INFRASTRUCTURE,
                )
            if envelope is not None and _envelope_is_error(envelope):
                raise CandidateError(
                    _bounded_message("Claude model/API call failed", combined_error),
                    code="claude_cli_model_failure",
                    retryable=_is_transient(combined_error),
                    origin=FailureOrigin.MODEL,
                )
            detail = stderr or (str(envelope_error) if envelope_error else "no error output")
            status = returncode if returncode is not None else "unknown"
            raise CandidateError(
                _bounded_message(
                    f"Claude CLI exited with status {status}",
                    detail,
                ),
                code="claude_cli_process_failed",
                retryable=_is_transient(detail),
                origin=FailureOrigin.INFRASTRUCTURE,
            )

        if envelope_error is not None:
            raise CandidateError(
                f"Claude CLI returned malformed JSON: {envelope_error}",
                code="claude_cli_malformed_json",
                retryable=False,
                origin=FailureOrigin.INFRASTRUCTURE,
            ) from envelope_error
        if envelope is None:
            raise CandidateError(
                "Claude CLI returned an empty JSON result envelope",
                code="claude_cli_malformed_json",
                retryable=False,
                origin=FailureOrigin.INFRASTRUCTURE,
            )
        if _envelope_is_error(envelope):
            if _is_authentication_failure(combined_error):
                raise CandidateError(
                    _bounded_message("Claude CLI authentication failed", combined_error),
                    code="claude_cli_authentication_failed",
                    retryable=False,
                    origin=FailureOrigin.INFRASTRUCTURE,
                )
            raise CandidateError(
                _bounded_message("Claude model/API call failed", combined_error),
                code="claude_cli_model_failure",
                retryable=_is_transient(combined_error),
                origin=FailureOrigin.MODEL,
            )

        try:
            return _success_response(envelope, requested_model=self.model)
        except _MalformedEnvelope as exc:
            raise CandidateError(
                f"Claude CLI returned a malformed JSON result envelope: {exc}",
                code="claude_cli_malformed_json",
                retryable=False,
                origin=FailureOrigin.INFRASTRUCTURE,
            ) from exc


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        await process.wait()


def _load_envelope(stdout: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise _MalformedEnvelope(f"non-finite number {value}")

    try:
        payload: Any = json.loads(stdout, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise _MalformedEnvelope(f"invalid JSON at line {exc.lineno}, column {exc.colno}") from exc
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise _MalformedEnvelope("root must be a JSON object with string keys")
    _envelope_is_error(payload)
    return payload


def _envelope_is_error(envelope: dict[str, Any]) -> bool:
    is_error = envelope.get("is_error")
    if is_error is not None and not isinstance(is_error, bool):
        raise _MalformedEnvelope("is_error must be a boolean when present")
    subtype = envelope.get("subtype")
    if subtype is not None and not isinstance(subtype, str):
        raise _MalformedEnvelope("subtype must be a string when present")
    return is_error is True or (isinstance(subtype, str) and subtype.startswith("error"))


def _success_response(envelope: dict[str, Any], *, requested_model: str) -> CandidateResponse:
    envelope_type = envelope.get("type")
    if envelope_type is not None and envelope_type != "result":
        raise _MalformedEnvelope("type must be 'result' when present")
    result = envelope.get("result")
    if not isinstance(result, str):
        raise _MalformedEnvelope("result must be a string")

    normalized = _normalize_usage(envelope)
    canonical_model = _canonical_model(envelope, normalized.model_ids, requested_model)
    metadata: dict[str, Any] = {
        "transport": "claude_code_cli",
        "requested_model": requested_model,
    }
    if canonical_model is not None:
        metadata["canonical_model"] = canonical_model
    if normalized.model_ids:
        metadata["usage_model_ids"] = list(normalized.model_ids)
    _copy_optional_string(envelope, "session_id", metadata)
    request_id = _optional_string(envelope, "request_id") or _optional_string(envelope, "uuid")
    if request_id is not None:
        metadata["request_id"] = request_id
    _copy_optional_number(envelope, "duration_ms", metadata)
    _copy_optional_number(envelope, "duration_api_ms", metadata)
    ttft = _optional_number(envelope, "ttft_ms")
    if ttft is None:
        ttft = _optional_number(envelope, "time_to_first_token_ms")
    if ttft is not None:
        metadata["ttft_ms"] = ttft
    _copy_optional_int(envelope, "num_turns", metadata)

    cache_total = _sum_available(
        normalized.cache_creation_input_tokens,
        normalized.cache_read_input_tokens,
    )
    if normalized.cache_creation_input_tokens is not None:
        metadata["cache_creation_input_tokens"] = normalized.cache_creation_input_tokens
    if normalized.cache_read_input_tokens is not None:
        metadata["cache_read_input_tokens"] = normalized.cache_read_input_tokens
    if cache_total is not None:
        metadata["cache_tokens"] = cache_total
    if normalized.cost_usd is not None:
        metadata["provider_reported_usage_cost_usd"] = normalized.cost_usd
        metadata["cost_interpretation"] = "provider_usage_accounting_not_billing_evidence"
    reported_total_cost = _optional_number(envelope, "total_cost_usd")
    if reported_total_cost is not None:
        metadata["provider_reported_total_cost_usd"] = round(reported_total_cost, 12)

    subtype = _optional_string(envelope, "subtype")
    finish_reason = _optional_string(envelope, "stop_reason") or subtype
    return CandidateResponse(
        output=result,
        model_id=canonical_model or requested_model,
        provider="anthropic_claude_code_cli",
        finish_reason=finish_reason,
        usage=Usage(
            input_tokens=normalized.input_tokens,
            output_tokens=normalized.output_tokens,
            total_tokens=normalized.total_tokens,
            cost_usd=normalized.cost_usd,
        ),
        metadata=metadata,
    )


def _normalize_usage(envelope: dict[str, Any]) -> _NormalizedUsage:
    usage = _optional_mapping(envelope, "usage")
    model_usage = _model_usage(envelope)
    input_tokens = _optional_int(usage, "input_tokens")
    output_tokens = _optional_int(usage, "output_tokens")
    cache_creation = _optional_int(usage, "cache_creation_input_tokens")
    cache_read = _optional_int(usage, "cache_read_input_tokens")
    if input_tokens is None:
        input_tokens = _sum_model_int(model_usage, "inputTokens")
    if output_tokens is None:
        output_tokens = _sum_model_int(model_usage, "outputTokens")
    if cache_creation is None:
        cache_creation = _sum_model_int(model_usage, "cacheCreationInputTokens")
    if cache_read is None:
        cache_read = _sum_model_int(model_usage, "cacheReadInputTokens")

    total_tokens = _optional_int(usage, "total_tokens")
    if total_tokens is None:
        total_tokens = _sum_available(input_tokens, output_tokens, cache_creation, cache_read)

    cost = _sum_model_number(model_usage, "costUSD")
    if cost is None:
        cost = _optional_number(envelope, "cost_usd")
    if cost is None:
        cost = _optional_number(envelope, "total_cost_usd")
    return _NormalizedUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        cost_usd=cost,
        model_ids=tuple(sorted(model_usage)),
    )


def _model_usage(envelope: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = envelope.get("modelUsage")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise _MalformedEnvelope("modelUsage must be an object when present")
    output: dict[str, dict[str, Any]] = {}
    for model_id, fields in raw.items():
        if not isinstance(model_id, str) or not isinstance(fields, dict):
            raise _MalformedEnvelope("modelUsage must map model IDs to objects")
        output[model_id] = fields
    return output


def _canonical_model(
    envelope: dict[str, Any], model_ids: tuple[str, ...], requested_model: str
) -> str | None:
    direct = _optional_string(envelope, "model")
    if direct is not None:
        return direct
    matching = [
        model_id for model_id in model_ids if requested_model.casefold() in model_id.casefold()
    ]
    if len(matching) == 1:
        return matching[0]
    if len(model_ids) == 1:
        return model_ids[0]
    return None


def _optional_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _MalformedEnvelope(f"{key} must be an object when present")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _MalformedEnvelope(f"{key} must be a string when present")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _MalformedEnvelope(f"{key} must be a non-negative integer when present")
    return value


def _optional_number(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise _MalformedEnvelope(f"{key} must be a non-negative number when present")
    return float(value)


def _sum_model_int(model_usage: dict[str, dict[str, Any]], key: str) -> int | None:
    values = [_optional_int(fields, key) for fields in model_usage.values()]
    available = [value for value in values if value is not None]
    return sum(available) if available else None


def _sum_model_number(model_usage: dict[str, dict[str, Any]], key: str) -> float | None:
    values = [_optional_number(fields, key) for fields in model_usage.values()]
    available = [value for value in values if value is not None]
    return round(sum(available), 12) if available else None


def _sum_available(*values: int | None) -> int | None:
    available = [value for value in values if value is not None]
    return sum(available) if available else None


def _copy_optional_string(source: dict[str, Any], key: str, target: dict[str, Any]) -> None:
    value = _optional_string(source, key)
    if value is not None:
        target[key] = value


def _copy_optional_number(source: dict[str, Any], key: str, target: dict[str, Any]) -> None:
    value = _optional_number(source, key)
    if value is not None:
        target[key] = value


def _copy_optional_int(source: dict[str, Any], key: str, target: dict[str, Any]) -> None:
    value = _optional_int(source, key)
    if value is not None:
        target[key] = value


def _error_text(envelope: dict[str, Any] | None, stderr: str) -> str:
    parts: list[str] = []
    if envelope is not None:
        for key in ("error", "message", "result"):
            value = envelope.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    if stderr:
        parts.append(stderr)
    return " | ".join(parts) or "Claude CLI reported an unspecified error"


def _is_authentication_failure(message: str) -> bool:
    normalized = message.casefold()
    return any(marker in normalized for marker in _AUTHENTICATION_MARKERS)


def _is_transient(message: str) -> bool:
    normalized = message.casefold()
    return any(marker in normalized for marker in _TRANSIENT_MARKERS)


def _bounded_message(prefix: str, detail: str, *, limit: int = 500) -> str:
    compact = " ".join(detail.split())
    if len(compact) > limit:
        compact = f"{compact[: limit - 3]}..."
    return f"{prefix}: {compact}"


__all__ = ["ClaudeCliCandidate", "DEFAULT_SYSTEM_PROMPT"]
