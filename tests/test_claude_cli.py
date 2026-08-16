from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import evalreliability.claude_cli as claude_cli_module
from evalreliability.claude_cli import ClaudeCliCandidate
from evalreliability.errors import CandidateError
from evalreliability.models import CandidateRequest, FailureOrigin


class FakeProcess:
    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int | None = 0,
        delay_seconds: float = 0.0,
    ) -> None:
        self.stdout = stdout.encode()
        self.stderr = stderr.encode()
        self.returncode = returncode
        self.delay_seconds = delay_seconds
        self.terminated = False
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return self.stdout, self.stderr

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


def _request() -> CandidateRequest:
    return CandidateRequest(example_id="smoke", input="Return only: ok")


def _success_envelope() -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "ok",
        "session_id": "session-123",
        "uuid": "request-456",
        "duration_ms": 950,
        "duration_api_ms": 730,
        "ttft_ms": 310,
        "num_turns": 1,
        "total_cost_usd": 0.012345000000000002,
        "usage": {
            "input_tokens": 208,
            "output_tokens": 7,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 20,
        },
        "modelUsage": {
            "claude-haiku-4-5": {
                "inputTokens": 0,
                "outputTokens": 0,
                "cacheCreationInputTokens": 0,
                "cacheReadInputTokens": 0,
                "costUSD": 0.0,
            },
            "claude-sonnet-4-6": {
                "inputTokens": 208,
                "outputTokens": 7,
                "cacheCreationInputTokens": 10,
                "cacheReadInputTokens": 20,
                "costUSD": 0.012345,
            },
        },
    }


@pytest.mark.asyncio
async def test_claude_cli_uses_argument_array_clean_directory_and_normalizes_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(stdout=json.dumps(_success_envelope()))
    invocation: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*args: str, **kwargs: Any) -> FakeProcess:
        invocation["args"] = args
        invocation["kwargs"] = kwargs
        working_directory = Path(kwargs["cwd"])
        invocation["working_directory"] = working_directory
        assert working_directory.is_dir()
        assert not list(working_directory.iterdir())
        return process

    monkeypatch.setattr(
        claude_cli_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    candidate = ClaudeCliCandidate(
        model="sonnet",
        system_prompt="Minimal system prompt",
        timeout_seconds=3,
        max_turns=1,
    )

    response = await candidate.generate(_request())

    assert invocation["args"] == (
        "claude",
        "-p",
        "Return only: ok",
        "--output-format",
        "json",
        "--model",
        "sonnet",
        "--max-turns",
        "1",
        "--tools",
        "",
        "--exclude-dynamic-system-prompt-sections",
        "--system-prompt",
        "Minimal system prompt",
    )
    assert "shell" not in invocation["kwargs"]
    assert not invocation["working_directory"].exists()
    assert response.output == "ok"
    assert response.model_id == "claude-sonnet-4-6"
    assert response.provider == "anthropic_claude_code_cli"
    assert response.usage.input_tokens == 208
    assert response.usage.output_tokens == 7
    assert response.usage.total_tokens == 245
    assert response.usage.cost_usd == 0.012345
    assert response.metadata == {
        "transport": "claude_code_cli",
        "requested_model": "sonnet",
        "canonical_model": "claude-sonnet-4-6",
        "usage_model_ids": ["claude-haiku-4-5", "claude-sonnet-4-6"],
        "session_id": "session-123",
        "request_id": "request-456",
        "duration_ms": 950.0,
        "duration_api_ms": 730.0,
        "ttft_ms": 310.0,
        "num_turns": 1,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 20,
        "cache_tokens": 30,
        "provider_reported_usage_cost_usd": 0.012345,
        "cost_interpretation": "provider_usage_accounting_not_billing_evidence",
        "provider_reported_total_cost_usd": 0.012345,
    }


@pytest.mark.asyncio
async def test_claude_cli_normalizes_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def missing_executable(*args: str, **kwargs: Any) -> FakeProcess:
        raise FileNotFoundError("claude")

    monkeypatch.setattr(
        claude_cli_module.asyncio,
        "create_subprocess_exec",
        missing_executable,
    )

    with pytest.raises(CandidateError) as captured:
        await ClaudeCliCandidate().generate(_request())

    assert captured.value.code == "claude_cli_executable_missing"
    assert captured.value.origin == FailureOrigin.INFRASTRUCTURE
    assert captured.value.retryable is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("process", "code", "origin", "retryable"),
    [
        (
            FakeProcess(returncode=1, stderr="Authentication failed: please run /login"),
            "claude_cli_authentication_failed",
            FailureOrigin.INFRASTRUCTURE,
            False,
        ),
        (
            FakeProcess(returncode=0, stdout="not-json"),
            "claude_cli_malformed_json",
            FailureOrigin.INFRASTRUCTURE,
            False,
        ),
        (
            FakeProcess(
                returncode=0,
                stdout=json.dumps(
                    {
                        "type": "result",
                        "subtype": "error_during_execution",
                        "is_error": True,
                        "result": "API overloaded",
                    }
                ),
            ),
            "claude_cli_model_failure",
            FailureOrigin.MODEL,
            True,
        ),
        (
            FakeProcess(returncode=2, stderr="unexpected command failure"),
            "claude_cli_process_failed",
            FailureOrigin.INFRASTRUCTURE,
            False,
        ),
        (
            FakeProcess(returncode=0, stdout='{"is_error":"false","result":"ok"}'),
            "claude_cli_malformed_json",
            FailureOrigin.INFRASTRUCTURE,
            False,
        ),
    ],
)
async def test_claude_cli_failure_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
    process: FakeProcess,
    code: str,
    origin: FailureOrigin,
    retryable: bool,
) -> None:
    async def fake_create_subprocess_exec(*args: str, **kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(
        claude_cli_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    with pytest.raises(CandidateError) as captured:
        await ClaudeCliCandidate().generate(_request())

    assert captured.value.code == code
    assert captured.value.origin == origin
    assert captured.value.retryable is retryable


@pytest.mark.asyncio
async def test_claude_cli_timeout_terminates_process(monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeProcess(returncode=None, delay_seconds=1)

    async def fake_create_subprocess_exec(*args: str, **kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(
        claude_cli_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    with pytest.raises(CandidateError) as captured:
        await ClaudeCliCandidate(timeout_seconds=0.001).generate(_request())

    assert captured.value.code == "claude_cli_timeout"
    assert captured.value.origin == FailureOrigin.INFRASTRUCTURE
    assert captured.value.retryable is True
    assert process.terminated is True


def test_claude_cli_configuration_contains_no_authentication_material() -> None:
    candidate = ClaudeCliCandidate(model="sonnet", timeout_seconds=45, max_turns=1)

    assert candidate.configuration() == {
        "type": "claude_cli",
        "executable": "claude",
        "model": "sonnet",
        "system_prompt": "You are a model under evaluation. Answer only the user request.",
        "timeout_seconds": 45.0,
        "max_turns": 1,
        "output_format": "json",
        "tools_disabled": True,
        "dynamic_system_prompt_sections_excluded": True,
    }
