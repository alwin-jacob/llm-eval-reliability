from __future__ import annotations

import asyncio

import pytest

from evalreliability.candidates import Candidate
from evalreliability.errors import CandidateError
from evalreliability.models import (
    AttemptOutcome,
    CandidateRequest,
    CandidateResponse,
    FailureOrigin,
)
from evalreliability.reliability import InvocationPolicy, invoke_candidate


class FlakyCandidate(Candidate):
    def __init__(self, failures: int, *, typed: bool = True) -> None:
        self.calls = 0
        self.failures = failures
        self.typed = typed

    @property
    def identifier(self) -> str:
        return "flaky"

    def configuration(self) -> dict[str, object]:
        return {"type": "test", "failures": self.failures}

    async def generate(self, request: CandidateRequest) -> CandidateResponse:
        self.calls += 1
        if self.calls <= self.failures:
            if self.typed:
                raise CandidateError(
                    "provider overloaded",
                    code="provider_overloaded",
                    retryable=True,
                    origin=FailureOrigin.INFRASTRUCTURE,
                )
            raise RuntimeError("adapter exploded")
        return CandidateResponse(output=f"ok:{request.example_id}")


class SlowCandidate(Candidate):
    @property
    def identifier(self) -> str:
        return "slow"

    def configuration(self) -> dict[str, object]:
        return {"type": "test"}

    async def generate(self, request: CandidateRequest) -> CandidateResponse:
        await asyncio.sleep(1)
        return CandidateResponse(output="late")


@pytest.mark.asyncio
async def test_retry_preserves_failed_attempt_and_then_succeeds() -> None:
    candidate = FlakyCandidate(failures=1)
    result = await invoke_candidate(
        candidate,
        CandidateRequest("example", "input"),
        InvocationPolicy(max_attempts=2, initial_backoff_seconds=0, jitter_ratio=0),
        seed=7,
    )

    assert result.response is not None
    assert result.response.output == "ok:example"
    assert [attempt.outcome for attempt in result.attempts] == [
        AttemptOutcome.ERROR,
        AttemptOutcome.SUCCESS,
    ]
    assert result.attempts[0].failure is not None
    assert result.attempts[0].failure.code == "provider_overloaded"


@pytest.mark.asyncio
async def test_non_retryable_model_error_stops_immediately() -> None:
    class RejectedCandidate(FlakyCandidate):
        async def generate(self, request: CandidateRequest) -> CandidateResponse:
            self.calls += 1
            raise CandidateError(
                "request rejected",
                code="content_filter",
                retryable=False,
                origin=FailureOrigin.MODEL,
            )

    candidate = RejectedCandidate(failures=10)
    result = await invoke_candidate(
        candidate,
        CandidateRequest("example", "input"),
        InvocationPolicy(max_attempts=3, initial_backoff_seconds=0),
        seed=0,
    )

    assert candidate.calls == 1
    assert result.response is None
    assert result.failure is not None
    assert result.failure.origin == FailureOrigin.MODEL
    assert result.failure.retryable is False


@pytest.mark.asyncio
async def test_timeouts_are_bounded_and_recorded_per_attempt() -> None:
    result = await invoke_candidate(
        SlowCandidate(),
        CandidateRequest("example", "input"),
        InvocationPolicy(
            timeout_seconds=0.01,
            max_attempts=2,
            initial_backoff_seconds=0,
            jitter_ratio=0,
        ),
        seed=0,
    )

    assert result.response is None
    assert len(result.attempts) == 2
    assert all(attempt.outcome == AttemptOutcome.TIMEOUT for attempt in result.attempts)
    assert result.failure is not None
    assert result.failure.code == "candidate_timeout"
    assert result.failure.origin == FailureOrigin.INFRASTRUCTURE


@pytest.mark.asyncio
async def test_unexpected_adapter_exception_is_infrastructure_failure() -> None:
    candidate = FlakyCandidate(failures=1, typed=False)
    result = await invoke_candidate(
        candidate,
        CandidateRequest("example", "input"),
        InvocationPolicy(max_attempts=1, initial_backoff_seconds=0),
        seed=None,
    )

    assert result.failure is not None
    assert result.failure.code == "candidate_exception"
    assert result.failure.origin == FailureOrigin.INFRASTRUCTURE


def test_invocation_policy_rejects_boolean_numeric_values() -> None:
    with pytest.raises(ValueError, match="max_attempts must be an integer"):
        InvocationPolicy(max_attempts=True)
    with pytest.raises(ValueError, match="timeout_seconds must be numeric"):
        InvocationPolicy(timeout_seconds=False)
