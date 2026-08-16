"""Timeout, retry, and failure-normalization behavior."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from evalreliability.candidates import Candidate
from evalreliability.errors import CandidateError
from evalreliability.models import (
    AttemptOutcome,
    AttemptRecord,
    CandidateRequest,
    CandidateResponse,
    FailureOrigin,
    FailureRecord,
)


@dataclass(frozen=True)
class InvocationPolicy:
    timeout_seconds: float = 30.0
    max_attempts: int = 2
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 4.0
    jitter_ratio: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "timeout_seconds",
            "initial_backoff_seconds",
            "max_backoff_seconds",
            "jitter_ratio",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
        if not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool):
            raise ValueError("max_attempts must be an integer")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("backoff values cannot be negative")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
            "initial_backoff_seconds": self.initial_backoff_seconds,
            "max_backoff_seconds": self.max_backoff_seconds,
            "jitter_ratio": self.jitter_ratio,
        }


@dataclass(frozen=True)
class InvocationResult:
    response: CandidateResponse | None
    attempts: tuple[AttemptRecord, ...]
    failure: FailureRecord | None


async def invoke_candidate(
    candidate: Candidate,
    request: CandidateRequest,
    policy: InvocationPolicy,
    *,
    seed: int | None,
    stage: str = "candidate",
    now: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.perf_counter,
) -> InvocationResult:
    """Invoke a candidate under an explicit retry policy and retain every attempt."""

    now = now or (lambda: datetime.now(UTC))
    attempts: list[AttemptRecord] = []
    final_failure: FailureRecord | None = None

    for attempt_number in range(1, policy.max_attempts + 1):
        started_at = now().isoformat().replace("+00:00", "Z")
        started = monotonic()
        try:
            response = await asyncio.wait_for(
                candidate.generate(request), timeout=policy.timeout_seconds
            )
            attempts.append(
                AttemptRecord(
                    attempt=attempt_number,
                    outcome=AttemptOutcome.SUCCESS,
                    started_at=started_at,
                    latency_ms=_milliseconds(monotonic() - started),
                )
            )
            return InvocationResult(response, tuple(attempts), None)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            final_failure = FailureRecord(
                origin=FailureOrigin.INFRASTRUCTURE,
                stage=stage,
                code="candidate_timeout",
                message=f"call exceeded {policy.timeout_seconds:g}s timeout",
                retryable=True,
                exception_type=type(exc).__name__,
            )
            outcome = AttemptOutcome.TIMEOUT
        except CandidateError as exc:
            final_failure = FailureRecord(
                origin=exc.origin,
                stage=stage,
                code=exc.code,
                message=str(exc),
                retryable=exc.retryable,
                exception_type=type(exc).__name__,
            )
            outcome = AttemptOutcome.ERROR
        except Exception as exc:  # adapter bugs and transport libraries are infrastructure
            final_failure = FailureRecord(
                origin=FailureOrigin.INFRASTRUCTURE,
                stage=stage,
                code="candidate_exception",
                message=str(exc) or type(exc).__name__,
                retryable=True,
                exception_type=type(exc).__name__,
            )
            outcome = AttemptOutcome.ERROR

        attempts.append(
            AttemptRecord(
                attempt=attempt_number,
                outcome=outcome,
                started_at=started_at,
                latency_ms=_milliseconds(monotonic() - started),
                failure=final_failure,
            )
        )
        if not final_failure.retryable or attempt_number == policy.max_attempts:
            break
        delay = _retry_delay(policy, request.example_id, attempt_number, seed)
        if delay:
            await asyncio.sleep(delay)

    return InvocationResult(None, tuple(attempts), final_failure)


def _retry_delay(
    policy: InvocationPolicy, example_id: str, attempt_number: int, seed: int | None
) -> float:
    base: float = min(
        policy.initial_backoff_seconds * float(2 ** (attempt_number - 1)),
        policy.max_backoff_seconds,
    )
    if base == 0 or policy.jitter_ratio == 0:
        return base
    material = f"{seed}:{example_id}:{attempt_number}".encode()
    unit: float = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / (2**64 - 1)
    multiplier: float = 1 - policy.jitter_ratio + (2 * policy.jitter_ratio * unit)
    return base * multiplier


def _milliseconds(seconds: float) -> float:
    return round(seconds * 1000, 3)
