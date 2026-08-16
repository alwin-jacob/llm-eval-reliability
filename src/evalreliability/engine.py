"""Concurrent evaluation orchestration with per-example and per-scorer isolation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from evalreliability import __version__
from evalreliability.candidates import Candidate
from evalreliability.dataset import EvaluationDataset
from evalreliability.errors import ConfigurationError, EvaluatorError
from evalreliability.models import (
    CandidateRequest,
    EvaluationExample,
    EvaluationRun,
    ExampleResult,
    ExampleStatus,
    FailureOrigin,
    FailureRecord,
    RunMetadata,
    ScoreOutcome,
    ScoreResult,
)
from evalreliability.reliability import InvocationPolicy, invoke_candidate
from evalreliability.scorers import Scorer


@dataclass(frozen=True)
class EvaluationConfig:
    concurrency: int = 4
    scorer_timeout_seconds: float = 30.0
    seed: int | None = 0
    candidate_policy: InvocationPolicy = InvocationPolicy()

    def __post_init__(self) -> None:
        if not isinstance(self.concurrency, int) or isinstance(self.concurrency, bool):
            raise ValueError("concurrency must be an integer")
        if not isinstance(self.scorer_timeout_seconds, (int, float)) or isinstance(
            self.scorer_timeout_seconds, bool
        ):
            raise ValueError("scorer_timeout_seconds must be numeric")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise ValueError("seed must be an integer or null")
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.scorer_timeout_seconds <= 0:
            raise ValueError("scorer_timeout_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "scorer_timeout_seconds": self.scorer_timeout_seconds,
            "seed": self.seed,
            "candidate_invocation": self.candidate_policy.to_dict(),
        }


class EvaluationEngine:
    def __init__(
        self,
        *,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
        environment: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._now = now or (lambda: datetime.now(UTC))
        self._monotonic = monotonic
        self._environment = environment or collect_environment

    async def evaluate(
        self,
        dataset: EvaluationDataset,
        candidate: Candidate,
        scorers: Sequence[Scorer],
        config: EvaluationConfig | None = None,
    ) -> EvaluationRun:
        if not scorers:
            raise ValueError("at least one scorer is required")
        scorer_names = [scorer.name for scorer in scorers]
        if len(scorer_names) != len(set(scorer_names)):
            raise ValueError("scorer names must be unique within a run")
        config = config or EvaluationConfig()
        started_dt = self._now()
        started = self._monotonic()
        candidate_id = candidate.identifier
        candidate_config = candidate.configuration()
        scorer_configs = tuple(scorer.configuration() for scorer in scorers)
        fingerprint = _configuration_fingerprint(
            dataset, candidate_id, candidate_config, scorer_configs, config
        )
        run_id = f"{_run_timestamp(started_dt)}-{fingerprint[:12]}"
        semaphore = asyncio.Semaphore(config.concurrency)

        async def evaluate_isolated(example: EvaluationExample) -> ExampleResult:
            async with semaphore:
                example_started = self._monotonic()
                try:
                    return await self._evaluate_example(example, candidate, scorers, config)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # final isolation boundary for engine defects
                    failure = FailureRecord(
                        origin=FailureOrigin.INFRASTRUCTURE,
                        stage="engine",
                        code="engine_exception",
                        message=str(exc) or type(exc).__name__,
                        retryable=False,
                        exception_type=type(exc).__name__,
                    )
                    return ExampleResult(
                        example_id=example.id,
                        status=ExampleStatus.FAILED,
                        input=example.input,
                        expected=example.expected,
                        metadata=example.metadata,
                        response=None,
                        attempts=(),
                        scores=(),
                        failures=(failure,),
                        latency_ms=_milliseconds(self._monotonic() - example_started),
                    )

        results = await asyncio.gather(*(evaluate_isolated(item) for item in dataset.examples))
        completed_dt = self._now()
        metadata = RunMetadata(
            schema_version="1.0",
            run_id=run_id,
            config_fingerprint=fingerprint,
            started_at=_isoformat(started_dt),
            completed_at=_isoformat(completed_dt),
            duration_ms=_milliseconds(self._monotonic() - started),
            dataset=dataset.descriptor,
            candidate_id=candidate_id,
            candidate_config=candidate_config,
            scorer_configs=scorer_configs,
            evaluation_config=config.to_dict(),
            environment=self._environment(),
        )
        return EvaluationRun(metadata=metadata, examples=tuple(results))

    async def _evaluate_example(
        self,
        example: EvaluationExample,
        candidate: Candidate,
        scorers: Sequence[Scorer],
        config: EvaluationConfig,
    ) -> ExampleResult:
        started = self._monotonic()
        request = CandidateRequest(
            example_id=example.id,
            input=example.input,
            context={"metadata": example.metadata},
        )
        invocation = await invoke_candidate(
            candidate,
            request,
            config.candidate_policy,
            seed=config.seed,
            now=self._now,
            monotonic=self._monotonic,
        )
        if invocation.response is None:
            candidate_failures = (invocation.failure,) if invocation.failure else ()
            return ExampleResult(
                example_id=example.id,
                status=ExampleStatus.FAILED,
                input=example.input,
                expected=example.expected,
                metadata=example.metadata,
                response=None,
                attempts=invocation.attempts,
                scores=(),
                failures=candidate_failures,
                latency_ms=_milliseconds(self._monotonic() - started),
            )

        candidate_response = invocation.response

        async def score_isolated(scorer: Scorer) -> ScoreResult:
            score_started = self._monotonic()
            error_details: dict[str, Any] = {}
            try:
                result = await asyncio.wait_for(
                    scorer.score(example, candidate_response),
                    timeout=config.scorer_timeout_seconds,
                )
                if result.scorer_name != scorer.name:
                    raise EvaluatorError(
                        f"scorer {scorer.name!r} returned name {result.scorer_name!r}",
                        code="scorer_contract_violation",
                    )
                return replace(result, latency_ms=_milliseconds(self._monotonic() - score_started))
            except asyncio.CancelledError:
                raise
            except TimeoutError as exc:
                failure = FailureRecord(
                    origin=FailureOrigin.EVALUATOR,
                    stage=f"scorer:{scorer.name}",
                    code="scorer_timeout",
                    message=f"scorer exceeded {config.scorer_timeout_seconds:g}s timeout",
                    retryable=False,
                    exception_type=type(exc).__name__,
                )
            except EvaluatorError as exc:
                error_details = dict(exc.details)
                failure = FailureRecord(
                    origin=FailureOrigin.EVALUATOR,
                    stage=f"scorer:{scorer.name}",
                    code=exc.code,
                    message=str(exc),
                    retryable=False,
                    exception_type=type(exc).__name__,
                )
            except Exception as exc:
                failure = FailureRecord(
                    origin=FailureOrigin.EVALUATOR,
                    stage=f"scorer:{scorer.name}",
                    code="scorer_exception",
                    message=str(exc) or type(exc).__name__,
                    retryable=False,
                    exception_type=type(exc).__name__,
                )
            return ScoreResult(
                scorer_name=scorer.name,
                scorer_version=scorer.version,
                outcome=ScoreOutcome.ERROR,
                score=None,
                passed=None,
                details=error_details,
                failure=failure,
                latency_ms=_milliseconds(self._monotonic() - score_started),
            )

        scores = tuple(await asyncio.gather(*(score_isolated(scorer) for scorer in scorers)))
        score_failures = tuple(score.failure for score in scores if score.failure is not None)
        status = (
            ExampleStatus.PARTIAL
            if any(score.outcome == ScoreOutcome.ERROR for score in scores)
            else ExampleStatus.COMPLETED
        )
        return ExampleResult(
            example_id=example.id,
            status=status,
            input=example.input,
            expected=example.expected,
            metadata=example.metadata,
            response=invocation.response,
            attempts=invocation.attempts,
            scores=scores,
            failures=score_failures,
            latency_ms=_milliseconds(self._monotonic() - started),
        )


def collect_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "evalreliability_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=1,
            ).stdout.strip()
        )
        environment.update({"git_commit": commit, "git_dirty": dirty})
    except (OSError, subprocess.SubprocessError):
        environment.update({"git_commit": None, "git_dirty": None})
    return environment


def _configuration_fingerprint(
    dataset: EvaluationDataset,
    candidate_id: str,
    candidate_config: dict[str, Any],
    scorer_configs: tuple[dict[str, Any], ...],
    config: EvaluationConfig,
) -> str:
    payload = {
        "dataset": {
            "dataset_id": dataset.descriptor.dataset_id,
            "version": dataset.descriptor.version,
            "checksum_sha256": dataset.descriptor.checksum_sha256,
        },
        "candidate_id": candidate_id,
        "candidate": candidate_config,
        "scorers": scorer_configs,
        "evaluation": config.to_dict(),
    }
    try:
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"component configuration is not JSON-safe: {exc}") from exc
    return hashlib.sha256(canonical).hexdigest()


def _run_timestamp(value: datetime) -> str:
    value = value.astimezone(UTC)
    return value.strftime("%Y%m%dT%H%M%S.%fZ")


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _milliseconds(seconds: float) -> float:
    return round(seconds * 1000, 3)
