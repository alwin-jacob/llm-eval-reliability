from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalreliability.artifacts import read_run, write_run
from evalreliability.candidates import Candidate, FixtureCandidate
from evalreliability.dataset import EvaluationDataset
from evalreliability.engine import EvaluationConfig, EvaluationEngine
from evalreliability.errors import CandidateError
from evalreliability.models import (
    CandidateRequest,
    CandidateResponse,
    DatasetDescriptor,
    EvaluationExample,
    ExampleStatus,
    FailureOrigin,
    ScoreOutcome,
    ScoreResult,
)
from evalreliability.reliability import InvocationPolicy
from evalreliability.scorers import ExactMatchScorer, Scorer


def _dataset(count: int = 3) -> EvaluationDataset:
    examples = tuple(
        EvaluationExample(
            id=f"e{index}",
            input=f"prompt {index}",
            expected={"text": f"answer {index}"},
            metadata={"slices": {"parity": "even" if index % 2 == 0 else "odd"}},
        )
        for index in range(count)
    )
    return EvaluationDataset(
        descriptor=DatasetDescriptor(
            dataset_id="engine-test",
            version="1.0.0",
            checksum_sha256="a" * 64,
            example_count=count,
            source="memory",
        ),
        examples=examples,
    )


class CountingCandidate(Candidate):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    @property
    def identifier(self) -> str:
        return "counting"

    def configuration(self) -> dict[str, str]:
        return {"type": "test"}

    async def generate(self, request: CandidateRequest) -> CandidateResponse:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        index = request.example_id.removeprefix("e")
        return CandidateResponse(output=f"answer {index}")


class ExplodingScorer(Scorer):
    name = "exploding"
    version = "1"

    def configuration(self) -> dict[str, str]:
        return {"type": "test", "name": self.name}

    async def score(self, example: EvaluationExample, response: CandidateResponse) -> ScoreResult:
        raise RuntimeError("scorer defect")


class SlowScorer(Scorer):
    name = "slow"
    version = "1"

    def configuration(self) -> dict[str, str]:
        return {"type": "test", "name": self.name}

    async def score(self, example: EvaluationExample, response: CandidateResponse) -> ScoreResult:
        await asyncio.sleep(1)
        return ScoreResult.passed_result(self.name, self.version)


class SelectiveCandidate(Candidate):
    @property
    def identifier(self) -> str:
        return "selective"

    def configuration(self) -> dict[str, str]:
        return {"type": "test"}

    async def generate(self, request: CandidateRequest) -> CandidateResponse:
        if request.example_id == "e1":
            raise CandidateError(
                "model refused",
                code="model_refusal",
                retryable=False,
                origin=FailureOrigin.MODEL,
            )
        return CandidateResponse(output=f"answer {request.example_id.removeprefix('e')}")


@pytest.mark.asyncio
async def test_engine_bounds_concurrency_and_preserves_dataset_order() -> None:
    candidate = CountingCandidate()
    run = await EvaluationEngine(environment=lambda: {"test": True}).evaluate(
        _dataset(4),
        candidate,
        [ExactMatchScorer()],
        EvaluationConfig(
            concurrency=2,
            candidate_policy=InvocationPolicy(max_attempts=1),
        ),
    )

    assert candidate.max_active == 2
    assert [result.example_id for result in run.examples] == ["e0", "e1", "e2", "e3"]
    assert all(result.status == ExampleStatus.COMPLETED for result in run.examples)
    assert all(result.scores[0].outcome == ScoreOutcome.PASS for result in run.examples)


@pytest.mark.asyncio
async def test_engine_isolates_candidate_and_evaluator_failures() -> None:
    run = await EvaluationEngine(environment=lambda: {"test": True}).evaluate(
        _dataset(),
        SelectiveCandidate(),
        [ExactMatchScorer(), ExplodingScorer()],
        EvaluationConfig(candidate_policy=InvocationPolicy(max_attempts=1)),
    )

    by_id = {result.example_id: result for result in run.examples}
    assert by_id["e1"].status == ExampleStatus.FAILED
    assert by_id["e1"].failures[0].origin == FailureOrigin.MODEL
    assert by_id["e0"].status == ExampleStatus.PARTIAL
    assert by_id["e0"].scores[0].outcome == ScoreOutcome.PASS
    assert by_id["e0"].scores[1].outcome == ScoreOutcome.ERROR
    assert by_id["e0"].scores[1].failure is not None
    assert by_id["e0"].scores[1].failure.origin == FailureOrigin.EVALUATOR


@pytest.mark.asyncio
async def test_engine_bounds_scorer_timeout_as_partial_evaluator_failure() -> None:
    run = await EvaluationEngine(environment=lambda: {"test": True}).evaluate(
        _dataset(1),
        FixtureCandidate("fixture", {"e0": "answer 0"}),
        [ExactMatchScorer(), SlowScorer()],
        EvaluationConfig(
            scorer_timeout_seconds=0.01,
            candidate_policy=InvocationPolicy(max_attempts=1),
        ),
    )

    result = run.examples[0]
    assert result.status == ExampleStatus.PARTIAL
    assert result.scores[0].outcome == ScoreOutcome.PASS
    assert result.scores[1].outcome == ScoreOutcome.ERROR
    assert result.scores[1].failure is not None
    assert result.scores[1].failure.code == "scorer_timeout"


@pytest.mark.asyncio
async def test_run_fingerprint_is_deterministic_and_artifact_round_trips(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
    engine = EvaluationEngine(now=lambda: fixed_time, environment=lambda: {"test": True})
    candidate = FixtureCandidate(
        "fixture-v1", {"e0": "answer 0", "e1": "answer 1", "e2": "answer 2"}
    )
    config = EvaluationConfig(candidate_policy=InvocationPolicy(max_attempts=1))

    first = await engine.evaluate(_dataset(), candidate, [ExactMatchScorer()], config)
    second = await engine.evaluate(_dataset(), candidate, [ExactMatchScorer()], config)
    path = write_run(first, tmp_path / "run.json")
    loaded = read_run(path)

    assert first.metadata.config_fingerprint == second.metadata.config_fingerprint
    assert first.metadata.run_id == second.metadata.run_id
    assert loaded.to_dict() == first.to_dict()


@pytest.mark.asyncio
async def test_engine_snapshots_component_configuration_once() -> None:
    class CountingConfigurationCandidate(FixtureCandidate):
        configuration_calls = 0

        def configuration(self) -> dict[str, object]:
            self.configuration_calls += 1
            return {"type": "test", "configuration_call": self.configuration_calls}

    candidate = CountingConfigurationCandidate("counting-config", {"e0": "answer 0"})

    run = await EvaluationEngine(environment=lambda: {"test": True}).evaluate(
        _dataset(1),
        candidate,
        [ExactMatchScorer()],
        EvaluationConfig(candidate_policy=InvocationPolicy(max_attempts=1)),
    )

    assert candidate.configuration_calls == 1
    assert run.metadata.candidate_config["configuration_call"] == 1
