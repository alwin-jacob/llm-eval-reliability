from __future__ import annotations

import pytest

from evalreliability.candidates import FixtureCandidate
from evalreliability.dataset import EvaluationDataset
from evalreliability.engine import EvaluationConfig, EvaluationEngine
from evalreliability.errors import ConfigurationError
from evalreliability.models import DatasetDescriptor, EvaluationExample
from evalreliability.regression import (
    MetricPolicy,
    RegressionPolicy,
    SlicePolicy,
    compare_runs,
)
from evalreliability.reliability import InvocationPolicy
from evalreliability.scorers import ExactMatchScorer


def _dataset() -> EvaluationDataset:
    examples = tuple(
        EvaluationExample(
            id=f"e{index}",
            input=f"question {index}",
            expected={"text": f"correct {index}"},
            metadata={"slices": {"difficulty": "hard" if index >= 2 else "easy"}},
        )
        for index in range(4)
    )
    return EvaluationDataset(
        DatasetDescriptor("regression", "1.0.0", "c" * 64, 4, "memory"), examples
    )


async def _run(
    identifier: str,
    responses: dict[str, str],
    scorer: ExactMatchScorer | None = None,
):
    return await EvaluationEngine(environment=lambda: {"test": True}).evaluate(
        _dataset(),
        FixtureCandidate(identifier, responses),
        [scorer or ExactMatchScorer()],
        EvaluationConfig(candidate_policy=InvocationPolicy(max_attempts=1)),
    )


@pytest.mark.asyncio
async def test_regression_policy_passes_with_allowed_measured_drop() -> None:
    baseline = await _run("baseline", {f"e{i}": f"correct {i}" for i in range(4)})
    candidate = await _run(
        "candidate",
        {"e0": "correct 0", "e1": "correct 1", "e2": "wrong", "e3": "correct 3"},
    )
    policy = RegressionPolicy(
        max_per_example_regressions=1,
        metrics=(MetricPolicy("exact_match", min_candidate=0.75, max_drop=0.25),),
        slices=(
            SlicePolicy("difficulty", "hard", "exact_match", min_examples=2, min_candidate=0.5),
        ),
    )

    report = compare_runs(baseline, candidate, policy)

    assert report["passed"] is True
    assert report["metric_comparisons"][0]["delta"] == -0.25
    assert len(report["per_example_regressions"]) == 1
    assert report["violations"] == []


@pytest.mark.asyncio
async def test_regression_policy_reports_each_actionable_violation() -> None:
    baseline = await _run("baseline", {f"e{i}": f"correct {i}" for i in range(4)})
    candidate = await _run("candidate", {f"e{i}": "wrong" for i in range(4)})
    policy = RegressionPolicy(
        max_per_example_regressions=0,
        metrics=(MetricPolicy("exact_match", min_candidate=0.8, max_drop=0.1),),
    )

    report = compare_runs(baseline, candidate, policy)

    assert report["passed"] is False
    codes = {item["code"] for item in report["violations"]}
    assert codes == {
        "per_example_regressions",
        "metric_below_minimum",
        "metric_drop_exceeded",
    }
    assert len(report["per_example_regressions"]) == 4


def test_regression_policy_rejects_unknown_or_vacuous_configuration() -> None:
    with pytest.raises(ConfigurationError, match="unknown regression policy fields"):
        RegressionPolicy.from_dict({"magic_threshold": 0.5})
    with pytest.raises(ConfigurationError, match="must configure"):
        RegressionPolicy.from_dict({"metrics": [{"scorer": "exact_match"}]})
    with pytest.raises(ConfigurationError, match="require_same_dataset must be a boolean"):
        RegressionPolicy.from_dict({"require_same_dataset": "false"})
    with pytest.raises(ConfigurationError, match="min_candidate must be numeric"):
        RegressionPolicy.from_dict({"metrics": [{"scorer": "exact_match", "min_candidate": True}]})


@pytest.mark.asyncio
async def test_regression_rejects_changed_scorer_configuration() -> None:
    responses = {f"e{i}": f"correct {i}" for i in range(4)}
    baseline = await _run("baseline", responses)
    candidate = await _run("candidate", responses, ExactMatchScorer(case_sensitive=True))

    report = compare_runs(baseline, candidate, RegressionPolicy())

    assert report["passed"] is False
    assert report["alignment"]["same_scorers"] is False
    assert {item["code"] for item in report["violations"]} == {"scorer_config_mismatch"}


@pytest.mark.asyncio
async def test_regression_policy_can_gate_scorer_coverage() -> None:
    baseline = await _run("baseline", {f"e{i}": f"correct {i}" for i in range(4)})
    candidate = await _run("candidate", {f"e{i}": f"correct {i}" for i in range(3)})
    policy = RegressionPolicy(
        max_new_failed_examples=1,
        max_new_operational_failures=1,
        max_failure_rate_increase=0.25,
        metrics=(MetricPolicy("exact_match", min_coverage=1.0),),
    )

    report = compare_runs(baseline, candidate, policy)

    assert report["passed"] is False
    assert report["metric_comparisons"][0]["candidate_coverage"] == 0.75
    assert {item["code"] for item in report["violations"]} == {"coverage_below_minimum"}
