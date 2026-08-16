from __future__ import annotations

import pytest

from evalreliability.analysis import analyze_judge_agreement, summarize_run
from evalreliability.candidates import FixtureCandidate
from evalreliability.dataset import EvaluationDataset
from evalreliability.engine import EvaluationConfig, EvaluationEngine
from evalreliability.models import DatasetDescriptor, EvaluationExample, Usage
from evalreliability.reliability import InvocationPolicy
from evalreliability.scorers import ExactMatchScorer, JudgeScorer


def _dataset() -> EvaluationDataset:
    labels = ("pass", "pass", "fail", "fail")
    examples = tuple(
        EvaluationExample(
            id=f"j{index}",
            input=f"question {index}",
            expected={"text": f"answer {index}", "reference_label": label},
            metadata={"slices": {"domain": "a" if index < 2 else "b"}},
        )
        for index, label in enumerate(labels)
    )
    return EvaluationDataset(
        descriptor=DatasetDescriptor(
            dataset_id="judge-test",
            version="1.0.0",
            checksum_sha256="b" * 64,
            example_count=4,
            source="memory",
            metadata={
                "label_provenance": {
                    "type": "synthetic_specification",
                    "annotator_count": 0,
                }
            },
        ),
        examples=examples,
    )


@pytest.mark.asyncio
async def test_summary_reports_scores_slices_failures_latency_and_usage() -> None:
    candidate = FixtureCandidate(
        "candidate",
        {f"j{index}": f"answer {index}" for index in range(4)},
        usage_by_example={"j0": Usage(input_tokens=2, output_tokens=1, total_tokens=3)},
    )
    run = await EvaluationEngine(environment=lambda: {"test": True}).evaluate(
        _dataset(),
        candidate,
        [ExactMatchScorer()],
        EvaluationConfig(candidate_policy=InvocationPolicy(max_attempts=1)),
    )

    summary = summarize_run(run)

    assert summary["counts"] == {"total": 4, "completed": 4, "partial": 0, "failed": 0}
    assert summary["reliability"]["candidate_success_rate"] == 1.0
    assert summary["scorers"]["exact_match"]["mean_score"] == 1.0
    assert summary["usage"]["total_tokens"] == {"available_count": 1, "total": 3}
    assert summary["latency_ms"]["p95"] is not None
    assert {(item["dimension"], item["value"]) for item in summary["slices"]} == {
        ("domain", "a"),
        ("domain", "b"),
    }


@pytest.mark.asyncio
async def test_judge_agreement_includes_confusion_disagreements_and_provenance() -> None:
    main_candidate = FixtureCandidate(
        "candidate", {f"j{index}": f"answer {index}" for index in range(4)}
    )
    judge_candidate = FixtureCandidate(
        "judge-fixture",
        {
            "j0": '{"score":0.9,"rationale":"match"}',
            "j1": '{"score":0.1,"rationale":"missed evidence"}',
            "j2": '{"score":0.2,"rationale":"incorrect"}',
            "j3": '{"score":0.8,"rationale":"over-credited"}',
        },
    )
    judge = JudgeScorer(
        judge_candidate,
        "Pass only if the answer is correct.",
        invocation_policy=InvocationPolicy(max_attempts=1),
    )
    run = await EvaluationEngine(environment=lambda: {"test": True}).evaluate(
        _dataset(),
        main_candidate,
        [judge],
        EvaluationConfig(candidate_policy=InvocationPolicy(max_attempts=1)),
    )

    analysis = analyze_judge_agreement(run, "judge")

    assert analysis["accuracy"] == 0.5
    assert analysis["cohen_kappa"] == 0.0
    assert analysis["confusion_matrix"] == {
        "pass": {"pass": 1, "fail": 1},
        "fail": {"pass": 1, "fail": 1},
    }
    assert [item["example_id"] for item in analysis["disagreements"]] == ["j1", "j3"]
    assert analysis["ground_truth_claim"] is False
    assert analysis["label_provenance"]["annotator_count"] == 0
    assert len(analysis["slices"]) == 2
