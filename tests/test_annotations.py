from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalreliability.annotations import (
    analyze_human_referenced_run,
    annotate_run_interactively,
    read_annotations,
)
from evalreliability.artifacts import read_run, write_run
from evalreliability.candidates import FixtureCandidate, RunArtifactCandidate
from evalreliability.dataset import EvaluationDataset
from evalreliability.engine import EvaluationConfig, EvaluationEngine
from evalreliability.models import DatasetDescriptor, EvaluationExample
from evalreliability.reliability import InvocationPolicy
from evalreliability.scorers import ExactMatchScorer, JudgeScorer


def _dataset() -> EvaluationDataset:
    examples = tuple(
        EvaluationExample(
            id=f"a{index}",
            input=f"prompt {index}",
            expected={
                "text": f"answer {index}",
                "rubric": f"Pass only for answer {index}.",
                "human_reference_label": None,
                "human_reference_status": "unlabeled",
            },
            metadata={"slices": {"constraint_type": "first" if index < 2 else "second"}},
        )
        for index in range(4)
    )
    return EvaluationDataset(
        descriptor=DatasetDescriptor(
            dataset_id="annotation-test",
            version="1.0.0",
            checksum_sha256="c" * 64,
            example_count=4,
            source="memory",
        ),
        examples=examples,
    )


def test_analysis_reports_invalid_judge_output_rate_and_raw_evidence() -> None:
    root = Path(__file__).parents[1] / "experiments/judge-calibration-v1"
    report = analyze_human_referenced_run(
        read_run(root / "artifacts/candidate-haiku.run.json"),
        read_annotations(root / "annotations/human-reference.json"),
        judge_run=read_run(root / "artifacts/judge-sonnet.run.json"),
        judge_scorer="sonnet_instruction_judge",
    )

    execution = report["judge_execution"]
    assert execution["invalid_output_count"] == 1
    assert execution["invalid_output_rate"] == 0.0625
    assert execution["invalid_results"][0]["example_id"] == "jcv1-structured-02"
    assert execution["invalid_results"][0]["raw_judge_output"].startswith('{"verdict": "PASS"')
    assert report["judge_error_rationales"]["invalid_result_count"] == 1


@pytest.mark.asyncio
async def test_annotation_workflow_preserves_raw_run_and_analysis_reports_agreement(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 8, 16, 2, 0, tzinfo=UTC)
    candidate_run = await EvaluationEngine(
        now=lambda: fixed_time, environment=lambda: {"test": True}
    ).evaluate(
        _dataset(),
        FixtureCandidate(
            "candidate",
            {
                "a0": "answer 0",
                "a1": "wrong",
                "a2": "wrong",
                "a3": "answer 3",
            },
        ),
        [ExactMatchScorer()],
        EvaluationConfig(candidate_policy=InvocationPolicy(max_attempts=1)),
    )
    raw_path = write_run(candidate_run, tmp_path / "candidate.run.json")
    raw_before = raw_path.read_bytes()
    annotation_path = tmp_path / "human-reference.json"
    answers = iter(
        [
            "PASS",
            "",
            "PASS",
            "semantic requirement met",
            "FAIL",
            "",
            "FAIL",
            "format alone is insufficient",
        ]
    )
    display: list[str] = []

    annotations = annotate_run_interactively(
        candidate_run,
        output_path=annotation_path,
        experiment_id="judge-calibration-v1",
        annotator_id="annotator-1",
        input_fn=lambda _: next(answers),
        output_fn=display.append,
        now=lambda: fixed_time,
    )

    assert raw_path.read_bytes() == raw_before
    assert "answer 0" not in annotation_path.read_text(encoding="utf-8")
    assert any("CANDIDATE RESPONSE" in line for line in display)
    assert len(annotations.annotations) == 4
    loaded = read_annotations(annotation_path)
    assert loaded.to_dict()["label_metadata"] == {
        "reference_type": "single_human_reference",
        "annotator_id": "annotator-1",
        "annotator_count": 1,
        "ground_truth_claim": False,
        "multi_annotator_consensus": False,
        "adjudicated": False,
    }

    replay = RunArtifactCandidate(candidate_run, source="candidate.run.json")
    judge = JudgeScorer(
        FixtureCandidate(
            "judge",
            {
                "a0": '{"score":1.0,"rationale":"pass"}',
                "a1": '{"score":0.0,"rationale":"fail"}',
                "a2": '{"score":0.0,"rationale":"fail"}',
                "a3": '{"score":1.0,"rationale":"pass"}',
            },
        ),
        "Apply expected.rubric.",
        invocation_policy=InvocationPolicy(max_attempts=1),
    )
    judge_run = await EvaluationEngine(
        now=lambda: fixed_time, environment=lambda: {"test": True}
    ).evaluate(
        _dataset(),
        replay,
        [judge],
        EvaluationConfig(candidate_policy=InvocationPolicy(max_attempts=1)),
    )

    report = analyze_human_referenced_run(
        candidate_run,
        loaded,
        judge_run=judge_run,
        judge_scorer="judge",
    )

    assert replay.configuration()["source_run_id"] == candidate_run.metadata.run_id
    assert report["sample_sizes"] == {
        "experiment_examples": 4,
        "candidate_responses": 4,
        "human_labeled": 4,
        "judge_labeled": 4,
        "judge_human_paired": 4,
    }
    assert report["human"]["pass_rate"] == 0.5
    assert report["judge"]["pass_rate"] == 0.5
    assert report["judge_vs_human"]["accuracy"] == 0.5
    assert report["judge_vs_human"]["cohen_kappa"] == 0.0
    assert report["judge_vs_human"]["false_positives"]["example_ids"] == ["a3"]
    assert report["judge_vs_human"]["false_negatives"]["example_ids"] == ["a1"]
    assert report["judge_vs_human"]["disagreements"]["count"] == 2
    assert report["judge_vs_human"]["accuracy_by_reference_label"] == {
        "pass": {"sample_size": 2, "correct": 1, "accuracy": 0.5},
        "fail": {"sample_size": 2, "correct": 1, "accuracy": 0.5},
    }
    assert report["judge_execution"]["requested_examples"] == 4
    assert report["judge_execution"]["valid_verdicts"] == 4
    assert report["judge_execution"]["invalid_output_count"] == 0
    assert report["judge_execution"]["invalid_output_rate"] == 0.0
    assert report["judge_execution"]["raw_judge_outputs_preserved"] == 4
    assert report["judge_execution"]["attempts_total"] == 4
    assert report["judge_execution"]["retried_examples"] == 0
    assert report["judge_execution"]["invalid_results"] == []
    assert report["judge_execution"]["model_ids"] == ["judge"]
    assert report["judge_execution"]["providers"] == ["fixture"]
    assert report["judge_execution"]["usage"]["input_tokens"] == {
        "available_count": 0,
        "total": None,
    }
    assert report["judge_execution"]["latency_ms"]["count"] == 4
    assert report["judge_error_rationales"]["valid_disagreement_count"] == 2
    assert [
        item["example_id"] for item in report["judge_error_rationales"]["valid_disagreements"]
    ] == ["a1", "a3"]
    assert report["judge_error_rationales"]["invalid_result_count"] == 0
    assert len(report["per_slice"]) == 2
    assert all(item["sample_size"] == 2 for item in report["per_slice"])
    deterministic = report["deterministic_scorer_vs_human"]["exact_match"]
    assert deterministic["scorer_sample_size"] == 4
    assert deterministic["sample_size"] == 4
    assert deterministic["accuracy"] == 0.5
    assert report["reference_metadata"]["ground_truth_claim"] is False


@pytest.mark.asyncio
async def test_analysis_without_judge_has_explicit_zero_sample_sizes(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 8, 16, 2, 0, tzinfo=UTC)
    engine = EvaluationEngine(now=lambda: fixed_time, environment=lambda: {"test": True})
    run = await engine.evaluate(
        _dataset(),
        FixtureCandidate("candidate", {f"a{index}": f"answer {index}" for index in range(4)}),
        [ExactMatchScorer()],
        EvaluationConfig(candidate_policy=InvocationPolicy(max_attempts=1)),
    )
    answers = iter(["PASS", "", "Q"])
    annotations = annotate_run_interactively(
        run,
        output_path=tmp_path / "partial.json",
        experiment_id="judge-calibration-v1",
        annotator_id="annotator-1",
        input_fn=lambda _: next(answers),
        output_fn=lambda _: None,
        now=lambda: fixed_time,
    )

    report = analyze_human_referenced_run(run, annotations)

    assert report["sample_sizes"]["human_labeled"] == 1
    assert report["sample_sizes"]["judge_labeled"] == 0
    assert report["judge"]["sample_size"] == 0
    assert report["judge"]["pass_rate"] is None
    assert report["judge_vs_human"]["sample_size"] == 0
    assert report["judge_vs_human"]["accuracy"] is None
    assert report["judge_vs_human"]["cohen_kappa"] is None
    assert report["judge_execution"] is None
    assert report["judge_error_rationales"] is None
