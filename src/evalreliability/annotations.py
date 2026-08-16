"""Single-annotator reference artifacts and human-referenced agreement analysis."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalreliability.artifacts import write_json
from evalreliability.errors import AnalysisError, ArtifactError, ConfigurationError
from evalreliability.models import EvaluationRun

REFERENCE_TYPE = "single_human_reference"


@dataclass(frozen=True)
class HumanAnnotation:
    example_id: str
    label: str
    note: str | None
    annotated_at: str

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("annotation example_id cannot be empty")
        if self.label not in ("pass", "fail"):
            raise ValueError("annotation label must be 'pass' or 'fail'")
        if self.note is not None and not isinstance(self.note, str):
            raise TypeError("annotation note must be a string or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "label": self.label,
            "note": self.note,
            "annotated_at": self.annotated_at,
        }


@dataclass(frozen=True)
class HumanAnnotationSet:
    schema_version: str
    experiment_id: str
    source_run_id: str
    dataset_id: str
    dataset_version: str
    dataset_checksum_sha256: str
    annotator_id: str
    created_at: str
    updated_at: str
    annotations: tuple[HumanAnnotation, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("annotation schema_version must be '1.0'")
        for name in (
            "experiment_id",
            "source_run_id",
            "dataset_id",
            "dataset_version",
            "dataset_checksum_sha256",
            "annotator_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"annotation {name} cannot be empty")
        ids = [item.example_id for item in self.annotations]
        if len(ids) != len(set(ids)):
            raise ValueError("annotation example IDs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "source_run_id": self.source_run_id,
            "dataset": {
                "dataset_id": self.dataset_id,
                "version": self.dataset_version,
                "checksum_sha256": self.dataset_checksum_sha256,
            },
            "label_metadata": {
                "reference_type": REFERENCE_TYPE,
                "annotator_id": self.annotator_id,
                "annotator_count": 1,
                "ground_truth_claim": False,
                "multi_annotator_consensus": False,
                "adjudicated": False,
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "counts": {
                "annotated": len(self.annotations),
                "pass": sum(item.label == "pass" for item in self.annotations),
                "fail": sum(item.label == "fail" for item in self.annotations),
            },
            "annotations": [item.to_dict() for item in self.annotations],
        }


def new_annotation_set(
    run: EvaluationRun,
    *,
    experiment_id: str,
    annotator_id: str,
    now: Callable[[], datetime] | None = None,
) -> HumanAnnotationSet:
    if not experiment_id:
        raise ConfigurationError("experiment_id cannot be empty")
    if not annotator_id:
        raise ConfigurationError("annotator_id cannot be empty")
    timestamp = _timestamp((now or (lambda: datetime.now(UTC)))())
    return HumanAnnotationSet(
        schema_version="1.0",
        experiment_id=experiment_id,
        source_run_id=run.metadata.run_id,
        dataset_id=run.metadata.dataset.dataset_id,
        dataset_version=run.metadata.dataset.version,
        dataset_checksum_sha256=run.metadata.dataset.checksum_sha256,
        annotator_id=annotator_id,
        created_at=timestamp,
        updated_at=timestamp,
    )


def write_annotations(annotations: HumanAnnotationSet, path: str | Path) -> Path:
    return write_json(annotations.to_dict(), path)


def read_annotations(path: str | Path) -> HumanAnnotationSet:
    source = Path(path)
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("annotation root must be an object")
        dataset = _required_mapping(payload, "dataset")
        label_metadata = _required_mapping(payload, "label_metadata")
        if label_metadata.get("reference_type") != REFERENCE_TYPE:
            raise ValueError(f"reference_type must be {REFERENCE_TYPE!r}")
        if label_metadata.get("annotator_count") != 1:
            raise ValueError("annotation artifact must identify exactly one annotator")
        if label_metadata.get("ground_truth_claim") is not False:
            raise ValueError("annotation artifact cannot claim ground truth")
        if label_metadata.get("multi_annotator_consensus") is not False:
            raise ValueError("annotation artifact cannot claim multi-annotator consensus")
        raw_annotations = payload.get("annotations")
        if not isinstance(raw_annotations, list):
            raise TypeError("annotations must be an array")
        records = tuple(_annotation_from_dict(item) for item in raw_annotations)
        result = HumanAnnotationSet(
            schema_version=_required_string(payload, "schema_version"),
            experiment_id=_required_string(payload, "experiment_id"),
            source_run_id=_required_string(payload, "source_run_id"),
            dataset_id=_required_string(dataset, "dataset_id"),
            dataset_version=_required_string(dataset, "version"),
            dataset_checksum_sha256=_required_string(dataset, "checksum_sha256"),
            annotator_id=_required_string(label_metadata, "annotator_id"),
            created_at=_required_string(payload, "created_at"),
            updated_at=_required_string(payload, "updated_at"),
            annotations=records,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"could not read annotation artifact {source}: {exc}") from exc
    return result


def annotate_run_interactively(
    run: EvaluationRun,
    *,
    output_path: str | Path,
    experiment_id: str,
    annotator_id: str,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
    now: Callable[[], datetime] | None = None,
) -> HumanAnnotationSet:
    """Collect PASS/FAIL labels, atomically checkpointing after each example."""

    input_fn = input_fn or input
    output_fn = output_fn or print
    destination = Path(output_path)
    if destination.exists():
        annotations = read_annotations(destination)
    else:
        annotations = new_annotation_set(
            run,
            experiment_id=experiment_id,
            annotator_id=annotator_id,
            now=now,
        )
        write_annotations(annotations, destination)
    _validate_annotation_source(annotations, run, experiment_id, annotator_id)
    annotated_ids = {item.example_id for item in annotations.annotations}
    now_fn = now or (lambda: datetime.now(UTC))

    for index, example in enumerate(run.examples, start=1):
        if example.example_id in annotated_ids:
            continue
        rubric = example.expected.get("rubric")
        if not isinstance(rubric, str) or not rubric:
            raise ConfigurationError(
                f"example {example.example_id!r} has no non-empty expected.rubric"
            )
        response = example.response.output if example.response is not None else "<NO RESPONSE>"
        output_fn(f"\n[{index}/{len(run.examples)}] {example.example_id}")
        output_fn(f"PROMPT\n{example.input}")
        output_fn(f"\nCANDIDATE RESPONSE\n{response}")
        output_fn(f"\nRUBRIC\n{rubric}")
        label = _prompt_label(input_fn)
        if label is None:
            break
        note = input_fn("Short note (optional): ").strip() or None
        record = HumanAnnotation(
            example_id=example.example_id,
            label=label,
            note=note,
            annotated_at=_timestamp(now_fn()),
        )
        annotations = replace(
            annotations,
            updated_at=record.annotated_at,
            annotations=(*annotations.annotations, record),
        )
        write_annotations(annotations, destination)
        annotated_ids.add(example.example_id)

    output_fn(
        f"Saved {len(annotations.annotations)}/{len(run.examples)} annotations to {destination}"
    )
    return annotations


def analyze_human_referenced_run(
    candidate_run: EvaluationRun,
    annotations: HumanAnnotationSet,
    *,
    judge_run: EvaluationRun | None = None,
    judge_scorer: str | None = None,
    slice_dimension: str = "constraint_type",
) -> dict[str, Any]:
    """Compare human, judge, and deterministic verdicts without ground-truth claims."""

    _validate_annotation_source(
        annotations,
        candidate_run,
        annotations.experiment_id,
        annotations.annotator_id,
    )
    candidate_by_id = {item.example_id: item for item in candidate_run.examples}
    human = {item.example_id: item.label for item in annotations.annotations}
    unknown_annotations = sorted(set(human) - set(candidate_by_id))
    if unknown_annotations:
        joined = ", ".join(unknown_annotations)
        raise AnalysisError(f"annotations reference examples absent from candidate run: {joined}")

    judge: dict[str, str] = {}
    if judge_run is not None:
        if not judge_scorer:
            raise AnalysisError("judge_scorer is required when judge_run is provided")
        _validate_aligned_runs(candidate_run, judge_run)
        judge = _score_verdicts(judge_run, judge_scorer, require_detail_verdict=True)
    elif judge_scorer is not None:
        raise AnalysisError("judge_run is required when judge_scorer is provided")

    deterministic_names = _deterministic_scorer_names(candidate_run)
    deterministic = {
        name: _score_verdicts(candidate_run, name, require_detail_verdict=False)
        for name in deterministic_names
    }
    human_summary = _label_summary(human)
    judge_summary = _label_summary(judge)
    judge_agreement = _agreement_report(human, judge)

    slices: list[dict[str, Any]] = []
    slice_values = sorted(
        {
            str(example.metadata.get("slices", {}).get(slice_dimension))
            for example in candidate_run.examples
            if isinstance(example.metadata.get("slices"), dict)
            and slice_dimension in example.metadata["slices"]
        }
    )
    for value in slice_values:
        members = {
            example.example_id
            for example in candidate_run.examples
            if isinstance(example.metadata.get("slices"), dict)
            and str(example.metadata["slices"].get(slice_dimension)) == value
        }
        slice_human = _restrict(human, members)
        slice_judge = _restrict(judge, members)
        slices.append(
            {
                "dimension": slice_dimension,
                "value": value,
                "sample_size": len(members),
                "human": _label_summary(slice_human),
                "judge": _label_summary(slice_judge),
                "judge_vs_human": _agreement_report(slice_human, slice_judge),
            }
        )

    return {
        "schema_version": "1.0",
        "experiment_id": annotations.experiment_id,
        "candidate_run_id": candidate_run.metadata.run_id,
        "judge_run_id": judge_run.metadata.run_id if judge_run is not None else None,
        "judge_scorer": judge_scorer,
        "reference_metadata": {
            "reference_type": REFERENCE_TYPE,
            "annotator_id": annotations.annotator_id,
            "annotator_count": 1,
            "ground_truth_claim": False,
            "multi_annotator_consensus": False,
        },
        "sample_sizes": {
            "experiment_examples": len(candidate_run.examples),
            "candidate_responses": sum(
                example.response is not None for example in candidate_run.examples
            ),
            "human_labeled": len(human),
            "judge_labeled": len(judge),
            "judge_human_paired": judge_agreement["sample_size"],
        },
        "human": human_summary,
        "judge": judge_summary,
        "judge_vs_human": judge_agreement,
        "per_slice": slices,
        "deterministic_scorer_vs_human": {
            name: {
                "scorer_sample_size": len(verdicts),
                **_agreement_report(human, verdicts),
            }
            for name, verdicts in deterministic.items()
        },
        "caution": (
            "The human labels are one annotator's references, not ground truth or "
            "multi-annotator consensus."
        ),
    }


def _prompt_label(input_fn: Callable[[str], str]) -> str | None:
    while True:
        value = input_fn("Label [PASS/FAIL, Q to stop]: ").strip().casefold()
        if value in ("pass", "p"):
            return "pass"
        if value in ("fail", "f"):
            return "fail"
        if value in ("q", "quit"):
            return None


def _validate_annotation_source(
    annotations: HumanAnnotationSet,
    run: EvaluationRun,
    experiment_id: str,
    annotator_id: str,
) -> None:
    if annotations.experiment_id != experiment_id:
        raise ConfigurationError("annotation experiment_id does not match requested experiment")
    if annotations.annotator_id != annotator_id:
        raise ConfigurationError("annotation artifact belongs to a different annotator")
    expected = (
        run.metadata.run_id,
        run.metadata.dataset.dataset_id,
        run.metadata.dataset.version,
        run.metadata.dataset.checksum_sha256,
    )
    actual = (
        annotations.source_run_id,
        annotations.dataset_id,
        annotations.dataset_version,
        annotations.dataset_checksum_sha256,
    )
    if actual != expected:
        raise ConfigurationError("annotation artifact does not match the source run and dataset")


def _validate_aligned_runs(candidate_run: EvaluationRun, judge_run: EvaluationRun) -> None:
    if candidate_run.metadata.dataset.checksum_sha256 != judge_run.metadata.dataset.checksum_sha256:
        raise AnalysisError("candidate and judge runs use different dataset checksums")
    candidate_ids = [item.example_id for item in candidate_run.examples]
    judge_ids = [item.example_id for item in judge_run.examples]
    if candidate_ids != judge_ids:
        raise AnalysisError("candidate and judge runs have different example IDs or order")


def _score_verdicts(
    run: EvaluationRun, scorer_name: str, *, require_detail_verdict: bool
) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    for example in run.examples:
        score = next((item for item in example.scores if item.scorer_name == scorer_name), None)
        if score is None or score.passed is None:
            continue
        detail_verdict = score.details.get("verdict")
        if require_detail_verdict and detail_verdict not in ("pass", "fail"):
            continue
        verdicts[example.example_id] = (
            str(detail_verdict) if require_detail_verdict else ("pass" if score.passed else "fail")
        )
    return verdicts


def _deterministic_scorer_names(run: EvaluationRun) -> list[str]:
    return sorted(
        str(config["name"])
        for config in run.metadata.scorer_configs
        if config.get("type") != "judge" and isinstance(config.get("name"), str)
    )


def _label_summary(labels: dict[str, str]) -> dict[str, Any]:
    counts = Counter(labels.values())
    sample_size = len(labels)
    return {
        "sample_size": sample_size,
        "pass_count": counts["pass"],
        "fail_count": counts["fail"],
        "pass_rate": round(counts["pass"] / sample_size, 6) if sample_size else None,
    }


def _agreement_report(reference: dict[str, str], predicted: dict[str, str]) -> dict[str, Any]:
    paired_ids = sorted(set(reference) & set(predicted))
    confusion = {
        "human_pass": {"predicted_pass": 0, "predicted_fail": 0},
        "human_fail": {"predicted_pass": 0, "predicted_fail": 0},
    }
    disagreements: list[dict[str, str]] = []
    false_positive_ids: list[str] = []
    false_negative_ids: list[str] = []
    for example_id in paired_ids:
        human_label = reference[example_id]
        predicted_label = predicted[example_id]
        confusion[f"human_{human_label}"][f"predicted_{predicted_label}"] += 1
        if human_label != predicted_label:
            disagreements.append(
                {
                    "example_id": example_id,
                    "human_label": human_label,
                    "predicted_label": predicted_label,
                }
            )
        if human_label == "fail" and predicted_label == "pass":
            false_positive_ids.append(example_id)
        if human_label == "pass" and predicted_label == "fail":
            false_negative_ids.append(example_id)

    sample_size = len(paired_ids)
    agreement_count = sum(reference[item] == predicted[item] for item in paired_ids)
    accuracy = round(agreement_count / sample_size, 6) if sample_size else None
    kappa = _cohen_kappa(reference, predicted, paired_ids)
    return {
        "sample_size": sample_size,
        "accuracy": accuracy,
        "cohen_kappa": kappa,
        "confusion_matrix": confusion,
        "false_positives": {
            "sample_size": sample_size,
            "count": len(false_positive_ids),
            "example_ids": false_positive_ids,
        },
        "false_negatives": {
            "sample_size": sample_size,
            "count": len(false_negative_ids),
            "example_ids": false_negative_ids,
        },
        "disagreements": {
            "sample_size": sample_size,
            "count": len(disagreements),
            "items": disagreements,
        },
    }


def _cohen_kappa(
    reference: dict[str, str], predicted: dict[str, str], paired_ids: list[str]
) -> float | None:
    if not paired_ids:
        return None
    sample_size = len(paired_ids)
    observed = sum(reference[item] == predicted[item] for item in paired_ids) / sample_size
    reference_counts = Counter(reference[item] for item in paired_ids)
    predicted_counts = Counter(predicted[item] for item in paired_ids)
    expected = sum(
        (reference_counts[label] / sample_size) * (predicted_counts[label] / sample_size)
        for label in ("pass", "fail")
    )
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 6)


def _restrict(labels: dict[str, str], members: set[str]) -> dict[str, str]:
    return {example_id: label for example_id, label in labels.items() if example_id in members}


def _annotation_from_dict(data: Any) -> HumanAnnotation:
    if not isinstance(data, dict):
        raise TypeError("annotation entries must be objects")
    note = data.get("note")
    if note is not None and not isinstance(note, str):
        raise TypeError("annotation note must be a string or null")
    return HumanAnnotation(
        example_id=_required_string(data, "example_id"),
        label=_required_string(data, "label"),
        note=note,
        annotated_at=_required_string(data, "annotated_at"),
    )


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "HumanAnnotation",
    "HumanAnnotationSet",
    "analyze_human_referenced_run",
    "annotate_run_interactively",
    "new_annotation_set",
    "read_annotations",
    "write_annotations",
]
