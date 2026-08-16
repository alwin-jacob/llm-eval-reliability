"""Derived run summaries, slice analysis, and judge/reference agreement."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import fmean
from typing import Any

from evalreliability.errors import AnalysisError
from evalreliability.models import EvaluationRun, ExampleResult, ExampleStatus, ScoreOutcome


def summarize_run(run: EvaluationRun) -> dict[str, Any]:
    """Derive aggregate metrics from raw results without mutating the run artifact."""

    total = len(run.examples)
    status_counts = Counter(result.status.value for result in run.examples)
    candidate_successes = sum(result.response is not None for result in run.examples)
    failures = [failure for result in run.examples for failure in result.failures]
    latency_values = [result.latency_ms for result in run.examples]

    summary = {
        "schema_version": "1.0",
        "run_id": run.metadata.run_id,
        "candidate_id": run.metadata.candidate_id,
        "dataset": {
            "dataset_id": run.metadata.dataset.dataset_id,
            "version": run.metadata.dataset.version,
            "checksum_sha256": run.metadata.dataset.checksum_sha256,
            "example_count": run.metadata.dataset.example_count,
        },
        "counts": {
            "total": total,
            "completed": status_counts[ExampleStatus.COMPLETED.value],
            "partial": status_counts[ExampleStatus.PARTIAL.value],
            "failed": status_counts[ExampleStatus.FAILED.value],
        },
        "reliability": {
            "candidate_success_rate": _ratio(candidate_successes, total),
            "evaluation_complete_rate": _ratio(status_counts[ExampleStatus.COMPLETED.value], total),
            "attempts_total": sum(len(result.attempts) for result in run.examples),
            "retried_examples": sum(len(result.attempts) > 1 for result in run.examples),
        },
        "latency_ms": _distribution(latency_values),
        "usage": _usage_summary(run.examples),
        "failures": _failure_summary(failures),
        "scorers": _scorer_summary(run.examples),
        "slices": _slice_summary(run.examples),
    }
    return summary


def analyze_judge_agreement(
    run: EvaluationRun,
    scorer_name: str,
    *,
    reference_key: str = "reference_label",
) -> dict[str, Any]:
    """Compare a judge's binary verdicts with labeled references.

    Reference labels are measurements supplied by the dataset. Their provenance is
    returned verbatim; this function does not promote them or the judge to ground truth.
    """

    records: list[dict[str, Any]] = []
    missing_references: list[str] = []
    missing_judgments: list[str] = []
    for example in run.examples:
        reference = example.expected.get(reference_key)
        if reference is None:
            missing_references.append(example.example_id)
            continue
        if reference not in ("pass", "fail"):
            raise AnalysisError(
                f"example {example.example_id!r} has unsupported {reference_key} {reference!r}; "
                "expected 'pass' or 'fail'"
            )
        score = next((item for item in example.scores if item.scorer_name == scorer_name), None)
        if score is None:
            missing_judgments.append(example.example_id)
            continue
        verdict = score.details.get("verdict")
        if verdict not in ("pass", "fail"):
            missing_judgments.append(example.example_id)
            continue
        records.append(
            {
                "example_id": example.example_id,
                "reference": reference,
                "predicted": verdict,
                "rationale": score.details.get("rationale"),
                "slices": _slices(example),
                "candidate_output": example.response.output if example.response else None,
            }
        )
    labeled_count = len(records) + len(missing_judgments)
    if labeled_count == 0:
        raise AnalysisError(f"run contains no labels at expected.{reference_key}")
    confusion = _confusion(records)
    metrics = _agreement_metrics(records)
    disagreements = [record for record in records if record["reference"] != record["predicted"]]
    descriptor_metadata = run.metadata.dataset.metadata
    return {
        "schema_version": "1.0",
        "run_id": run.metadata.run_id,
        "judge_scorer": scorer_name,
        "reference_key": reference_key,
        "label_provenance": descriptor_metadata.get("label_provenance", "unspecified"),
        "ground_truth_claim": False,
        "caution": (
            "Agreement measures consistency with the supplied references; it does not prove "
            "that either references or judge outputs are ground truth."
        ),
        "counts": {
            "run_examples": len(run.examples),
            "labeled": labeled_count,
            "judged": len(records),
            "missing_reference": len(missing_references),
            "missing_judgment": len(missing_judgments),
            "disagreements": len(disagreements),
        },
        "coverage": _ratio(len(records), labeled_count),
        "accuracy": metrics["accuracy"],
        "cohen_kappa": metrics["cohen_kappa"],
        "confusion_matrix": confusion,
        "missing_reference_ids": missing_references,
        "missing_judgment_ids": missing_judgments,
        "disagreements": disagreements,
        "slices": _judge_slice_analysis(records),
    }


def _scorer_summary(examples: Iterable[ExampleResult]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    total = 0
    for example in examples:
        total += 1
        for score in example.scores:
            grouped[score.scorer_name].append(score)
    result: dict[str, dict[str, Any]] = {}
    for scorer_name in sorted(grouped):
        scores = grouped[scorer_name]
        measured = [item for item in scores if item.score is not None]
        passed = sum(item.outcome == ScoreOutcome.PASS for item in scores)
        result[scorer_name] = {
            "mean_score": _mean([float(item.score) for item in measured]),
            "pass_rate": _ratio(passed, len(measured)),
            "coverage": _ratio(len(measured), total),
            "counts": {
                "pass": passed,
                "fail": sum(item.outcome == ScoreOutcome.FAIL for item in scores),
                "skipped": sum(item.outcome == ScoreOutcome.SKIPPED for item in scores),
                "error": sum(item.outcome == ScoreOutcome.ERROR for item in scores),
                "measured": len(measured),
            },
            "latency_ms": _distribution([item.latency_ms for item in scores]),
        }
    return result


def _slice_summary(examples: Iterable[ExampleResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ExampleResult]] = defaultdict(list)
    for example in examples:
        for dimension, value in _slices(example).items():
            grouped[(dimension, value)].append(example)
    output = []
    for (dimension, value), members in sorted(grouped.items()):
        status_counts = Counter(member.status.value for member in members)
        output.append(
            {
                "dimension": dimension,
                "value": value,
                "count": len(members),
                "counts": {
                    "completed": status_counts[ExampleStatus.COMPLETED.value],
                    "partial": status_counts[ExampleStatus.PARTIAL.value],
                    "failed": status_counts[ExampleStatus.FAILED.value],
                },
                "scorers": _scorer_summary(members),
            }
        )
    return output


def _usage_summary(examples: Iterable[ExampleResult]) -> dict[str, Any]:
    responses = [example.response for example in examples if example.response is not None]
    fields = ("input_tokens", "output_tokens", "total_tokens", "cost_usd")
    output: dict[str, Any] = {"responses": len(responses)}
    for field in fields:
        values = [getattr(response.usage, field) for response in responses]
        available = [value for value in values if value is not None]
        output[field] = {
            "available_count": len(available),
            "total": round(sum(available), 8) if available else None,
        }
    return output


def _failure_summary(failures: Iterable[Any]) -> dict[str, Any]:
    failures = list(failures)
    return {
        "total": len(failures),
        "by_origin": dict(sorted(Counter(item.origin.value for item in failures).items())),
        "by_code": dict(sorted(Counter(item.code for item in failures).items())),
        "by_stage": dict(sorted(Counter(item.stage for item in failures).items())),
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": round(fmean(ordered), 3),
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
        "max": ordered[-1],
    }


def _nearest_rank(ordered: list[float], quantile: float) -> float:
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _slices(example: ExampleResult) -> dict[str, str]:
    raw = example.metadata.get("slices", {})
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _confusion(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {
        reference: {predicted: 0 for predicted in ("pass", "fail")}
        for reference in ("pass", "fail")
    }
    for record in records:
        matrix[record["reference"]][record["predicted"]] += 1
    return matrix


def _agreement_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    if not records:
        return {"accuracy": None, "cohen_kappa": None}
    count = len(records)
    agreement = sum(item["reference"] == item["predicted"] for item in records) / count
    reference_counts = Counter(item["reference"] for item in records)
    predicted_counts = Counter(item["predicted"] for item in records)
    expected = sum(
        (reference_counts[label] / count) * (predicted_counts[label] / count)
        for label in ("pass", "fail")
    )
    kappa = None if expected == 1 else (agreement - expected) / (1 - expected)
    return {
        "accuracy": round(agreement, 6),
        "cohen_kappa": None if kappa is None else round(kappa, 6),
    }


def _judge_slice_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for dimension, value in record["slices"].items():
            grouped[(dimension, value)].append(record)
    output = []
    for (dimension, value), members in sorted(grouped.items()):
        metrics = _agreement_metrics(members)
        output.append(
            {
                "dimension": dimension,
                "value": value,
                "count": len(members),
                "accuracy": metrics["accuracy"],
                "cohen_kappa": metrics["cohen_kappa"],
                "disagreements": sum(
                    member["reference"] != member["predicted"] for member in members
                ),
                "confusion_matrix": _confusion(members),
            }
        )
    return output


def _mean(values: list[float]) -> float | None:
    return round(fmean(values), 6) if values else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None
