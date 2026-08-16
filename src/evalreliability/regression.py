"""Baseline/candidate comparison and configurable quality gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalreliability.analysis import summarize_run
from evalreliability.errors import ConfigurationError
from evalreliability.models import EvaluationRun, ExampleStatus


@dataclass(frozen=True)
class MetricPolicy:
    scorer: str
    min_candidate: float | None = None
    max_drop: float | None = None
    min_coverage: float | None = None
    max_coverage_drop: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scorer, str) or not self.scorer:
            raise ConfigurationError("metric policy scorer cannot be empty")
        _validate_unit_interval(self.min_candidate, "min_candidate")
        _validate_unit_interval(self.max_drop, "max_drop")
        _validate_unit_interval(self.min_coverage, "min_coverage")
        _validate_unit_interval(self.max_coverage_drop, "max_coverage_drop")
        if all(
            value is None
            for value in (
                self.min_candidate,
                self.max_drop,
                self.min_coverage,
                self.max_coverage_drop,
            )
        ):
            raise ConfigurationError("metric policy must configure a score or coverage gate")


@dataclass(frozen=True)
class SlicePolicy:
    dimension: str
    value: str
    scorer: str
    min_examples: int = 1
    min_candidate: float | None = None
    max_drop: float | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(item, str) and item for item in (self.dimension, self.value, self.scorer)
        ):
            raise ConfigurationError("slice dimension, value, and scorer cannot be empty")
        if not isinstance(self.min_examples, int) or isinstance(self.min_examples, bool):
            raise ConfigurationError("slice min_examples must be an integer")
        if self.min_examples < 1:
            raise ConfigurationError("slice min_examples must be at least 1")
        _validate_unit_interval(self.min_candidate, "min_candidate")
        _validate_unit_interval(self.max_drop, "max_drop")
        if self.min_candidate is None and self.max_drop is None:
            raise ConfigurationError("slice policy must set min_candidate or max_drop")


@dataclass(frozen=True)
class RegressionPolicy:
    require_same_dataset: bool = True
    require_same_examples: bool = True
    require_same_scorers: bool = True
    max_new_failed_examples: int = 0
    max_new_operational_failures: int = 0
    max_failure_rate_increase: float = 0.0
    max_per_example_regressions: int = 0
    per_example_min_drop: float = 0.0
    metrics: tuple[MetricPolicy, ...] = ()
    slices: tuple[SlicePolicy, ...] = ()

    def __post_init__(self) -> None:
        for name in ("require_same_dataset", "require_same_examples", "require_same_scorers"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(f"{name} must be a boolean")
        for name in (
            "max_new_failed_examples",
            "max_new_operational_failures",
            "max_per_example_regressions",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigurationError(f"{name} must be an integer")
            if value < 0:
                raise ConfigurationError(f"{name} cannot be negative")
        _validate_unit_interval(self.max_failure_rate_increase, "max_failure_rate_increase")
        _validate_unit_interval(self.per_example_min_drop, "per_example_min_drop")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegressionPolicy:
        allowed = {
            "require_same_dataset",
            "require_same_examples",
            "require_same_scorers",
            "max_new_failed_examples",
            "max_new_operational_failures",
            "max_failure_rate_increase",
            "max_per_example_regressions",
            "per_example_min_drop",
            "metrics",
            "slices",
        }
        _reject_unknown(data, allowed, "regression policy")
        metric_items = data.get("metrics", [])
        slice_items = data.get("slices", [])
        if not isinstance(metric_items, list) or not isinstance(slice_items, list):
            raise ConfigurationError("policy metrics and slices must be arrays")
        metrics = tuple(_parse_metric_policy(item) for item in metric_items)
        slices = tuple(_parse_slice_policy(item) for item in slice_items)
        scalar = {key: value for key, value in data.items() if key not in ("metrics", "slices")}
        try:
            return cls(metrics=metrics, slices=slices, **scalar)
        except TypeError as exc:
            raise ConfigurationError(f"invalid regression policy: {exc}") from exc


def compare_runs(
    baseline: EvaluationRun,
    candidate: EvaluationRun,
    policy: RegressionPolicy,
) -> dict[str, Any]:
    baseline_summary = summarize_run(baseline)
    candidate_summary = summarize_run(candidate)
    violations: list[dict[str, Any]] = []
    baseline_by_id = {item.example_id: item for item in baseline.examples}
    candidate_by_id = {item.example_id: item for item in candidate.examples}
    baseline_ids = set(baseline_by_id)
    candidate_ids = set(candidate_by_id)
    common_ids = sorted(baseline_ids & candidate_ids)
    missing_in_candidate = sorted(baseline_ids - candidate_ids)
    added_in_candidate = sorted(candidate_ids - baseline_ids)

    same_dataset = (
        baseline.metadata.dataset.dataset_id == candidate.metadata.dataset.dataset_id
        and baseline.metadata.dataset.version == candidate.metadata.dataset.version
        and baseline.metadata.dataset.checksum_sha256 == candidate.metadata.dataset.checksum_sha256
    )
    baseline_scorers = _scorer_configs(baseline)
    candidate_scorers = _scorer_configs(candidate)
    same_scorers = baseline_scorers == candidate_scorers
    if policy.require_same_dataset and not same_dataset:
        violations.append(
            _violation(
                "dataset_mismatch",
                "baseline and candidate must use the same dataset ID, version, and checksum",
                observed={
                    "baseline": _dataset_identity(baseline),
                    "candidate": _dataset_identity(candidate),
                },
                threshold="identical",
            )
        )
    if policy.require_same_examples and (missing_in_candidate or added_in_candidate):
        violations.append(
            _violation(
                "example_set_mismatch",
                "baseline and candidate example IDs differ",
                observed={
                    "missing_in_candidate": missing_in_candidate,
                    "added": added_in_candidate,
                },
                threshold="identical",
            )
        )
    if policy.require_same_scorers and not same_scorers:
        violations.append(
            _violation(
                "scorer_config_mismatch",
                "baseline and candidate must use identical scorer names and configurations",
                observed={"baseline": baseline_scorers, "candidate": candidate_scorers},
                threshold="identical",
            )
        )

    new_failed = [
        example_id
        for example_id in common_ids
        if baseline_by_id[example_id].status != ExampleStatus.FAILED
        and candidate_by_id[example_id].status == ExampleStatus.FAILED
    ]
    if len(new_failed) > policy.max_new_failed_examples:
        violations.append(
            _violation(
                "new_failed_examples",
                "candidate has too many newly failed examples",
                observed={"count": len(new_failed), "example_ids": new_failed},
                threshold=policy.max_new_failed_examples,
            )
        )

    new_operational = [
        example_id
        for example_id in common_ids
        if baseline_by_id[example_id].status == ExampleStatus.COMPLETED
        and candidate_by_id[example_id].status != ExampleStatus.COMPLETED
    ]
    if len(new_operational) > policy.max_new_operational_failures:
        violations.append(
            _violation(
                "new_operational_failures",
                "candidate has too many new partial or failed evaluations",
                observed={"count": len(new_operational), "example_ids": new_operational},
                threshold=policy.max_new_operational_failures,
            )
        )

    baseline_failure_rate = _failure_rate(baseline)
    candidate_failure_rate = _failure_rate(candidate)
    failure_rate_increase = candidate_failure_rate - baseline_failure_rate
    if failure_rate_increase > policy.max_failure_rate_increase:
        violations.append(
            _violation(
                "failure_rate_increase",
                "candidate failure rate increase exceeds policy",
                observed=round(failure_rate_increase, 6),
                threshold=policy.max_failure_rate_increase,
            )
        )

    per_example_regressions = _per_example_regressions(
        baseline_by_id, candidate_by_id, common_ids, policy.per_example_min_drop
    )
    if len(per_example_regressions) > policy.max_per_example_regressions:
        violations.append(
            _violation(
                "per_example_regressions",
                "candidate has too many per-example score regressions",
                observed={"count": len(per_example_regressions)},
                threshold=policy.max_per_example_regressions,
            )
        )

    metric_comparisons = []
    for metric_policy in policy.metrics:
        comparison = _metric_comparison(baseline_summary, candidate_summary, metric_policy.scorer)
        metric_comparisons.append(comparison)
        violations.extend(_check_metric_policy(comparison, metric_policy))

    slice_comparisons = []
    for slice_policy in policy.slices:
        comparison = _slice_comparison(baseline_summary, candidate_summary, slice_policy)
        slice_comparisons.append(comparison)
        violations.extend(_check_slice_policy(comparison, slice_policy))

    return {
        "schema_version": "1.0",
        "baseline_run_id": baseline.metadata.run_id,
        "candidate_run_id": candidate.metadata.run_id,
        "passed": not violations,
        "policy": _policy_dict(policy),
        "alignment": {
            "same_dataset": same_dataset,
            "same_scorers": same_scorers,
            "common_examples": len(common_ids),
            "missing_in_candidate": missing_in_candidate,
            "added_in_candidate": added_in_candidate,
        },
        "reliability": {
            "baseline_failure_rate": baseline_failure_rate,
            "candidate_failure_rate": candidate_failure_rate,
            "failure_rate_delta": round(failure_rate_increase, 6),
            "new_failed_example_ids": new_failed,
            "new_operational_failure_ids": new_operational,
        },
        "metric_comparisons": metric_comparisons,
        "slice_comparisons": slice_comparisons,
        "per_example_regressions": per_example_regressions,
        "violations": violations,
    }


def _per_example_regressions(
    baseline_by_id: dict[str, Any],
    candidate_by_id: dict[str, Any],
    common_ids: list[str],
    min_drop: float,
) -> list[dict[str, Any]]:
    regressions = []
    for example_id in common_ids:
        baseline_scores = {
            score.scorer_name: score.score
            for score in baseline_by_id[example_id].scores
            if score.score is not None
        }
        candidate_scores = {
            score.scorer_name: score.score
            for score in candidate_by_id[example_id].scores
            if score.score is not None
        }
        for scorer in sorted(baseline_scores.keys() & candidate_scores.keys()):
            baseline_score = float(baseline_scores[scorer])
            candidate_score = float(candidate_scores[scorer])
            drop = baseline_score - candidate_score
            if drop > min_drop:
                regressions.append(
                    {
                        "example_id": example_id,
                        "scorer": scorer,
                        "baseline": baseline_score,
                        "candidate": candidate_score,
                        "drop": round(drop, 6),
                    }
                )
    return regressions


def _metric_comparison(
    baseline_summary: dict[str, Any], candidate_summary: dict[str, Any], scorer: str
) -> dict[str, Any]:
    baseline = baseline_summary["scorers"].get(scorer)
    candidate = candidate_summary["scorers"].get(scorer)
    baseline_value = baseline.get("mean_score") if baseline else None
    candidate_value = candidate.get("mean_score") if candidate else None
    delta = (
        round(candidate_value - baseline_value, 6)
        if baseline_value is not None and candidate_value is not None
        else None
    )
    return {
        "scorer": scorer,
        "baseline_mean_score": baseline_value,
        "candidate_mean_score": candidate_value,
        "delta": delta,
        "baseline_measured": baseline["counts"]["measured"] if baseline else 0,
        "candidate_measured": candidate["counts"]["measured"] if candidate else 0,
        "baseline_coverage": baseline["coverage"] if baseline else 0.0,
        "candidate_coverage": candidate["coverage"] if candidate else 0.0,
        "coverage_delta": (
            round(candidate["coverage"] - baseline["coverage"], 6)
            if baseline and candidate
            else None
        ),
    }


def _slice_comparison(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    policy: SlicePolicy,
) -> dict[str, Any]:
    baseline = _find_slice(baseline_summary, policy.dimension, policy.value)
    candidate = _find_slice(candidate_summary, policy.dimension, policy.value)
    baseline_scorer = baseline.get("scorers", {}).get(policy.scorer) if baseline else None
    candidate_scorer = candidate.get("scorers", {}).get(policy.scorer) if candidate else None
    baseline_value = baseline_scorer.get("mean_score") if baseline_scorer else None
    candidate_value = candidate_scorer.get("mean_score") if candidate_scorer else None
    delta = (
        round(candidate_value - baseline_value, 6)
        if baseline_value is not None and candidate_value is not None
        else None
    )
    return {
        "dimension": policy.dimension,
        "value": policy.value,
        "scorer": policy.scorer,
        "baseline_count": baseline.get("count", 0) if baseline else 0,
        "candidate_count": candidate.get("count", 0) if candidate else 0,
        "baseline_measured": baseline_scorer["counts"]["measured"] if baseline_scorer else 0,
        "candidate_measured": candidate_scorer["counts"]["measured"] if candidate_scorer else 0,
        "baseline_mean_score": baseline_value,
        "candidate_mean_score": candidate_value,
        "delta": delta,
    }


def _check_metric_policy(comparison: dict[str, Any], policy: MetricPolicy) -> list[dict[str, Any]]:
    value = comparison["candidate_mean_score"]
    baseline = comparison["baseline_mean_score"]
    scope = {"scorer": policy.scorer}
    if value is None or baseline is None:
        return [
            _violation(
                "metric_unavailable",
                "required scorer metric is unavailable in baseline or candidate",
                observed=comparison,
                threshold="both metrics present",
                scope=scope,
            )
        ]
    violations = []
    if policy.min_candidate is not None and value < policy.min_candidate:
        violations.append(
            _violation(
                "metric_below_minimum",
                "candidate scorer mean is below policy minimum",
                observed=value,
                threshold=policy.min_candidate,
                scope=scope,
            )
        )
    drop = baseline - value
    if policy.max_drop is not None and drop > policy.max_drop:
        violations.append(
            _violation(
                "metric_drop_exceeded",
                "candidate scorer mean dropped more than policy allows",
                observed=round(drop, 6),
                threshold=policy.max_drop,
                scope=scope,
            )
        )
    candidate_coverage = comparison.get("candidate_coverage")
    baseline_coverage = comparison.get("baseline_coverage")
    if (
        policy.min_coverage is not None
        and candidate_coverage is not None
        and candidate_coverage < policy.min_coverage
    ):
        violations.append(
            _violation(
                "coverage_below_minimum",
                "candidate scorer coverage is below policy minimum",
                observed=candidate_coverage,
                threshold=policy.min_coverage,
                scope=scope,
            )
        )
    if (
        policy.max_coverage_drop is not None
        and baseline_coverage is not None
        and candidate_coverage is not None
        and baseline_coverage - candidate_coverage > policy.max_coverage_drop
    ):
        violations.append(
            _violation(
                "coverage_drop_exceeded",
                "candidate scorer coverage dropped more than policy allows",
                observed=round(baseline_coverage - candidate_coverage, 6),
                threshold=policy.max_coverage_drop,
                scope=scope,
            )
        )
    return violations


def _check_slice_policy(comparison: dict[str, Any], policy: SlicePolicy) -> list[dict[str, Any]]:
    scope = {"dimension": policy.dimension, "value": policy.value, "scorer": policy.scorer}
    candidate_measured = comparison["candidate_measured"]
    if candidate_measured < policy.min_examples:
        return [
            _violation(
                "slice_sample_too_small",
                "candidate slice has fewer measured scores than policy requires",
                observed=candidate_measured,
                threshold=policy.min_examples,
                scope=scope,
            )
        ]
    metric_policy = MetricPolicy(
        scorer=policy.scorer,
        min_candidate=policy.min_candidate,
        max_drop=policy.max_drop,
    )
    normalized = {
        "candidate_mean_score": comparison["candidate_mean_score"],
        "baseline_mean_score": comparison["baseline_mean_score"],
    }
    violations = _check_metric_policy(normalized, metric_policy)
    for violation in violations:
        violation["scope"] = scope
        violation["code"] = f"slice_{violation['code']}"
    return violations


def _find_slice(summary: dict[str, Any], dimension: str, value: str) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in summary["slices"]
            if item["dimension"] == dimension and item["value"] == value
        ),
        None,
    )


def _parse_metric_policy(data: Any) -> MetricPolicy:
    if not isinstance(data, dict):
        raise ConfigurationError("each metric policy must be an object")
    _reject_unknown(
        data,
        {"scorer", "min_candidate", "max_drop", "min_coverage", "max_coverage_drop"},
        "metric policy",
    )
    try:
        return MetricPolicy(**data)
    except TypeError as exc:
        raise ConfigurationError(f"invalid metric policy: {exc}") from exc


def _parse_slice_policy(data: Any) -> SlicePolicy:
    if not isinstance(data, dict):
        raise ConfigurationError("each slice policy must be an object")
    _reject_unknown(
        data,
        {"dimension", "value", "scorer", "min_examples", "min_candidate", "max_drop"},
        "slice policy",
    )
    try:
        return SlicePolicy(**data)
    except TypeError as exc:
        raise ConfigurationError(f"invalid slice policy: {exc}") from exc


def _policy_dict(policy: RegressionPolicy) -> dict[str, Any]:
    return {
        "require_same_dataset": policy.require_same_dataset,
        "require_same_examples": policy.require_same_examples,
        "require_same_scorers": policy.require_same_scorers,
        "max_new_failed_examples": policy.max_new_failed_examples,
        "max_new_operational_failures": policy.max_new_operational_failures,
        "max_failure_rate_increase": policy.max_failure_rate_increase,
        "max_per_example_regressions": policy.max_per_example_regressions,
        "per_example_min_drop": policy.per_example_min_drop,
        "metrics": [vars(item) for item in policy.metrics],
        "slices": [vars(item) for item in policy.slices],
    }


def _dataset_identity(run: EvaluationRun) -> dict[str, str]:
    return {
        "dataset_id": run.metadata.dataset.dataset_id,
        "version": run.metadata.dataset.version,
        "checksum_sha256": run.metadata.dataset.checksum_sha256,
    }


def _scorer_configs(run: EvaluationRun) -> dict[str, dict[str, Any]]:
    return {
        str(config.get("name", f"unnamed-{index}")): config
        for index, config in enumerate(run.metadata.scorer_configs)
    }


def _failure_rate(run: EvaluationRun) -> float:
    if not run.examples:
        return 0.0
    return round(
        sum(item.status == ExampleStatus.FAILED for item in run.examples) / len(run.examples),
        6,
    )


def _validate_unit_interval(value: float | None, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigurationError(f"{name} must be numeric")
    if not 0 <= value <= 1:
        raise ConfigurationError(f"{name} must be between 0 and 1")


def _reject_unknown(data: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown {context} fields: {', '.join(unknown)}")


def _violation(
    code: str,
    message: str,
    *,
    observed: Any,
    threshold: Any,
    scope: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "scope": scope or {},
        "observed": observed,
        "threshold": threshold,
    }
