"""Strict JSON configuration and built-in component construction."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalreliability.artifacts import read_run
from evalreliability.candidates import Candidate, FixtureCandidate, RunArtifactCandidate
from evalreliability.claude_cli import ClaudeCliCandidate
from evalreliability.dataset import EvaluationDataset, load_dataset
from evalreliability.engine import EvaluationConfig
from evalreliability.errors import ConfigurationError
from evalreliability.models import CandidateRequest, CandidateResponse
from evalreliability.regression import RegressionPolicy
from evalreliability.reliability import InvocationPolicy
from evalreliability.scorers import (
    ContainsAllScorer,
    ExactMatchScorer,
    JsonFieldsScorer,
    JsonSchemaScorer,
    JudgeScorer,
    Scorer,
    TextConstraintScorer,
)


@dataclass(frozen=True)
class EvaluationSpec:
    dataset: EvaluationDataset
    candidate: Candidate
    scorers: tuple[Scorer, ...]
    evaluation: EvaluationConfig
    config_path: Path


class _ImportedCandidate(Candidate):
    """Retain the configured factory identity without persisting factory arguments."""

    def __init__(self, delegate: Candidate, factory: str) -> None:
        self.delegate = delegate
        self.factory = factory

    @property
    def identifier(self) -> str:
        return self.delegate.identifier

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "python",
            "factory": self.factory,
            "delegate": self.delegate.configuration(),
        }

    async def generate(self, request: CandidateRequest) -> CandidateResponse:
        return await self.delegate.generate(request)


def load_evaluation_spec(
    path: str | Path,
    *,
    allowed_root: str | Path | None = None,
    allow_python_plugins: bool = True,
    allow_external_processes: bool = True,
) -> EvaluationSpec:
    root = Path(allowed_root).resolve() if allowed_root is not None else None
    config_path = _resolve_path(path, Path.cwd(), root)
    payload = load_json_object(config_path)
    _reject_unknown(
        payload,
        {"schema_version", "dataset", "candidate", "scorers", "evaluation"},
        "evaluation config",
    )
    if payload.get("schema_version") != "1":
        raise ConfigurationError("evaluation config schema_version must be '1'")
    base_dir = config_path.parent

    dataset_value = payload.get("dataset")
    if not isinstance(dataset_value, str) or not dataset_value:
        raise ConfigurationError("evaluation config dataset must be a non-empty path string")
    dataset_path = _resolve_path(dataset_value, base_dir, root)
    candidate_value = _require_object(payload, "candidate")
    scorers_value = payload.get("scorers")
    if not isinstance(scorers_value, list) or not scorers_value:
        raise ConfigurationError("evaluation config scorers must be a non-empty array")
    evaluation_value = payload.get("evaluation", {})
    if not isinstance(evaluation_value, dict):
        raise ConfigurationError("evaluation must be an object")

    scorers = tuple(
        build_scorer(
            item,
            base_dir=base_dir,
            allowed_root=root,
            allow_python_plugins=allow_python_plugins,
            allow_external_processes=allow_external_processes,
        )
        for item in scorers_value
    )
    candidate = build_candidate(
        candidate_value,
        base_dir=base_dir,
        allowed_root=root,
        allow_python_plugins=allow_python_plugins,
        allow_external_processes=allow_external_processes,
    )
    dataset = load_dataset(dataset_path, source=dataset_value)
    if (
        isinstance(candidate, RunArtifactCandidate)
        and candidate.source_dataset_checksum != dataset.descriptor.checksum_sha256
    ):
        raise ConfigurationError(
            "run-artifact candidate dataset checksum does not match evaluation dataset"
        )
    return EvaluationSpec(
        dataset=dataset,
        candidate=candidate,
        scorers=scorers,
        evaluation=_parse_evaluation_config(evaluation_value),
        config_path=config_path,
    )


def build_candidate(
    data: Any,
    *,
    base_dir: Path,
    allowed_root: Path | None = None,
    allow_python_plugins: bool = True,
    allow_external_processes: bool = True,
) -> Candidate:
    if not isinstance(data, dict):
        raise ConfigurationError("candidate configuration must be an object")
    candidate_type = data.get("type")
    if candidate_type == "fixture":
        _reject_unknown(
            data,
            {"type", "identifier", "responses_file", "default_response", "delay_seconds"},
            "fixture candidate",
        )
        identifier = data.get("identifier")
        responses_file = data.get("responses_file")
        if not isinstance(identifier, str) or not identifier:
            raise ConfigurationError("fixture candidate identifier must be a non-empty string")
        if not isinstance(responses_file, str) or not responses_file:
            raise ConfigurationError("fixture responses_file must be a non-empty path string")
        responses_path = _resolve_path(responses_file, base_dir, allowed_root)
        responses_payload = load_json_object(responses_path)
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in responses_payload.items()
        ):
            raise ConfigurationError("fixture responses file must map string IDs to string outputs")
        default = data.get("default_response")
        if default is not None and not isinstance(default, str):
            raise ConfigurationError("fixture default_response must be a string or null")
        delay = data.get("delay_seconds", 0.0)
        if not isinstance(delay, (int, float)) or isinstance(delay, bool):
            raise ConfigurationError("fixture delay_seconds must be numeric")
        return FixtureCandidate(
            identifier,
            responses_payload,
            default_response=default,
            delay_seconds=float(delay),
        )
    if candidate_type == "claude_cli":
        if not allow_external_processes:
            raise ConfigurationError("external-process candidates are disabled for this interface")
        _reject_unknown(
            data,
            {"type", "model", "system_prompt", "timeout_seconds", "max_turns", "executable"},
            "Claude CLI candidate",
        )
        _validate_string_fields(
            data,
            {"model", "system_prompt", "executable"},
            "Claude CLI candidate",
        )
        if "timeout_seconds" in data and (
            not isinstance(data["timeout_seconds"], (int, float))
            or isinstance(data["timeout_seconds"], bool)
        ):
            raise ConfigurationError("Claude CLI candidate timeout_seconds must be numeric")
        if "max_turns" in data and (
            not isinstance(data["max_turns"], int) or isinstance(data["max_turns"], bool)
        ):
            raise ConfigurationError("Claude CLI candidate max_turns must be an integer")
        kwargs = {key: value for key, value in data.items() if key != "type"}
        try:
            return ClaudeCliCandidate(**kwargs)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid Claude CLI candidate: {exc}") from exc
    if candidate_type == "run_artifact":
        _reject_unknown(data, {"type", "run_file"}, "run-artifact candidate")
        run_file = data.get("run_file")
        if not isinstance(run_file, str) or not run_file:
            raise ConfigurationError("run-artifact candidate run_file must be a path string")
        run_path = _resolve_path(run_file, base_dir, allowed_root)
        return RunArtifactCandidate(read_run(run_path), source=run_file)
    if candidate_type == "python":
        if not allow_python_plugins:
            raise ConfigurationError("Python candidate plugins are disabled for this interface")
        _reject_unknown(data, {"type", "factory", "kwargs"}, "Python candidate")
        factory_path = data.get("factory")
        kwargs = data.get("kwargs", {})
        if not isinstance(factory_path, str) or ":" not in factory_path:
            raise ConfigurationError("Python candidate factory must use 'module:callable' syntax")
        if not isinstance(kwargs, dict):
            raise ConfigurationError("Python candidate kwargs must be an object")
        module_name, attribute = factory_path.split(":", 1)
        try:
            factory = getattr(importlib.import_module(module_name), attribute)
            candidate = factory(**kwargs)
        except Exception as exc:
            raise ConfigurationError(
                f"could not construct Python candidate {factory_path!r}: {exc}"
            ) from exc
        if not isinstance(candidate, Candidate):
            raise ConfigurationError(
                f"Python candidate factory {factory_path!r} did not return a Candidate"
            )
        return _ImportedCandidate(candidate, factory_path)
    raise ConfigurationError(f"unsupported candidate type {candidate_type!r}")


def build_scorer(
    data: Any,
    *,
    base_dir: Path,
    allowed_root: Path | None = None,
    allow_python_plugins: bool = True,
    allow_external_processes: bool = True,
) -> Scorer:
    if not isinstance(data, dict):
        raise ConfigurationError("scorer configuration must be an object")
    scorer_type = data.get("type")
    if scorer_type == "exact_match":
        allowed = {
            "type",
            "name",
            "expected_key",
            "strip",
            "case_sensitive",
            "collapse_whitespace",
        }
        _reject_unknown(data, allowed, "exact-match scorer")
        _validate_string_fields(data, {"name", "expected_key"}, "exact-match scorer")
        _validate_boolean_fields(
            data,
            {"strip", "case_sensitive", "collapse_whitespace"},
            "exact-match scorer",
        )
        return _construct(ExactMatchScorer, data)
    if scorer_type == "contains_all":
        _reject_unknown(
            data,
            {"type", "name", "expected_key", "case_sensitive"},
            "contains-all scorer",
        )
        _validate_string_fields(data, {"name", "expected_key"}, "contains-all scorer")
        _validate_boolean_fields(data, {"case_sensitive"}, "contains-all scorer")
        return _construct(ContainsAllScorer, data)
    if scorer_type == "text_constraints":
        _reject_unknown(
            data,
            {"type", "name", "expected_key", "case_sensitive"},
            "text-constraint scorer",
        )
        _validate_string_fields(data, {"name", "expected_key"}, "text-constraint scorer")
        _validate_boolean_fields(data, {"case_sensitive"}, "text-constraint scorer")
        return _construct(TextConstraintScorer, data)
    if scorer_type == "json_schema":
        _reject_unknown(data, {"type", "name", "expected_key", "schema"}, "JSON-schema scorer")
        _validate_string_fields(data, {"name", "expected_key"}, "JSON-schema scorer")
        if "schema" in data and data["schema"] is not None and not isinstance(data["schema"], dict):
            raise ConfigurationError("JSON-schema scorer schema must be an object or null")
        return _construct(JsonSchemaScorer, data)
    if scorer_type == "json_fields":
        _reject_unknown(data, {"type", "name", "expected_key"}, "JSON-fields scorer")
        _validate_string_fields(data, {"name", "expected_key"}, "JSON-fields scorer")
        return _construct(JsonFieldsScorer, data)
    if scorer_type == "judge":
        _reject_unknown(
            data,
            {
                "type",
                "name",
                "candidate",
                "rubric",
                "pass_threshold",
                "invocation",
                "seed",
                "exclude_expected_keys",
                "rubric_key",
                "output_contract",
            },
            "judge scorer",
        )
        judge_candidate = build_candidate(
            _require_object(data, "candidate"),
            base_dir=base_dir,
            allowed_root=allowed_root,
            allow_python_plugins=allow_python_plugins,
            allow_external_processes=allow_external_processes,
        )
        rubric = data.get("rubric")
        rubric_key = data.get("rubric_key")
        if rubric is not None and not isinstance(rubric, str):
            raise ConfigurationError("judge rubric must be a string")
        if rubric_key is not None and not isinstance(rubric_key, str):
            raise ConfigurationError("judge rubric_key must be a string")
        if (rubric is None) == (rubric_key is None):
            raise ConfigurationError("judge requires exactly one of rubric or rubric_key")
        _validate_string_fields(data, {"name", "rubric_key", "output_contract"}, "judge scorer")
        if "pass_threshold" in data and (
            not isinstance(data["pass_threshold"], (int, float))
            or isinstance(data["pass_threshold"], bool)
        ):
            raise ConfigurationError("judge pass_threshold must be numeric")
        if (
            "seed" in data
            and data["seed"] is not None
            and (not isinstance(data["seed"], int) or isinstance(data["seed"], bool))
        ):
            raise ConfigurationError("judge seed must be an integer or null")
        if "exclude_expected_keys" in data and (
            not isinstance(data["exclude_expected_keys"], list)
            or not all(isinstance(key, str) and key for key in data["exclude_expected_keys"])
        ):
            raise ConfigurationError(
                "judge exclude_expected_keys must be an array of non-empty strings"
            )
        invocation = _parse_invocation_policy(data.get("invocation", {}))
        kwargs = {
            key: value
            for key, value in data.items()
            if key
            in {
                "name",
                "pass_threshold",
                "seed",
                "exclude_expected_keys",
                "rubric_key",
                "output_contract",
            }
        }
        try:
            return JudgeScorer(
                judge_candidate,
                rubric,
                invocation_policy=invocation,
                **kwargs,
            )
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid judge scorer: {exc}") from exc
    raise ConfigurationError(f"unsupported scorer type {scorer_type!r}")


def load_regression_policy(path: str | Path) -> RegressionPolicy:
    return RegressionPolicy.from_dict(load_json_object(path))


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite number {value}")

        payload = json.loads(raw, parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ConfigurationError(f"could not read JSON object {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"JSON root must be an object: {source}")
    return payload


def _parse_evaluation_config(data: dict[str, Any]) -> EvaluationConfig:
    _reject_unknown(
        data,
        {"concurrency", "scorer_timeout_seconds", "seed", "candidate_invocation"},
        "evaluation settings",
    )
    invocation = _parse_invocation_policy(data.get("candidate_invocation", {}))
    kwargs = {key: value for key, value in data.items() if key != "candidate_invocation"}
    try:
        return EvaluationConfig(candidate_policy=invocation, **kwargs)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid evaluation settings: {exc}") from exc


def _parse_invocation_policy(data: Any) -> InvocationPolicy:
    if not isinstance(data, dict):
        raise ConfigurationError("invocation policy must be an object")
    _reject_unknown(
        data,
        {
            "timeout_seconds",
            "max_attempts",
            "initial_backoff_seconds",
            "max_backoff_seconds",
            "jitter_ratio",
        },
        "invocation policy",
    )
    try:
        return InvocationPolicy(**data)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid invocation policy: {exc}") from exc


def _construct(component: Any, data: dict[str, Any]) -> Scorer:
    kwargs = {key: value for key, value in data.items() if key != "type"}
    try:
        scorer = component(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid {data.get('type')} scorer: {exc}") from exc
    if not isinstance(scorer, Scorer):  # pragma: no cover - internal constructor contract
        raise ConfigurationError(f"{data.get('type')} constructor did not return a Scorer")
    return scorer


def _resolve_path(value: str | Path, base_dir: Path, allowed_root: Path | None) -> Path:
    path = Path(value)
    resolved = (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
    if allowed_root is not None and not resolved.is_relative_to(allowed_root):
        raise ConfigurationError(f"path escapes allowed root {allowed_root}: {resolved}")
    return resolved


def _require_object(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be an object")
    return value


def _reject_unknown(data: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown {context} fields: {', '.join(unknown)}")


def _validate_string_fields(data: dict[str, Any], fields: set[str], context: str) -> None:
    invalid = [
        key
        for key in sorted(fields & data.keys())
        if not isinstance(data[key], str) or not data[key]
    ]
    if invalid:
        raise ConfigurationError(
            f"{context} fields must be non-empty strings: {', '.join(invalid)}"
        )


def _validate_boolean_fields(data: dict[str, Any], fields: set[str], context: str) -> None:
    invalid = [key for key in sorted(fields & data.keys()) if not isinstance(data[key], bool)]
    if invalid:
        raise ConfigurationError(f"{context} fields must be booleans: {', '.join(invalid)}")
