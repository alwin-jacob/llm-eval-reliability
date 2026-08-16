"""Stable in-memory and artifact data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from typing import Any


class FailureOrigin(StrEnum):
    MODEL = "model"
    EVALUATOR = "evaluator"
    INFRASTRUCTURE = "infrastructure"


class AttemptOutcome(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ScoreOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"
    ERROR = "error"


class ExampleStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")


@dataclass(frozen=True)
class CandidateRequest:
    example_id: str
    input: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateResponse:
    output: str
    model_id: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.output, str):
            raise TypeError("candidate output must be a string")


@dataclass(frozen=True)
class FailureRecord:
    origin: FailureOrigin
    stage: str
    code: str
    message: str
    retryable: bool
    exception_type: str | None = None


@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    outcome: AttemptOutcome
    started_at: str
    latency_ms: float
    failure: FailureRecord | None = None


@dataclass(frozen=True)
class ScoreResult:
    scorer_name: str
    scorer_version: str
    outcome: ScoreOutcome
    score: float | None
    passed: bool | None
    details: dict[str, Any] = field(default_factory=dict)
    failure: FailureRecord | None = None
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        if self.outcome == ScoreOutcome.PASS and self.passed is not True:
            raise ValueError("pass outcome requires passed=True")
        if self.outcome == ScoreOutcome.FAIL and self.passed is not False:
            raise ValueError("fail outcome requires passed=False")
        if self.outcome in (ScoreOutcome.SKIPPED, ScoreOutcome.ERROR) and self.passed is not None:
            raise ValueError("skipped/error outcomes require passed=None")

    @classmethod
    def passed_result(
        cls,
        scorer_name: str,
        scorer_version: str,
        *,
        score: float = 1.0,
        details: dict[str, Any] | None = None,
    ) -> ScoreResult:
        return cls(
            scorer_name,
            scorer_version,
            ScoreOutcome.PASS,
            score,
            True,
            details or {},
        )

    @classmethod
    def failed_result(
        cls,
        scorer_name: str,
        scorer_version: str,
        *,
        score: float = 0.0,
        details: dict[str, Any] | None = None,
        failure: FailureRecord | None = None,
    ) -> ScoreResult:
        return cls(
            scorer_name,
            scorer_version,
            ScoreOutcome.FAIL,
            score,
            False,
            details or {},
            failure,
        )

    @classmethod
    def skipped_result(cls, scorer_name: str, scorer_version: str, *, reason: str) -> ScoreResult:
        return cls(
            scorer_name,
            scorer_version,
            ScoreOutcome.SKIPPED,
            None,
            None,
            {"reason": reason},
        )


@dataclass(frozen=True)
class EvaluationExample:
    id: str
    input: str
    expected: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetDescriptor:
    dataset_id: str
    version: str
    checksum_sha256: str
    example_count: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExampleResult:
    example_id: str
    status: ExampleStatus
    input: str
    expected: dict[str, Any]
    metadata: dict[str, Any]
    response: CandidateResponse | None
    attempts: tuple[AttemptRecord, ...]
    scores: tuple[ScoreResult, ...]
    failures: tuple[FailureRecord, ...]
    latency_ms: float


@dataclass(frozen=True)
class RunMetadata:
    schema_version: str
    run_id: str
    config_fingerprint: str
    started_at: str
    completed_at: str
    duration_ms: float
    dataset: DatasetDescriptor
    candidate_id: str
    candidate_config: dict[str, Any]
    scorer_configs: tuple[dict[str, Any], ...]
    evaluation_config: dict[str, Any]
    environment: dict[str, Any]


@dataclass(frozen=True)
class EvaluationRun:
    metadata: RunMetadata
    examples: tuple[ExampleResult, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = _jsonable(asdict(self))
        if not isinstance(payload, dict):  # pragma: no cover - dataclass shape is fixed
            raise TypeError("run serialization did not produce an object")
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRun:
        metadata_data = _required_mapping(data, "metadata")
        dataset_data = _required_mapping(metadata_data, "dataset")
        metadata = RunMetadata(
            schema_version=str(metadata_data["schema_version"]),
            run_id=str(metadata_data["run_id"]),
            config_fingerprint=str(metadata_data["config_fingerprint"]),
            started_at=str(metadata_data["started_at"]),
            completed_at=str(metadata_data["completed_at"]),
            duration_ms=float(metadata_data["duration_ms"]),
            dataset=DatasetDescriptor(
                dataset_id=str(dataset_data["dataset_id"]),
                version=str(dataset_data["version"]),
                checksum_sha256=str(dataset_data["checksum_sha256"]),
                example_count=int(dataset_data["example_count"]),
                source=str(dataset_data["source"]),
                metadata=dict(dataset_data.get("metadata", {})),
            ),
            candidate_id=str(metadata_data["candidate_id"]),
            candidate_config=dict(metadata_data["candidate_config"]),
            scorer_configs=tuple(dict(item) for item in metadata_data["scorer_configs"]),
            evaluation_config=dict(metadata_data["evaluation_config"]),
            environment=dict(metadata_data["environment"]),
        )
        examples = tuple(_example_result_from_dict(item) for item in data["examples"])
        return cls(metadata=metadata, examples=examples)


def _example_result_from_dict(data: dict[str, Any]) -> ExampleResult:
    response_data = data.get("response")
    response = None
    if response_data is not None:
        usage_data = response_data.get("usage", {})
        response = CandidateResponse(
            output=str(response_data["output"]),
            model_id=response_data.get("model_id"),
            provider=response_data.get("provider"),
            finish_reason=response_data.get("finish_reason"),
            usage=Usage(
                input_tokens=usage_data.get("input_tokens"),
                output_tokens=usage_data.get("output_tokens"),
                total_tokens=usage_data.get("total_tokens"),
                cost_usd=usage_data.get("cost_usd"),
            ),
            metadata=dict(response_data.get("metadata", {})),
        )
    attempts = tuple(
        AttemptRecord(
            attempt=int(item["attempt"]),
            outcome=AttemptOutcome(item["outcome"]),
            started_at=str(item["started_at"]),
            latency_ms=float(item["latency_ms"]),
            failure=_failure_from_dict(item.get("failure")),
        )
        for item in data.get("attempts", [])
    )
    scores = tuple(
        ScoreResult(
            scorer_name=str(item["scorer_name"]),
            scorer_version=str(item["scorer_version"]),
            outcome=ScoreOutcome(item["outcome"]),
            score=None if item.get("score") is None else float(item["score"]),
            passed=item.get("passed"),
            details=dict(item.get("details", {})),
            failure=_failure_from_dict(item.get("failure")),
            latency_ms=float(item.get("latency_ms", 0.0)),
        )
        for item in data.get("scores", [])
    )
    failures = tuple(_failure_from_dict(item) for item in data.get("failures", []))
    if any(item is None for item in failures):  # pragma: no cover - defensive artifact check
        raise ValueError("failure entries cannot be null")
    return ExampleResult(
        example_id=str(data["example_id"]),
        status=ExampleStatus(data["status"]),
        input=str(data["input"]),
        expected=dict(data.get("expected", {})),
        metadata=dict(data.get("metadata", {})),
        response=response,
        attempts=attempts,
        scores=scores,
        failures=tuple(item for item in failures if item is not None),
        latency_ms=float(data["latency_ms"]),
    )


def _failure_from_dict(data: dict[str, Any] | None) -> FailureRecord | None:
    if data is None:
        return None
    return FailureRecord(
        origin=FailureOrigin(data["origin"]),
        stage=str(data["stage"]),
        code=str(data["code"]),
        message=str(data["message"]),
        retryable=bool(data["retryable"]),
        exception_type=data.get("exception_type"),
    )


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
