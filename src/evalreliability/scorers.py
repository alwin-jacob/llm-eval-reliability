"""Deterministic, structured-output, and candidate-backed judge scorers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from jsonschema import Draft202012Validator, SchemaError

from evalreliability.candidates import Candidate
from evalreliability.errors import ConfigurationError, EvaluatorError
from evalreliability.models import (
    CandidateRequest,
    CandidateResponse,
    EvaluationExample,
    FailureOrigin,
    FailureRecord,
    ScoreResult,
)
from evalreliability.reliability import InvocationPolicy, invoke_candidate


class Scorer(ABC):
    """One independently isolated evaluation of a candidate response."""

    name: str
    version: str

    @abstractmethod
    def configuration(self) -> dict[str, Any]:
        """Return JSON-safe scorer provenance."""

    @abstractmethod
    async def score(self, example: EvaluationExample, response: CandidateResponse) -> ScoreResult:
        """Return a quality result; raise EvaluatorError when evaluation is untrustworthy."""


class ExactMatchScorer(Scorer):
    version = "1"

    def __init__(
        self,
        *,
        name: str = "exact_match",
        expected_key: str = "text",
        strip: bool = True,
        case_sensitive: bool = False,
        collapse_whitespace: bool = True,
    ) -> None:
        self.name = name
        self.expected_key = expected_key
        self.strip = strip
        self.case_sensitive = case_sensitive
        self.collapse_whitespace = collapse_whitespace

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "exact_match",
            "name": self.name,
            "version": self.version,
            "expected_key": self.expected_key,
            "strip": self.strip,
            "case_sensitive": self.case_sensitive,
            "collapse_whitespace": self.collapse_whitespace,
        }

    async def score(self, example: EvaluationExample, response: CandidateResponse) -> ScoreResult:
        expected = example.expected.get(self.expected_key)
        if expected is None:
            return ScoreResult.skipped_result(
                self.name, self.version, reason=f"expected.{self.expected_key} is absent"
            )
        if not isinstance(expected, str):
            raise EvaluatorError(
                f"expected.{self.expected_key} must be a string", code="invalid_expectation"
            )
        actual_normalized = self._normalize(response.output)
        expected_normalized = self._normalize(expected)
        details = {
            "expected": expected,
            "normalization": {
                "strip": self.strip,
                "case_sensitive": self.case_sensitive,
                "collapse_whitespace": self.collapse_whitespace,
            },
        }
        if actual_normalized == expected_normalized:
            return ScoreResult.passed_result(self.name, self.version, details=details)
        return ScoreResult.failed_result(self.name, self.version, details=details)

    def _normalize(self, value: str) -> str:
        if self.strip:
            value = value.strip()
        if self.collapse_whitespace:
            value = " ".join(value.split())
        if not self.case_sensitive:
            value = value.casefold()
        return value


class ContainsAllScorer(Scorer):
    version = "1"

    def __init__(
        self,
        *,
        name: str = "contains_all",
        expected_key: str = "required_terms",
        case_sensitive: bool = False,
    ) -> None:
        self.name = name
        self.expected_key = expected_key
        self.case_sensitive = case_sensitive

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "contains_all",
            "name": self.name,
            "version": self.version,
            "expected_key": self.expected_key,
            "case_sensitive": self.case_sensitive,
        }

    async def score(self, example: EvaluationExample, response: CandidateResponse) -> ScoreResult:
        expected = example.expected.get(self.expected_key)
        if expected is None:
            return ScoreResult.skipped_result(
                self.name, self.version, reason=f"expected.{self.expected_key} is absent"
            )
        if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
            raise EvaluatorError(
                f"expected.{self.expected_key} must be a list of strings",
                code="invalid_expectation",
            )
        haystack = response.output if self.case_sensitive else response.output.casefold()
        missing = [
            term
            for term in expected
            if (term if self.case_sensitive else term.casefold()) not in haystack
        ]
        details = {"required_terms": expected, "missing_terms": missing}
        if not missing:
            return ScoreResult.passed_result(self.name, self.version, details=details)
        score = (len(expected) - len(missing)) / len(expected) if expected else 1.0
        return ScoreResult.failed_result(self.name, self.version, score=score, details=details)


class TextConstraintScorer(Scorer):
    """Check explicit lexical and layout constraints without judging semantics."""

    version = "1"
    _allowed_constraints = {
        "required_terms",
        "forbidden_terms",
        "exact_word_count",
        "exact_line_count",
        "line_prefix",
        "starts_with",
        "ends_with",
    }

    def __init__(
        self,
        *,
        name: str = "text_constraints",
        expected_key: str = "text_constraints",
        case_sensitive: bool = False,
    ) -> None:
        self.name = name
        self.expected_key = expected_key
        self.case_sensitive = case_sensitive

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "text_constraints",
            "name": self.name,
            "version": self.version,
            "expected_key": self.expected_key,
            "case_sensitive": self.case_sensitive,
        }

    async def score(self, example: EvaluationExample, response: CandidateResponse) -> ScoreResult:
        expected = example.expected.get(self.expected_key)
        if expected is None:
            return ScoreResult.skipped_result(
                self.name, self.version, reason=f"expected.{self.expected_key} is absent"
            )
        constraints = self._validate_constraints(expected)
        output = response.output.strip()
        comparable = output if self.case_sensitive else output.casefold()
        lines = output.splitlines()
        words = output.split()
        checks: dict[str, bool] = {}

        required = constraints.get("required_terms")
        if isinstance(required, list):
            checks["required_terms"] = all(
                (term if self.case_sensitive else term.casefold()) in comparable
                for term in required
            )
        forbidden = constraints.get("forbidden_terms")
        if isinstance(forbidden, list):
            checks["forbidden_terms"] = all(
                (term if self.case_sensitive else term.casefold()) not in comparable
                for term in forbidden
            )
        exact_word_count = constraints.get("exact_word_count")
        if isinstance(exact_word_count, int):
            checks["exact_word_count"] = len(words) == exact_word_count
        exact_line_count = constraints.get("exact_line_count")
        if isinstance(exact_line_count, int):
            checks["exact_line_count"] = len(lines) == exact_line_count
        line_prefix = constraints.get("line_prefix")
        if isinstance(line_prefix, str):
            prefix = line_prefix if self.case_sensitive else line_prefix.casefold()
            checks["line_prefix"] = bool(lines) and all(
                (line if self.case_sensitive else line.casefold()).startswith(prefix)
                for line in lines
            )
        starts_with = constraints.get("starts_with")
        if isinstance(starts_with, str):
            prefix = starts_with if self.case_sensitive else starts_with.casefold()
            checks["starts_with"] = comparable.startswith(prefix)
        ends_with = constraints.get("ends_with")
        if isinstance(ends_with, str):
            suffix = ends_with if self.case_sensitive else ends_with.casefold()
            checks["ends_with"] = comparable.endswith(suffix)

        details = {
            "constraints": constraints,
            "checks": checks,
            "observed_word_count": len(words),
            "observed_line_count": len(lines),
        }
        passed = all(checks.values())
        if passed:
            return ScoreResult.passed_result(self.name, self.version, details=details)
        score = sum(checks.values()) / len(checks) if checks else 1.0
        return ScoreResult.failed_result(
            self.name,
            self.version,
            score=score,
            details=details,
        )

    def _validate_constraints(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise EvaluatorError(
                f"expected.{self.expected_key} must be an object",
                code="invalid_expectation",
            )
        unknown = sorted(set(value) - self._allowed_constraints)
        if unknown:
            raise EvaluatorError(
                f"unknown text constraints: {', '.join(unknown)}",
                code="invalid_expectation",
            )
        if not value:
            raise EvaluatorError("text constraints cannot be empty", code="invalid_expectation")
        for key in ("required_terms", "forbidden_terms"):
            items = value.get(key)
            if items is not None and (
                not isinstance(items, list)
                or not items
                or not all(isinstance(item, str) and item for item in items)
            ):
                raise EvaluatorError(
                    f"text constraint {key} must be a non-empty string array",
                    code="invalid_expectation",
                )
        for key in ("exact_word_count", "exact_line_count"):
            count = value.get(key)
            if count is not None and (
                not isinstance(count, int) or isinstance(count, bool) or count < 1
            ):
                raise EvaluatorError(
                    f"text constraint {key} must be a positive integer",
                    code="invalid_expectation",
                )
        for key in ("line_prefix", "starts_with", "ends_with"):
            text = value.get(key)
            if text is not None and (not isinstance(text, str) or not text):
                raise EvaluatorError(
                    f"text constraint {key} must be a non-empty string",
                    code="invalid_expectation",
                )
        return value


class JsonSchemaScorer(Scorer):
    version = "1"

    def __init__(
        self,
        *,
        name: str = "json_schema",
        expected_key: str = "schema",
        schema: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.expected_key = expected_key
        self.schema = schema
        if schema is not None:
            self._check_schema(schema)

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "name": self.name,
            "version": self.version,
            "expected_key": self.expected_key,
            "schema": self.schema,
        }

    async def score(self, example: EvaluationExample, response: CandidateResponse) -> ScoreResult:
        schema = self.schema or example.expected.get(self.expected_key)
        if schema is None:
            return ScoreResult.skipped_result(
                self.name, self.version, reason="no JSON schema configured for example"
            )
        if not isinstance(schema, dict):
            raise EvaluatorError("JSON schema must be an object", code="invalid_expectation")
        self._check_schema(schema)
        parsed, failure = _parse_candidate_json(response.output, self.name)
        if failure is not None:
            return ScoreResult.failed_result(
                self.name,
                self.version,
                details={"valid_json": False},
                failure=failure,
            )
        errors = sorted(
            Draft202012Validator(schema).iter_errors(parsed), key=lambda e: list(e.path)
        )
        if not errors:
            return ScoreResult.passed_result(
                self.name, self.version, details={"valid_json": True, "schema_errors": []}
            )
        summarized = [
            {
                "path": list(error.absolute_path),
                "validator": error.validator,
                "message": error.message,
            }
            for error in errors[:10]
        ]
        failure = FailureRecord(
            origin=FailureOrigin.MODEL,
            stage=f"scorer:{self.name}",
            code="schema_mismatch",
            message=f"candidate JSON failed schema validation ({len(errors)} error(s))",
            retryable=False,
        )
        return ScoreResult.failed_result(
            self.name,
            self.version,
            details={"valid_json": True, "schema_errors": summarized},
            failure=failure,
        )

    @staticmethod
    def _check_schema(schema: dict[str, Any]) -> None:
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise EvaluatorError(
                f"invalid JSON schema: {exc.message}", code="invalid_schema"
            ) from exc


class JsonFieldsScorer(Scorer):
    version = "1"

    def __init__(self, *, name: str = "json_fields", expected_key: str = "values") -> None:
        self.name = name
        self.expected_key = expected_key

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "json_fields",
            "name": self.name,
            "version": self.version,
            "expected_key": self.expected_key,
        }

    async def score(self, example: EvaluationExample, response: CandidateResponse) -> ScoreResult:
        expected = example.expected.get(self.expected_key)
        if expected is None:
            return ScoreResult.skipped_result(
                self.name, self.version, reason=f"expected.{self.expected_key} is absent"
            )
        if not isinstance(expected, dict):
            raise EvaluatorError(
                f"expected.{self.expected_key} must be an object", code="invalid_expectation"
            )
        parsed, failure = _parse_candidate_json(response.output, self.name)
        if failure is not None:
            return ScoreResult.failed_result(
                self.name,
                self.version,
                details={"valid_json": False, "mismatches": sorted(expected)},
                failure=failure,
            )
        if not isinstance(parsed, dict):
            mismatch_keys = sorted(expected)
        else:
            mismatch_keys = sorted(
                key for key, value in expected.items() if parsed.get(key) != value
            )
        details = {"expected_values": expected, "mismatches": mismatch_keys}
        if not mismatch_keys:
            return ScoreResult.passed_result(self.name, self.version, details=details)
        score = (len(expected) - len(mismatch_keys)) / len(expected) if expected else 1.0
        return ScoreResult.failed_result(self.name, self.version, score=score, details=details)


class JudgeScorer(Scorer):
    """Use another Candidate as a configurable judge; its output remains fallible evidence."""

    version = "1"
    _score_output_schema = {
        "type": "object",
        "required": ["score", "rationale"],
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
        },
        "additionalProperties": False,
    }
    _binary_verdict_output_schema = {
        "type": "object",
        "required": ["verdict", "rationale"],
        "properties": {
            "verdict": {"enum": ["PASS", "FAIL"]},
            "rationale": {"type": "string"},
        },
        "additionalProperties": False,
    }
    _output_contracts = ("score", "binary_verdict")

    def __init__(
        self,
        candidate: Candidate,
        rubric: str | None,
        *,
        name: str = "judge",
        pass_threshold: float = 0.5,
        invocation_policy: InvocationPolicy | None = None,
        seed: int | None = 0,
        exclude_expected_keys: Sequence[str] = ("reference_label", "human_label"),
        rubric_key: str | None = None,
        output_contract: str = "score",
    ) -> None:
        if rubric is not None and not rubric.strip():
            raise ConfigurationError("judge rubric cannot be empty")
        if rubric_key is not None and not rubric_key.strip():
            raise ConfigurationError("judge rubric_key cannot be empty")
        if (rubric is None) == (rubric_key is None):
            raise ConfigurationError("judge requires exactly one of rubric or rubric_key")
        if not 0 <= pass_threshold <= 1:
            raise ConfigurationError("judge pass_threshold must be between 0 and 1")
        if not all(isinstance(key, str) and key for key in exclude_expected_keys):
            raise ConfigurationError("judge exclude_expected_keys must contain non-empty strings")
        if output_contract not in self._output_contracts:
            allowed = ", ".join(self._output_contracts)
            raise ConfigurationError(f"judge output_contract must be one of: {allowed}")
        self.name = name
        self.candidate = candidate
        self.rubric = rubric
        self.rubric_key = rubric_key
        self.pass_threshold = pass_threshold
        self.invocation_policy = invocation_policy or InvocationPolicy()
        self.seed = seed
        self.exclude_expected_keys = tuple(exclude_expected_keys)
        self.output_contract = output_contract

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "judge",
            "name": self.name,
            "version": self.version,
            "rubric": self.rubric,
            "rubric_key": self.rubric_key,
            "pass_threshold": self.pass_threshold,
            "output_contract": self.output_contract,
            "candidate_id": self.candidate.identifier,
            "candidate": self.candidate.configuration(),
            "invocation": self.invocation_policy.to_dict(),
            "seed": self.seed,
            "exclude_expected_keys": (
                list(self.exclude_expected_keys) if self.rubric_key is None else None
            ),
            "prompt_payload_fields": (
                ["input", "candidate_output", "rubric"]
                if self.rubric_key is not None
                else ["example_id", "input", "reference", "candidate_output", "rubric"]
            ),
            "claim": "judge output is a measurement, not ground truth",
        }

    async def score(self, example: EvaluationExample, response: CandidateResponse) -> ScoreResult:
        prompt = self._prompt(example, response)
        request = CandidateRequest(
            example_id=example.id,
            input=prompt,
            context={"role": "evaluator", "scorer": self.name},
        )
        invocation = await invoke_candidate(
            self.candidate,
            request,
            self.invocation_policy,
            seed=self.seed,
            stage=f"judge:{self.name}",
        )
        if invocation.failure is not None or invocation.response is None:
            failure = invocation.failure
            message = failure.message if failure else "judge returned no response"
            code = f"judge_{failure.code}" if failure else "judge_no_response"
            raise EvaluatorError(
                message,
                code=code,
                details={"judge_attempts": [asdict(item) for item in invocation.attempts]},
            )
        evidence = {
            "judge_attempts": [asdict(item) for item in invocation.attempts],
            "judge_output": invocation.response.output,
            "judge_model_id": invocation.response.model_id,
            "judge_provider": invocation.response.provider,
            "judge_finish_reason": invocation.response.finish_reason,
            "judge_usage": asdict(invocation.response.usage),
            "judge_metadata": invocation.response.metadata,
        }
        try:
            parsed = json.loads(invocation.response.output)
        except json.JSONDecodeError as exc:
            raise EvaluatorError(
                f"judge returned malformed JSON: {exc.msg}",
                code="judge_invalid_json",
                details=evidence,
            ) from exc
        output_schema = (
            self._binary_verdict_output_schema
            if self.output_contract == "binary_verdict"
            else self._score_output_schema
        )
        errors = list(Draft202012Validator(output_schema).iter_errors(parsed))
        if errors:
            raise EvaluatorError(
                f"judge output failed contract: {errors[0].message}",
                code="judge_invalid_output",
                details=evidence,
            )
        if self.output_contract == "binary_verdict":
            passed = parsed["verdict"] == "PASS"
            numeric_score = 1.0 if passed else 0.0
        else:
            numeric_score = float(parsed["score"])
            passed = numeric_score >= self.pass_threshold
        details = {
            "verdict": "pass" if passed else "fail",
            "rationale": parsed["rationale"],
            "judge_candidate_id": self.candidate.identifier,
            "pass_threshold": self.pass_threshold,
            **evidence,
        }
        if passed:
            return ScoreResult.passed_result(
                self.name, self.version, score=numeric_score, details=details
            )
        return ScoreResult.failed_result(
            self.name, self.version, score=numeric_score, details=details
        )

    def _prompt(self, example: EvaluationExample, response: CandidateResponse) -> str:
        if self.rubric_key is not None:
            case_rubric = example.expected.get(self.rubric_key)
            if not isinstance(case_rubric, str) or not case_rubric.strip():
                raise EvaluatorError(
                    f"expected.{self.rubric_key} must be a non-empty string",
                    code="invalid_judge_rubric",
                )
            payload = {
                "input": example.input,
                "candidate_output": response.output,
                "rubric": case_rubric,
            }
        else:
            payload = self._reference_prompt_payload(example, response)
        if self.output_contract == "binary_verdict":
            contract = (
                'Return exactly one JSON object with verdict equal to "PASS" or "FAIL" '
                "and a concise rationale; do not add markdown.\n"
            )
        else:
            contract = (
                "Return exactly one JSON object with numeric score in [0,1] and a concise "
                "rationale; do not add markdown.\n"
            )
        return (
            "Evaluate the candidate output using only the supplied rubric and evidence. "
            + contract
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )

    def _reference_prompt_payload(
        self, example: EvaluationExample, response: CandidateResponse
    ) -> dict[str, Any]:
        reference = {
            key: value
            for key, value in example.expected.items()
            if key not in self.exclude_expected_keys
        }
        return {
            "example_id": example.id,
            "input": example.input,
            "reference": reference,
            "candidate_output": response.output,
            "rubric": self.rubric,
        }


def _parse_candidate_json(output: str, scorer_name: str) -> tuple[Any | None, FailureRecord | None]:
    try:
        return json.loads(output), None
    except json.JSONDecodeError as exc:
        return None, FailureRecord(
            origin=FailureOrigin.MODEL,
            stage=f"scorer:{scorer_name}",
            code="malformed_json",
            message=f"candidate output is not valid JSON: {exc.msg}",
            retryable=False,
            exception_type=type(exc).__name__,
        )
