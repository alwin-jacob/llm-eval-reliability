from __future__ import annotations

import pytest

from evalreliability.candidates import Candidate, FixtureCandidate
from evalreliability.errors import EvaluatorError
from evalreliability.models import (
    CandidateRequest,
    CandidateResponse,
    EvaluationExample,
    FailureOrigin,
    ScoreOutcome,
)
from evalreliability.reliability import InvocationPolicy
from evalreliability.scorers import (
    ContainsAllScorer,
    ExactMatchScorer,
    JsonFieldsScorer,
    JsonSchemaScorer,
    JudgeScorer,
    TextConstraintScorer,
)


@pytest.mark.asyncio
async def test_deterministic_scorers_are_explicit_about_normalization_and_coverage() -> None:
    example = EvaluationExample(
        "one",
        "question",
        {"text": "Hello world", "required_terms": ["hello", "world"]},
    )
    response = CandidateResponse(output="  HELLO   WORLD ")

    exact = await ExactMatchScorer().score(example, response)
    contains = await ContainsAllScorer().score(example, response)
    skipped = await ExactMatchScorer(expected_key="missing").score(example, response)

    assert exact.outcome == ScoreOutcome.PASS
    assert contains.outcome == ScoreOutcome.PASS
    assert skipped.outcome == ScoreOutcome.SKIPPED


@pytest.mark.asyncio
async def test_text_constraint_scorer_reports_each_lexical_and_layout_check() -> None:
    example = EvaluationExample(
        "one",
        "question",
        {
            "text_constraints": {
                "required_terms": ["risk", "rollback"],
                "forbidden_terms": ["0", "1"],
                "exact_line_count": 1,
                "starts_with": "DECISION:",
                "ends_with": "approved.",
            }
        },
    )

    passed = await TextConstraintScorer(case_sensitive=True).score(
        example,
        CandidateResponse(output="DECISION: risk review and rollback make this approved."),
    )
    failed = await TextConstraintScorer(case_sensitive=True).score(
        example,
        CandidateResponse(output="DECISION: risk review makes this approved.\nrollback option 1"),
    )

    assert passed.outcome == ScoreOutcome.PASS
    assert all(passed.details["checks"].values())
    assert failed.outcome == ScoreOutcome.FAIL
    assert failed.details["checks"] == {
        "required_terms": True,
        "forbidden_terms": False,
        "exact_line_count": False,
        "starts_with": True,
        "ends_with": False,
    }


@pytest.mark.asyncio
async def test_malformed_candidate_json_is_model_quality_failure() -> None:
    example = EvaluationExample(
        "one",
        "question",
        {
            "schema": {
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
            },
            "values": {"answer": "yes"},
        },
    )
    response = CandidateResponse(output="not JSON")

    schema_result = await JsonSchemaScorer().score(example, response)
    fields_result = await JsonFieldsScorer().score(example, response)

    assert schema_result.outcome == ScoreOutcome.FAIL
    assert schema_result.failure is not None
    assert schema_result.failure.origin == FailureOrigin.MODEL
    assert schema_result.failure.code == "malformed_json"
    assert fields_result.failure is not None
    assert fields_result.failure.code == "malformed_json"


@pytest.mark.asyncio
async def test_schema_mismatch_reports_bounded_validation_details() -> None:
    example = EvaluationExample(
        "one",
        "question",
        {
            "schema": {
                "type": "object",
                "required": ["count"],
                "properties": {"count": {"type": "integer"}},
                "additionalProperties": False,
            }
        },
    )

    result = await JsonSchemaScorer().score(
        example, CandidateResponse(output='{"count":"three","extra":true}')
    )

    assert result.outcome == ScoreOutcome.FAIL
    assert result.failure is not None
    assert result.failure.code == "schema_mismatch"
    assert len(result.details["schema_errors"]) == 2


@pytest.mark.asyncio
async def test_judge_derives_verdict_from_threshold() -> None:
    judge_candidate = FixtureCandidate(
        "scripted-judge",
        {"one": '{"score":0.75,"rationale":"The required fact is present."}'},
    )
    scorer = JudgeScorer(
        judge_candidate,
        "Pass if the response contains the required fact.",
        pass_threshold=0.7,
        invocation_policy=InvocationPolicy(max_attempts=1),
    )
    example = EvaluationExample("one", "question", {"text": "answer"})

    result = await scorer.score(example, CandidateResponse(output="answer"))

    assert result.outcome == ScoreOutcome.PASS
    assert result.details["verdict"] == "pass"
    assert "ground truth" in scorer.configuration()["claim"]


@pytest.mark.asyncio
async def test_malformed_judge_output_is_evaluator_failure() -> None:
    judge_candidate = FixtureCandidate("bad-judge", {"one": "I think it passes"})
    scorer = JudgeScorer(
        judge_candidate,
        "Pass only correct answers.",
        invocation_policy=InvocationPolicy(max_attempts=1),
    )

    with pytest.raises(EvaluatorError, match="malformed JSON") as captured:
        await scorer.score(
            EvaluationExample("one", "question", {"text": "answer"}),
            CandidateResponse(output="answer"),
        )

    assert captured.value.code == "judge_invalid_json"
    assert captured.value.details["judge_output"] == "I think it passes"


@pytest.mark.asyncio
async def test_judge_prompt_excludes_calibration_label_but_keeps_evaluation_reference() -> None:
    class CapturingJudge(Candidate):
        prompt = ""

        @property
        def identifier(self) -> str:
            return "capturing-judge"

        def configuration(self) -> dict[str, str]:
            return {"type": "test"}

        async def generate(self, request: CandidateRequest) -> CandidateResponse:
            self.prompt = request.input
            return CandidateResponse(output='{"score":0.8,"rationale":"meets criteria"}')

    candidate = CapturingJudge()
    scorer = JudgeScorer(candidate, "Apply the criteria.")
    example = EvaluationExample(
        "one",
        "question",
        {"reference_label": "pass", "criteria": "Must state the correct answer."},
    )

    await scorer.score(example, CandidateResponse(output="correct answer"))

    assert "reference_label" not in candidate.prompt
    assert "Must state the correct answer." in candidate.prompt
