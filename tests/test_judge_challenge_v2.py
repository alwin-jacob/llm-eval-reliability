from __future__ import annotations

import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from evalreliability.artifacts import read_run
from evalreliability.candidates import Candidate, ControlledResponseCandidate, RunArtifactCandidate
from evalreliability.config import load_evaluation_spec
from evalreliability.dataset import load_dataset
from evalreliability.models import CandidateRequest, CandidateResponse, ScoreOutcome
from evalreliability.scorers import JudgeScorer

PROJECT_ROOT = Path(__file__).parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/judge-challenge-v2"


def test_controlled_challenge_set_has_balanced_close_pairs() -> None:
    dataset = load_dataset(EXPERIMENT_ROOT / "dataset/v1")
    by_pair: dict[str, list] = defaultdict(list)
    labels: Counter[str] = Counter()
    families: set[str] = set()

    for example in dataset.examples:
        metadata = example.metadata
        by_pair[metadata["pair_id"]].append(example)
        labels[metadata["construction"]["intended_label"]] += 1
        families.add(metadata["challenge_family"])
        assert metadata["construction"]["human_reference"] is False
        assert "human_reference_label" not in example.expected
        assert isinstance(metadata["controlled_response"], str)
        assert metadata["slices"]["challenge_family"] == metadata["challenge_family"]

    assert dataset.descriptor.example_count == 24
    assert len(by_pair) == 12
    assert len(families) == 12
    assert labels == {"pass": 12, "fail": 12}
    for examples in by_pair.values():
        assert len(examples) == 2
        assert len({example.input for example in examples}) == 1
        assert len({example.expected["rubric"] for example in examples}) == 1
        assert len({example.metadata["controlled_response"] for example in examples}) == 2
        responses = [example.metadata["controlled_response"] for example in examples]
        assert SequenceMatcher(None, *responses).ratio() >= 0.60
        assert {example.metadata["variant_identifier"] for example in examples} == {"a", "b"}
        assert {example.metadata["construction"]["intended_label"] for example in examples} == {
            "pass",
            "fail",
        }


def test_controlled_response_artifact_is_explicitly_non_model_and_exact() -> None:
    spec = load_evaluation_spec(EXPERIMENT_ROOT / "configs/materialize-controlled-responses.json")
    run = read_run(EXPERIMENT_ROOT / "artifacts/controlled-responses.run.json")

    assert isinstance(spec.candidate, ControlledResponseCandidate)
    assert spec.candidate.configuration()["model_calls"] is False
    assert run.metadata.candidate_config["response_origin"] == (
        "project_authored_controlled_challenge_set"
    )
    assert run.metadata.candidate_config["model_calls"] is False
    assert len(run.examples) == 24
    for example, result in zip(spec.dataset.examples, run.examples, strict=True):
        assert result.response is not None
        assert result.response.output == example.metadata["controlled_response"]
        assert result.response.provider == "controlled_experiment_fixture"
        assert result.response.metadata["model_call"] is False
        assert result.scores[0].outcome == ScoreOutcome.SKIPPED


def test_sonnet_config_replays_frozen_responses_once_with_rubric_only_prompt() -> None:
    spec = load_evaluation_spec(EXPERIMENT_ROOT / "configs/judge-sonnet.json")
    scorer_config = spec.scorers[0].configuration()

    assert isinstance(spec.candidate, RunArtifactCandidate)
    assert spec.evaluation.concurrency == 2
    assert scorer_config["candidate_id"] == "claude-cli:sonnet"
    assert scorer_config["candidate"]["tools_disabled"] is True
    assert scorer_config["candidate"]["dynamic_system_prompt_sections_excluded"] is True
    assert scorer_config["invocation"]["max_attempts"] == 1
    assert scorer_config["rubric_key"] == "rubric"
    assert scorer_config["output_contract"] == "binary_verdict"
    assert scorer_config["prompt_payload_fields"] == [
        "input",
        "candidate_output",
        "rubric",
    ]


@pytest.mark.asyncio
async def test_judge_prompt_leaks_neither_labels_pair_metadata_nor_sibling_response() -> None:
    class CapturingJudge(Candidate):
        prompts: dict[str, str]

        def __init__(self) -> None:
            self.prompts = {}

        @property
        def identifier(self) -> str:
            return "capturing-judge"

        def configuration(self) -> dict[str, str]:
            return {"type": "test"}

        async def generate(self, request: CandidateRequest) -> CandidateResponse:
            self.prompts[request.example_id] = request.input
            return CandidateResponse(output='{"verdict":"PASS","rationale":"captured"}')

    dataset = load_dataset(EXPERIMENT_ROOT / "dataset/v1")
    by_pair = {
        pair_id: [example for example in dataset.examples if example.metadata["pair_id"] == pair_id]
        for pair_id in {example.metadata["pair_id"] for example in dataset.examples}
    }
    judge = CapturingJudge()
    scorer = JudgeScorer(
        judge,
        None,
        rubric_key="rubric",
        output_contract="binary_verdict",
    )

    for example in dataset.examples:
        output = example.metadata["controlled_response"]
        result = await scorer.score(example, CandidateResponse(output=output))
        assert result.outcome == ScoreOutcome.PASS
        payload = json.loads(judge.prompts[example.id].split("\n", 1)[1])
        sibling = next(
            item for item in by_pair[example.metadata["pair_id"]] if item.id != example.id
        )
        serialized = json.dumps(payload, ensure_ascii=False)

        assert payload == {
            "candidate_output": output,
            "input": example.input,
            "rubric": example.expected["rubric"],
        }
        assert example.id not in serialized
        assert example.metadata["pair_id"] not in serialized
        assert example.metadata["variant_identifier"] not in payload
        assert "construction" not in payload
        assert payload["candidate_output"] != sibling.metadata["controlled_response"]
