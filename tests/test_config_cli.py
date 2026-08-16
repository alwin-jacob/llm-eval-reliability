from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalreliability.candidates import RunArtifactCandidate
from evalreliability.claude_cli import ClaudeCliCandidate
from evalreliability.cli import EXIT_ERROR, EXIT_OK, EXIT_REGRESSION, main
from evalreliability.config import load_evaluation_spec
from evalreliability.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).parents[1]


def test_example_config_constructs_all_components() -> None:
    spec = load_evaluation_spec(PROJECT_ROOT / "examples/configs/candidate.json")

    assert spec.dataset.descriptor.example_count == 6
    assert spec.dataset.descriptor.source == "../datasets/core/v1"
    assert spec.candidate.identifier == "candidate-fixture-v2"
    assert [scorer.name for scorer in spec.scorers] == [
        "exact_match",
        "contains_all",
        "json_schema",
        "json_fields",
    ]
    assert spec.evaluation.concurrency == 4


def test_real_claude_example_config_constructs_without_invoking_provider() -> None:
    spec = load_evaluation_spec(PROJECT_ROOT / "examples/configs/real/claude-sonnet-smoke.json")

    assert spec.dataset.descriptor.dataset_id == "claude-cli-smoke"
    assert spec.dataset.descriptor.example_count == 1
    assert isinstance(spec.candidate, ClaudeCliCandidate)
    assert spec.candidate.identifier == "claude-cli:sonnet"
    assert spec.candidate.max_turns == 1
    assert spec.evaluation.candidate_policy.max_attempts == 1


def test_judge_calibration_candidate_config_constructs_without_invoking_provider() -> None:
    spec = load_evaluation_spec(
        PROJECT_ROOT / "experiments/judge-calibration-v1/configs/candidate-haiku.json"
    )

    assert spec.dataset.descriptor.example_count == 16
    assert isinstance(spec.candidate, ClaudeCliCandidate)
    assert spec.candidate.identifier == "claude-cli:haiku"
    assert spec.evaluation.concurrency == 1
    assert spec.evaluation.candidate_policy.max_attempts == 1
    assert [scorer.name for scorer in spec.scorers] == [
        "json_schema",
        "json_fields",
        "exact_constraints",
        "text_constraints",
    ]


def test_future_judge_config_targets_frozen_outputs_and_excludes_human_fields() -> None:
    config_path = PROJECT_ROOT / "experiments/judge-calibration-v1/configs/future-judge-sonnet.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    judge = payload["scorers"][0]

    assert payload["candidate"]["type"] == "run_artifact"
    assert judge["candidate"]["type"] == "claude_cli"
    assert judge["candidate"]["model"] == "sonnet"
    assert judge["invocation"]["max_attempts"] == 1
    assert judge["exclude_expected_keys"] == [
        "human_reference_label",
        "human_reference_status",
    ]


def test_config_can_disable_external_process_candidates() -> None:
    config = PROJECT_ROOT / "examples/configs/real/claude-sonnet-smoke.json"

    with pytest.raises(ConfigurationError, match="external-process candidates are disabled"):
        load_evaluation_spec(config, allow_external_processes=False)


def test_run_artifact_candidate_config_replays_frozen_responses(tmp_path: Path) -> None:
    config = tmp_path / "replay.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "dataset": str(PROJECT_ROOT / "examples/datasets/core/v1"),
                "candidate": {
                    "type": "run_artifact",
                    "run_file": str(PROJECT_ROOT / "examples/artifacts/candidate.run.json"),
                },
                "scorers": [{"type": "exact_match"}],
            }
        ),
        encoding="utf-8",
    )

    spec = load_evaluation_spec(config)

    assert isinstance(spec.candidate, RunArtifactCandidate)
    assert spec.candidate.configuration()["response_count"] == 6


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "dataset": ".",
                "candidate": {"type": "fixture", "identifier": "x", "responses_file": "x"},
                "scorers": [{"type": "exact_match", "mystery": True}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unknown exact-match scorer fields"):
        load_evaluation_spec(config)


def test_config_rejects_string_scorer_boolean(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "dataset": ".",
                "candidate": {"type": "fixture", "identifier": "x", "responses_file": "x"},
                "scorers": [{"type": "exact_match", "case_sensitive": "false"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="fields must be booleans: case_sensitive"):
        load_evaluation_spec(config)


def test_cli_runs_examples_and_enforces_failing_regression_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_config = PROJECT_ROOT / "examples/configs/baseline.json"
    candidate_config = PROJECT_ROOT / "examples/configs/candidate.json"
    policy_path = PROJECT_ROOT / "examples/configs/regression-policy.json"

    assert main(["run", str(baseline_config), "--output", str(baseline_path)]) == EXIT_OK
    assert main(["run", str(candidate_config), "--output", str(candidate_path)]) == EXIT_OK
    assert (
        main(
            [
                "compare",
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--policy",
                str(policy_path),
            ]
        )
        == EXIT_OK
    )

    strict_policy = tmp_path / "strict-policy.json"
    strict_policy.write_text(
        json.dumps(
            {
                "max_per_example_regressions": 0,
                "metrics": [{"scorer": "contains_all", "max_drop": 0.0}],
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(
        [
            "compare",
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--policy",
            str(strict_policy),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == EXIT_REGRESSION
    assert '"passed": false' in output
    assert "metric_drop_exceeded" in output
    assert "per_example_regressions" in output


def test_cli_returns_error_for_bad_config(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.json"
    bad_config.write_text("{}", encoding="utf-8")

    assert main(["run", str(bad_config), "--output", str(tmp_path / "run.json")]) == EXIT_ERROR


def test_cli_can_select_one_fixture_example(tmp_path: Path) -> None:
    output = tmp_path / "selected.run.json"

    exit_code = main(
        [
            "run",
            str(PROJECT_ROOT / "examples/configs/candidate.json"),
            "--example-id",
            "multiply",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == EXIT_OK
    assert payload["metadata"]["dataset"]["example_count"] == 1
    assert [item["example_id"] for item in payload["examples"]] == ["multiply"]


def test_cli_refuses_to_overwrite_raw_run_with_annotations() -> None:
    raw_run = PROJECT_ROOT / "examples/artifacts/candidate.run.json"

    exit_code = main(
        [
            "annotate",
            str(raw_run),
            "--output",
            str(raw_run),
            "--experiment",
            "judge-calibration-v1",
            "--annotator",
            "annotator-1",
        ]
    )

    assert exit_code == EXIT_ERROR
