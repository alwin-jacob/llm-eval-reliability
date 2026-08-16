from __future__ import annotations

import json
from pathlib import Path

import pytest

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
