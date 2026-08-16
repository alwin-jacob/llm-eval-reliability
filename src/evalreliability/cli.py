"""Command-line interface for evaluation, analysis, and regression gates."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evalreliability.analysis import analyze_judge_agreement, summarize_run
from evalreliability.artifacts import read_run, write_json, write_run
from evalreliability.config import load_evaluation_spec, load_regression_policy
from evalreliability.dataset import load_dataset
from evalreliability.engine import EvaluationEngine
from evalreliability.errors import EvalReliabilityError
from evalreliability.regression import compare_runs

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REGRESSION = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalrel",
        description="Failure-aware evaluation for stochastic model and agent behavior.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-dataset", help="validate and identify a dataset")
    validate.add_argument("dataset", type=Path)

    run = subparsers.add_parser("run", help="execute an evaluation config")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--summary-output", type=Path)

    summary = subparsers.add_parser("summarize", help="derive metrics from a run artifact")
    summary.add_argument("run", type=Path)
    summary.add_argument("--output", type=Path)

    compare = subparsers.add_parser("compare", help="apply regression policy to two runs")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--policy", type=Path, required=True)
    compare.add_argument("--output", type=Path)

    agreement = subparsers.add_parser(
        "judge-agreement", help="compare judge verdicts with dataset reference labels"
    )
    agreement.add_argument("run", type=Path)
    agreement.add_argument("--scorer", required=True)
    agreement.add_argument("--reference-key", default="reference_label")
    agreement.add_argument("--output", type=Path)

    serve = subparsers.add_parser("serve", help="run the local FastAPI interface")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--allowed-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-dataset":
            dataset = load_dataset(args.dataset)
            _print_json(asdict(dataset.descriptor))
            return EXIT_OK
        if args.command == "run":
            return _run_command(args.config, args.output, args.summary_output)
        if args.command == "summarize":
            payload = summarize_run(read_run(args.run))
            _write_optional_and_print(payload, args.output)
            return EXIT_OK
        if args.command == "compare":
            report = compare_runs(
                read_run(args.baseline),
                read_run(args.candidate),
                load_regression_policy(args.policy),
            )
            _write_optional_and_print(report, args.output)
            return EXIT_OK if report["passed"] else EXIT_REGRESSION
        if args.command == "judge-agreement":
            payload = analyze_judge_agreement(
                read_run(args.run), args.scorer, reference_key=args.reference_key
            )
            _write_optional_and_print(payload, args.output)
            return EXIT_OK
        if args.command == "serve":
            return _serve(args.host, args.port, args.allowed_root)
        raise RuntimeError(f"unhandled command {args.command}")
    except (EvalReliabilityError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _run_command(config_path: Path, output: Path, summary_output: Path | None) -> int:
    spec = load_evaluation_spec(config_path)
    run = asyncio.run(
        EvaluationEngine().evaluate(spec.dataset, spec.candidate, spec.scorers, spec.evaluation)
    )
    artifact_path = write_run(run, output)
    summary = summarize_run(run)
    if summary_output is not None:
        write_json(summary, summary_output)
    _print_json(
        {
            "run_id": run.metadata.run_id,
            "artifact": str(artifact_path),
            "summary_artifact": str(summary_output) if summary_output else None,
            "counts": summary["counts"],
            "candidate_success_rate": summary["reliability"]["candidate_success_rate"],
        }
    )
    return EXIT_OK


def _write_optional_and_print(payload: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        write_json(payload, output)
    _print_json(payload)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _serve(host: str, port: int, allowed_root: Path) -> int:
    try:
        import uvicorn

        from evalreliability.api import create_app
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise RuntimeError("install evalreliability[api] to use the API") from exc
    uvicorn.run(create_app(allowed_root=allowed_root), host=host, port=port)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
