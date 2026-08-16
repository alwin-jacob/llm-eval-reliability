"""Small local FastAPI surface over the same evaluation and comparison core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from evalreliability import __version__
from evalreliability.analysis import summarize_run
from evalreliability.artifacts import read_run, write_run
from evalreliability.config import load_evaluation_spec
from evalreliability.engine import EvaluationEngine
from evalreliability.errors import EvalReliabilityError
from evalreliability.regression import RegressionPolicy, compare_runs


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: str
    artifact_path: str | None = None


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_path: str
    candidate_path: str
    policy: dict[str, Any]


def create_app(*, allowed_root: str | Path | None = None) -> FastAPI:
    root = Path(allowed_root or Path.cwd()).resolve()
    app = FastAPI(
        title="evalreliability",
        version=__version__,
        description="Local, synchronous API for the failure-aware evaluation core.",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/evaluations")
    async def evaluate(request: EvaluationRequest) -> dict[str, Any]:
        try:
            config_path = _confined(request.config_path, root)
            spec = load_evaluation_spec(config_path, allowed_root=root, allow_python_plugins=False)
            run = await EvaluationEngine().evaluate(
                spec.dataset, spec.candidate, spec.scorers, spec.evaluation
            )
            if request.artifact_path is not None:
                write_run(run, _confined(request.artifact_path, root))
            return {"run": run.to_dict(), "summary": summarize_run(run)}
        except (EvalReliabilityError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/comparisons")
    async def compare(request: ComparisonRequest) -> dict[str, Any]:
        try:
            baseline = read_run(_confined(request.baseline_path, root))
            candidate = read_run(_confined(request.candidate_path, root))
            policy = RegressionPolicy.from_dict(request.policy)
            return compare_runs(baseline, candidate, policy)
        except (EvalReliabilityError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _confined(value: str, root: Path) -> Path:
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes allowed root {root}")
    return resolved


app = create_app()
