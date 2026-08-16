from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from evalreliability.api import create_app

PROJECT_ROOT = Path(__file__).parents[1]


@pytest.mark.asyncio
async def test_api_health_and_local_evaluation() -> None:
    transport = httpx.ASGITransport(app=create_app(allowed_root=PROJECT_ROOT))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        response = await client.post(
            "/v1/evaluations",
            json={"config_path": "examples/configs/candidate.json"},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["counts"]["total"] == 6
    assert payload["summary"]["counts"]["failed"] == 0
    assert payload["run"]["metadata"]["candidate_id"] == "candidate-fixture-v2"


@pytest.mark.asyncio
async def test_api_rejects_paths_outside_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    transport = httpx.ASGITransport(app=create_app(allowed_root=allowed))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/evaluations",
            json={"config_path": str(PROJECT_ROOT / "examples/configs/candidate.json")},
        )

    assert response.status_code == 400
    assert "escapes allowed root" in response.json()["detail"]


@pytest.mark.asyncio
async def test_api_compares_checked_in_runs() -> None:
    policy = json.loads(
        (PROJECT_ROOT / "examples/configs/regression-policy.json").read_text(encoding="utf-8")
    )
    transport = httpx.ASGITransport(app=create_app(allowed_root=PROJECT_ROOT))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/comparisons",
            json={
                "baseline_path": "examples/artifacts/baseline.run.json",
                "candidate_path": "examples/artifacts/candidate.run.json",
                "policy": policy,
            },
        )

    assert response.status_code == 200
    assert response.json()["passed"] is True
