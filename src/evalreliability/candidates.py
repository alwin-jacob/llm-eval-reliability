"""Provider-independent candidate contract and deterministic local adapter."""

from __future__ import annotations

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from evalreliability.errors import CandidateError
from evalreliability.models import CandidateRequest, CandidateResponse, FailureOrigin, Usage


class Candidate(ABC):
    """An asynchronous model, agent, workflow, or other system under evaluation."""

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Stable, human-readable candidate/model identifier."""

    @abstractmethod
    def configuration(self) -> dict[str, Any]:
        """Return a JSON-safe, secret-free description for provenance."""

    @abstractmethod
    async def generate(self, request: CandidateRequest) -> CandidateResponse:
        """Generate one response or raise a typed CandidateError."""


class FixtureCandidate(Candidate):
    """Deterministic adapter for examples, CI, and adapter contract tests."""

    def __init__(
        self,
        identifier: str,
        responses: Mapping[str, str],
        *,
        default_response: str | None = None,
        delay_seconds: float = 0.0,
        usage_by_example: Mapping[str, Usage] | None = None,
    ) -> None:
        if not identifier:
            raise ValueError("fixture identifier cannot be empty")
        if delay_seconds < 0:
            raise ValueError("fixture delay_seconds cannot be negative")
        self._identifier = identifier
        self._responses = dict(responses)
        self._default_response = default_response
        self._delay_seconds = delay_seconds
        self._usage_by_example = dict(usage_by_example or {})
        canonical = json.dumps(
            {
                "responses": self._responses,
                "default_response": self._default_response,
                "usage_by_example": {
                    key: asdict(value) for key, value in self._usage_by_example.items()
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self._responses_sha256 = hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def identifier(self) -> str:
        return self._identifier

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "fixture",
            "identifier": self.identifier,
            "response_count": len(self._responses),
            "fixture_payload_sha256": self._responses_sha256,
            "delay_seconds": self._delay_seconds,
        }

    async def generate(self, request: CandidateRequest) -> CandidateResponse:
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        output = self._responses.get(request.example_id, self._default_response)
        if output is None:
            raise CandidateError(
                f"fixture has no response for example {request.example_id!r}",
                code="fixture_response_missing",
                retryable=False,
                origin=FailureOrigin.INFRASTRUCTURE,
            )
        return CandidateResponse(
            output=output,
            model_id=self.identifier,
            provider="fixture",
            finish_reason="stop",
            usage=self._usage_by_example.get(request.example_id, Usage()),
        )
