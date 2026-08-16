"""Typed errors used at reliability boundaries."""

from __future__ import annotations

from typing import Any

from evalreliability.models import FailureOrigin


class EvalReliabilityError(Exception):
    """Base exception for errors callers may handle intentionally."""


class DatasetValidationError(EvalReliabilityError):
    """The dataset cannot be evaluated safely."""


class ConfigurationError(EvalReliabilityError):
    """Configuration is invalid or references an unsupported component."""


class ArtifactError(EvalReliabilityError):
    """A run artifact cannot be read or written."""


class AnalysisError(EvalReliabilityError):
    """Run data is insufficient or inconsistent for the requested analysis."""


class CandidateError(EvalReliabilityError):
    """A candidate call failed in an expected, classifiable way."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool,
        origin: FailureOrigin = FailureOrigin.MODEL,
    ) -> None:
        if origin not in (FailureOrigin.MODEL, FailureOrigin.INFRASTRUCTURE):
            raise ValueError("candidate errors must be model or infrastructure failures")
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.origin = origin


class EvaluatorError(EvalReliabilityError):
    """A scorer or judge could not produce a trustworthy evaluation."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
