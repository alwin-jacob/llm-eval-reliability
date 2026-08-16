# Contributing

Contributions should preserve the central invariant: a partial or failed evaluation must
remain inspectable rather than disappearing into aggregate metrics.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
mypy src
pytest -q
```

## Change expectations

- Add tests for both success and failure semantics. Candidate adapters should cover typed
  provider errors, unexpected exceptions, timeout/cancellation, and usage normalization.
- Treat artifact fields as a compatibility surface. Schema changes require a version bump,
  reader tests, and a migration or explicit incompatibility note.
- New scorers must distinguish quality failure (`FAIL`) from inability to evaluate
  (`EvaluatorError`). Report skipped applicability rather than silently assigning zero.
- Do not put credentials, API keys, raw authorization headers, or secret factory arguments
  in configuration provenance.
- Benchmark or judge-quality claims must include dataset/version/checksum, provider/model,
  configuration, environment, sample size, measurements, and limitations. Synthetic or
  scripted evidence must be labeled as such.
- Update `docs/design.md` when changing an architectural contract and append relevant
  discoveries or rejected assumptions to `docs/engineering-log.md`.

Keep commits coherent and substantive. A change is ready when the four local quality gates
pass and a reviewer can explain why its failure behavior is trustworthy.

