# Contributing

Contributions should preserve the central invariant: partial and failed evaluations remain inspectable rather than disappearing into aggregate metrics.

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

* Add tests for both success and failure behavior. Candidate adapters should cover typed provider failures, unexpected exceptions, timeout and cancellation behavior, and usage normalization.
* Treat artifact fields as a compatibility surface. Schema changes require an appropriate version change together with reader coverage and an explicit compatibility or migration decision.
* New scorers must distinguish a quality failure (`FAIL`) from inability to evaluate (`EvaluatorError`). When a scorer does not apply, report that explicitly rather than assigning an artificial zero.
* Keep credentials, API keys, authorization-header values, and other secrets out of configuration provenance and persisted artifacts.
* Report benchmark or judge-quality results with enough provenance to interpret them: dataset identity, provider and model, configuration, environment, sample size, measurements, and relevant limitations.
* Identify synthetic, scripted, controlled, or replayed evidence accurately.
* Update `docs/design.md` when changing an architectural contract.
* Keep experiment-specific methodology and provenance with the corresponding experiment artifacts and documentation.

Keep commits coherent and substantive.

A change is ready when the local quality gates pass and its success, failure, and artifact behavior can be inspected from the resulting implementation and tests.
