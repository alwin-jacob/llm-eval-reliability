# Engineering notes

Technical findings, known limitations, and unresolved questions for v1.

## Correctness and provenance

- The engine binds a non-optional response before starting scorer work so a failed
  invocation cannot expose a `None` response to a scorer.
- Fixture-candidate provenance includes keyed responses, the default response, and usage
  data so behaviorally different fixtures do not share a configuration fingerprint.
- Candidate and scorer configuration is captured and JSON-validated exactly once per run.
  The same captured values are used for both the fingerprint and persisted metadata.
- Regression comparison requires identical scorer configuration by default. Policies can
  gate measured coverage as well as aggregate score.
- Slice `min_examples` counts examples with a measured scorer result rather than every
  example assigned to the slice.
- Dataset, invocation, engine, scorer, and regression-policy scalar types are checked
  explicitly instead of relying on Python coercion.
- Configured datasets retain the logical source path from the configuration. Resolution
  still uses an absolute path confined to the configured base directory.

## Judge calibration

- `reference_label` and `human_label` are excluded from judge prompts by default to prevent
  calibration-label leakage. The exclusion list is configurable.
- Judge results retain the raw judge output, attempt, model, provider, and usage evidence.
- The checked-in calibration example uses a scripted judge and synthetic specification
  labels because no provider model or collected human annotation is available locally. Its
  agreement result exercises the analysis code; it is not evidence about model-judge or
  human agreement.
- Slice metrics are descriptive and include sample counts. V1 does not provide confidence
  intervals or make significance claims from the small example datasets.

## Known limitations

- Completed examples remain in memory until the final artifact is written atomically. A
  process crash therefore loses in-progress results.
- Evaluation concurrency is asynchronous but confined to one process.
- V1 records one candidate response per example; it does not estimate within-example
  stochastic variance from repeated samples.
- Python 3.14 surfaces a `pytest-asyncio` event-loop-policy deprecation outside the supported
  Python 3.11–3.13 CI matrix. A narrow warning filter remains until the dependency removes
  the compatibility warning.
- Claude CLI authentication is inherited from the local CLI session. The adapter does not
  provide credential management or an Anthropic API transport.
- Claude CLI `costUSD` values are retained as provider usage accounting, not interpreted as
  proof of a separate bill. TTFT remains absent when the CLI envelope does not report it.

## Open engineering questions

1. How should append-only per-example checkpoints verify resume compatibility?
2. How should multiple annotators and adjudication replace the single reference label?
3. Which bootstrap intervals and minimum-power guidance should precede default slice gates?
