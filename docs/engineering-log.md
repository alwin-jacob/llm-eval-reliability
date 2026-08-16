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
- Exclusion lists were insufficient for a blinded calibration prompt because other expected
  fields could still expose deterministic answers. Rubric-key mode now constructs a payload
  containing only the original input, frozen candidate output, and case rubric.
- Judge results retain the raw judge output, attempt, model, provider, usage, and
  provider-specific metadata evidence.
- The real Sonnet calibration run returned one syntactically invalid judgment: quotation
  marks copied into a rationale were not escaped in the outer JSON. The strict binary
  contract preserved it as `judge_invalid_json`; it was neither repaired nor retried.
- Standard replay-run summaries attribute top-level usage to the replayed candidate response.
  Human-reference analysis therefore derives judge tokens, cost accounting, TTFT, and
  latency separately from scorer evidence.
- The Sonnet run's CLI usage metadata listed both Sonnet and Haiku model IDs even though the
  requested and canonical response model was Sonnet. Aggregate CLI usage is retained without
  assigning every token or cost unit exclusively to the response model.
- Controlled judge challenge sets use a distinct candidate adapter that reads authored
  responses from versioned dataset metadata and records `model_calls=false`. This prevents
  controlled near misses from being represented as organic model outputs.
- Rubric-key judge prompts omit all example metadata, which also prevents pair IDs,
  construction intent, and sibling responses from entering paired challenge judgments.
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
- Human-reference labels are checkpointed separately from raw run artifacts. The annotation
  artifact binds to the exact run and dataset checksum and explicitly represents one
  annotator rather than consensus or ground truth.
- A later judge must score replayed candidate responses. Regenerating the candidate during
  judge execution would confound judge disagreement with candidate stochasticity.

## Open engineering questions

1. How should append-only per-example checkpoints verify resume compatibility?
2. How should multiple annotators and adjudication replace the single reference label?
3. Which bootstrap intervals and minimum-power guidance should precede default slice gates?
