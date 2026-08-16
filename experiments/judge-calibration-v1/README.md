# judge-calibration-v1

Research question: **How reliably can an LLM judge detect subtle instruction-following
failures, and where does it disagree with a human reference?**

This experiment contains 16 project-authored prompts: four each in `structured_output`,
`exact_constraints`, `multi_constraint`, and `semantic_constraint`. The tasks avoid factual
recall and can be evaluated from the prompt and response. Every case has a case-specific
rubric. Human-reference labels are currently null, and no judge result is claimed.

The Haiku candidate uses the clean Claude Code CLI transport with tools disabled, a minimal
system prompt, an empty temporary working directory, one turn, conservative concurrency
two, and no automatic retry. The structured, exact, and multi-constraint slices have
deterministic checks where their requirements can be measured mechanically. These checks do
not replace the human rubric, especially where semantic adequacy is also required.

## Files and execution order

- `dataset/v1/`: versioned 16-case dataset and rubrics.
- `configs/candidate-haiku.json`: configuration used for the full candidate run.
- `artifacts/smoke-haiku-structured-01.run.json`: actual one-case setup smoke evidence.
- `artifacts/smoke-haiku-structured-01.summary.json`: derived smoke summary.
- `artifacts/candidate-haiku.run.json`: frozen raw responses from the 16-case candidate run.
- `artifacts/candidate-haiku.summary.json`: derived candidate-run summary.
- `annotations/human-reference.json`: future single-annotator labels, stored separately.
- `configs/future-judge-sonnet.json`: future Sonnet judge over replayed Haiku outputs.
- `artifacts/judge-sonnet.run.json`: future judge evidence aligned to the frozen responses.

The full candidate collection was executed exactly once with this command:

```bash
evalrel run experiments/judge-calibration-v1/configs/candidate-haiku.json \
  --output experiments/judge-calibration-v1/artifacts/candidate-haiku.run.json \
  --summary-output experiments/judge-calibration-v1/artifacts/candidate-haiku.summary.json
```

Re-running it would perform another 16 real-model calls and replace the named artifacts;
use a distinct output path for any replication.

The setup smoke check selects one case and therefore makes one call without changing the
versioned source dataset:

```bash
evalrel run experiments/judge-calibration-v1/configs/candidate-haiku.json \
  --example-id jcv1-structured-01 \
  --output /tmp/judge-calibration-v1-haiku-smoke.run.json \
  --summary-output /tmp/judge-calibration-v1-haiku-smoke.summary.json
```

The 2026-08-16 UTC setup run made exactly one Haiku call on `jcv1-structured-01`.
`claude-haiku-4-5-20251001` returned the requested keys and values but wrapped the object in
a Markdown JSON fence. The candidate call completed, while both strict JSON scorers failed
with `malformed_json`. This is retained as a negative instruction-following result, not a
human label or judge result. The raw artifact records the configuration, environment,
response, usage, timing, and limitations implied by a sample size of one.

## Candidate-generation result

The full run started on 2026-08-16 UTC and made exactly one Haiku call for each of the 16
committed examples. It used dataset version `1.0.0` (checksum
`d7be3a8cae585cdd50a77b204ccdf7a47ab66eef03dcbcdc956cda69078450ff`), configuration
fingerprint `3d9596d782e493a27b83954cbc74a852bbb8f3a08971cd9a9e8380b02b81fc04`,
requested alias `haiku`, and returned canonical model `claude-haiku-4-5-20251001` for every
response. All 16 transports completed with no retry; no human labels or judge calls exist.

Case-level deterministic outcomes count a case as failed if any applicable deterministic
scorer failed. Cases with no applicable scorer remain unmeasured:

| Slice | n | Pass | Fail | Unmeasured |
| --- | ---: | ---: | ---: | ---: |
| `structured_output` | 4 | 1 | 3 | 0 |
| `exact_constraints` | 4 | 4 | 0 | 0 |
| `multi_constraint` | 4 | 4 | 0 | 0 |
| `semantic_constraint` | 4 | 0 | 0 | 4 |

The three structured failures (`jcv1-structured-01`, `-03`, and `-04`) contain the requested
JSON inside Markdown fences. The raw fences are preserved, so both strict JSON scorers emit
`malformed_json`; the six failure records refer to three unique responses. The remaining
structured response is a raw JSON array and passes its schema check. Some semantic responses
also add headings, quotations, or explanatory prose. Those details are retained for the
annotator and are not assigned labels here.

Across 16 responses, the CLI reported 3,309 input tokens, 5,377 output tokens, no cache
tokens, and USD 0.040319 of provider usage accounting. That cost field is not evidence of a
separate charge to the user. End-to-end response latency had mean 4,485.555 ms, p50 3,551.915
ms, p95 11,813.158 ms, and range 1,983.350–11,813.158 ms. Reported TTFT was available for
all 16 responses, with mean 3,502.938 ms and range 1,271–10,899 ms. These are measurements of
this small run, not general reliability or performance claims.

## Human annotation

The frozen candidate run is ready for one-annotator labeling. Start or resume it with:

```bash
evalrel annotate experiments/judge-calibration-v1/artifacts/candidate-haiku.run.json \
  --output experiments/judge-calibration-v1/annotations/human-reference.json \
  --experiment judge-calibration-v1 \
  --annotator annotator-1
```

For each case the command prints the prompt, frozen candidate response, and rubric, then
accepts `PASS` or `FAIL` and an optional short note. It atomically checkpoints after every
label. The annotation artifact contains labels and source-run identity but does not copy or
overwrite model responses. Its metadata states that the reference comes from one annotator,
is not ground truth, is not adjudicated, and is not multi-annotator consensus.

Create a human/deterministic report before running a judge:

```bash
evalrel annotation-report \
  --candidate-run experiments/judge-calibration-v1/artifacts/candidate-haiku.run.json \
  --annotations experiments/judge-calibration-v1/annotations/human-reference.json \
  --output experiments/judge-calibration-v1/artifacts/human-reference.report.json
```

The future judge config replays the exact Haiku responses from the raw run, then makes one
Sonnet call per case. It excludes human-reference fields from judge prompts. After that run,
add `--judge-run .../judge-sonnet.run.json --judge-scorer sonnet_instruction_judge` to the
report command. The report includes human and judge pass rates, paired accuracy, Cohen's
kappa when defined, false positives, false negatives, disagreements, per-slice agreement,
deterministic-scorer agreement, and explicit sample sizes.

## Interpretation limits

Sixteen cases and one reference annotator can expose concrete disagreement modes but cannot
establish general judge reliability. Case construction is project-authored, label order is
not blinded by tooling, and no repeated candidate or judge samples are planned in v1. Any
later result must record the actual model IDs, run artifacts, annotation metadata, sample
sizes, failures, and limitations.
