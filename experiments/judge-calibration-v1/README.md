# judge-calibration-v1

Research question: **How reliably can an LLM judge detect subtle instruction-following
failures, and where does it disagree with an assisted/operator reference?**

This experiment contains 16 project-authored prompts: four each in `structured_output`,
`exact_constraints`, `multi_constraint`, and `semantic_constraint`. The tasks avoid factual
recall and can be evaluated from the prompt and response. Every case has a case-specific
rubric. The frozen responses now have 16 assisted/operator labels and 16 blinded Sonnet
judge attempts. The labels were entered with AI assistance under operator supervision.
They are not independent human annotations, human ground truth, adjudication, or
multi-annotator consensus.

The frozen dataset manifest records the pre-label design state (`unlabeled`, zero
annotators) and the originally planned workflow. It is part of the committed dataset
checksum and is intentionally not rewritten after collection. The provenance correction in
this README and `experiment.json` describes how the reference labels were actually entered.

The Haiku candidate uses the clean Claude Code CLI transport with tools disabled, a minimal
system prompt, an empty temporary working directory, one turn, conservative concurrency
two, and no automatic retry. The structured, exact, and multi-constraint slices have
deterministic checks where their requirements can be measured mechanically. These checks do
not replace rubric-based reference evaluation, especially where semantic adequacy is also
required.

## Files and execution order

- `dataset/v1/`: versioned 16-case dataset and rubrics.
- `configs/candidate-haiku.json`: configuration used for the full candidate run.
- `artifacts/smoke-haiku-structured-01.run.json`: actual one-case setup smoke evidence.
- `artifacts/smoke-haiku-structured-01.summary.json`: derived smoke summary.
- `artifacts/candidate-haiku.run.json`: frozen raw responses from the 16-case candidate run.
- `artifacts/candidate-haiku.summary.json`: derived candidate-run summary.
- `annotations/human-reference.json`: legacy filename for completed assisted/operator labels,
  stored separately from model outputs.
- `configs/future-judge-sonnet.json`: configuration used to judge replayed Haiku outputs.
- `artifacts/judge-sonnet.run.json`: raw Sonnet evidence aligned to the frozen responses.
- `artifacts/judge-sonnet.summary.json`: replay-run status summary.
- `artifacts/judge-sonnet.analysis.json`: human, judge, deterministic, slice, and true
  judge-execution summaries.

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
reference label or judge result. The raw artifact records the configuration, environment,
response, usage, timing, and limitations implied by a sample size of one.

## Candidate-generation result

The full run started on 2026-08-16 UTC and made exactly one Haiku call for each of the 16
committed examples. It used dataset version `1.0.0` (checksum
`d7be3a8cae585cdd50a77b204ccdf7a47ab66eef03dcbcdc956cda69078450ff`), configuration
fingerprint `3d9596d782e493a27b83954cbc74a852bbb8f3a08971cd9a9e8380b02b81fc04`,
requested alias `haiku`, and returned canonical model `claude-haiku-4-5-20251001` for every
response. All 16 transports completed with no retry. At candidate-collection time, no
reference labels or judge calls existed.

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
reference-labeling process and are not assigned labels here.

Across 16 responses, the CLI reported 3,309 input tokens, 5,377 output tokens, no cache
tokens, and USD 0.040319 of provider usage accounting. That cost field is not evidence of a
separate charge to the user. End-to-end response latency had mean 4,485.555 ms, p50 3,551.915
ms, p95 11,813.158 ms, and range 1,983.350–11,813.158 ms. Reported TTFT was available for
all 16 responses, with mean 3,502.938 ms and range 1,271–10,899 ms. These are measurements of
this small run, not general reliability or performance claims.

## Assisted/operator reference

All 16 frozen responses received 13 PASS and 3 FAIL reference labels. Label entry used AI
assistance under operator supervision, so this is not an independently produced human
calibration set. This command was used for the resumable annotation workflow:

```bash
evalrel annotate experiments/judge-calibration-v1/artifacts/candidate-haiku.run.json \
  --output experiments/judge-calibration-v1/annotations/human-reference.json \
  --experiment judge-calibration-v1 \
  --annotator annotator-1
```

For each case the command prints the prompt, frozen candidate response, and rubric, then
accepts `PASS` or `FAIL` and an optional short note. It atomically checkpoints after every
label. The annotation artifact contains labels and source-run identity but does not copy or
overwrite model responses. Its legacy schema identifies a single annotator and disclaims
ground truth, adjudication, and multi-annotator consensus; that schema metadata does not
imply independent human provenance. Re-running the command is unnecessary for the frozen
experiment.

## Blinded Sonnet judge result

The judge run replayed the committed Haiku responses; it did not call Haiku. The prompt
payload contained only the original prompt, frozen candidate output, and case rubric.
Reference labels, annotation notes, deterministic scores, and other expected fields were not
included.
The requested model was `sonnet`, the canonical returned model was `claude-sonnet-5`, tools
and dynamic system-prompt sections were disabled, concurrency was two, and each case had one
attempt. The 16-call run used:

```bash
evalrel run experiments/judge-calibration-v1/configs/future-judge-sonnet.json \
  --output experiments/judge-calibration-v1/artifacts/judge-sonnet.run.json \
  --summary-output experiments/judge-calibration-v1/artifacts/judge-sonnet.summary.json
```

All 16 CLI calls returned raw judge output. Fifteen satisfied the strict JSON contract: 12
PASS and 3 FAIL. The assisted/operator reference marks `jcv1-structured-02` PASS, but the
judge emitted unescaped quotation marks inside its rationale, making the outer JSON
malformed. It remains an invalid verdict; it was not repaired or retried. On the 15 parseable
assisted-reference/judge pairs, accuracy and Cohen's kappa were both 1.0, with no false
positives, false negatives, or disagreements. The missing pair is reported separately rather
than treated as agreement or disagreement.

| Slice | Experiment n | Paired n | Assisted reference PASS/FAIL | Judge PASS/FAIL | Accuracy | Kappa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `structured_output` | 4 | 3 | 1/3 | 0/3 | 1.0 | undefined |
| `exact_constraints` | 4 | 4 | 4/0 | 4/0 | 1.0 | undefined |
| `multi_constraint` | 4 | 4 | 4/0 | 4/0 | 1.0 | undefined |
| `semantic_constraint` | 4 | 4 | 4/0 | 4/0 | 1.0 | undefined |

Kappa is undefined within each slice because the paired labels contain only one class. The
deterministic scorers also matched the assisted/operator reference on every case they
measured:
`json_schema` n=4 (kappa 1.0), `json_fields` n=3, `exact_constraints` n=4, and
`text_constraints` n=4. Kappa is undefined for the latter three single-class comparisons.

Judge calls reported 7,291 input tokens, 2,294 output tokens, 9,585 total tokens, no cache
tokens, and USD 0.068678 of provider usage accounting. This is not proof of a separate bill.
Judge-scorer latency across n=16 had mean 3,105.520 ms, p50 2,922.252 ms, p95 4,066.091 ms,
and range 2,416.330–4,066.091 ms. TTFT was available for all calls, with mean 2,094.313 ms
and range 1,685–2,540 ms. Although every requested and canonical response model was Sonnet,
the CLI usage metadata listed both Sonnet and Haiku model IDs. The usage totals are therefore
reported as CLI-level accounting and are not attributed exclusively to the response model.

The valid structured-output rationales explicitly rejected Markdown fences even when their
contents were semantically correct, showing no formatting tolerance on those three cases.
The valid semantic rationales focused on the semantic requirements their rubrics specified.
No clear rubric misinterpretation or inconsistent within-slice strictness appears in the
valid rationales. The malformed PASS output is an output-serialization reliability failure,
not evidence of disagreement. These observations are descriptive; the slices are too small
and class-imbalanced to support comparative strictness claims.

The committed analysis was generated without further model calls:

```bash
evalrel annotation-report \
  --candidate-run experiments/judge-calibration-v1/artifacts/candidate-haiku.run.json \
  --annotations experiments/judge-calibration-v1/annotations/human-reference.json \
  --judge-run experiments/judge-calibration-v1/artifacts/judge-sonnet.run.json \
  --judge-scorer sonnet_instruction_judge \
  --output experiments/judge-calibration-v1/artifacts/judge-sonnet.analysis.json
```

## Interpretation limits

Sixteen cases and an AI-assisted/operator reference can expose concrete failure modes but
cannot establish human-calibrated or general judge reliability. Case construction is
project-authored, label entry was not independent, label order is not blinded by tooling,
and no repeated candidate or judge samples are planned in v1. Any later result must record
the actual model IDs, run artifacts, annotation metadata, sample sizes, failures, and
limitations.
