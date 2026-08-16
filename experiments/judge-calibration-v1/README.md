# judge-calibration-v1

Research question: **How reliably can an LLM judge detect subtle instruction-following
failures, and where does it disagree with a human reference?**

This experiment contains 16 project-authored prompts: four each in `structured_output`,
`exact_constraints`, `multi_constraint`, and `semantic_constraint`. The tasks avoid factual
recall and can be evaluated from the prompt and response. Every case has a case-specific
rubric. Human-reference labels are currently null, and no judge result is claimed.

The Haiku candidate uses the clean Claude Code CLI transport with tools disabled, a minimal
system prompt, an empty temporary working directory, one turn, concurrency one, and no
automatic retry. The structured, exact, and multi-constraint slices have deterministic
checks where their requirements can be measured mechanically. These checks do not replace
the human rubric, especially where semantic adequacy is also required.

## Files and execution order

- `dataset/v1/`: versioned 16-case dataset and rubrics.
- `configs/candidate-haiku.json`: full candidate run; not executed during experiment setup.
- `artifacts/smoke-haiku-structured-01.run.json`: actual one-case setup smoke evidence.
- `artifacts/smoke-haiku-structured-01.summary.json`: derived smoke summary.
- `artifacts/candidate-haiku.run.json`: future immutable raw candidate responses.
- `annotations/human-reference.json`: future single-annotator labels, stored separately.
- `configs/future-judge-sonnet.json`: future Sonnet judge over replayed Haiku outputs.
- `artifacts/judge-sonnet.run.json`: future judge evidence aligned to the frozen responses.

Run the full candidate collection only when ready to make 16 Haiku calls:

```bash
evalrel run experiments/judge-calibration-v1/configs/candidate-haiku.json \
  --output experiments/judge-calibration-v1/artifacts/candidate-haiku.run.json \
  --summary-output experiments/judge-calibration-v1/artifacts/candidate-haiku.summary.json
```

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

## Human annotation

After the full Haiku run, start or resume one-annotator labeling:

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
