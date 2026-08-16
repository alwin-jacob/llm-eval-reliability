# judge-challenge-v2

This is a controlled judge challenge set for testing whether a judge detects subtle
instruction-following near misses. It is not a benchmark of candidate-model quality. All 24
responses are project-authored experiment material; none is presented as an organically
generated model response.

The design has 12 underlying tasks with two frozen variants each. One variant was authored
to satisfy the shared rubric and one was authored to violate exactly one important
requirement while remaining fluent and mostly correct. The 12/12 construction balance is a
design invariant, not a human reference. Independent annotation by Alwin remains required
before any agreement result can be reported.

## Pairs

| Pair ID | Challenge family | Controlled difference |
| --- | --- | --- |
| `jcv2-pair-01` | `permission_vs_obligation` | Preserves discretionary “may” versus silently changing it to mandatory “must.” |
| `jcv2-pair-02` | `uncertainty_overclaim` | Keeps “likely” deployment status versus asserting canary coverage as certain. |
| `jcv2-pair-03` | `causal_overclaim` | Reports a post-email decline without causation versus claiming the email caused it. |
| `jcv2-pair-04` | `exception_scope` | Protects active investigations versus subtly substituting completed investigations. |
| `jcv2-pair-05` | `quantifier_shift` | Keeps automatic retry at “most” integrations versus broadening it to “every.” |
| `jcv2-pair-06` | `numerical_boundary` | Preserves strictly below $10,000 versus including the $10,000 boundary. |
| `jcv2-pair-07` | `negation_scope` | Keeps external backups always encrypted versus extending the test exception externally. |
| `jcv2-pair-08` | `conditionality` | Requires both approvals before access versus allowing security approval afterward. |
| `jcv2-pair-09` | `unsupported_detail` | Retains an unknown cause versus inventing a plausible storage-quota diagnosis. |
| `jcv2-pair-10` | `instruction_priority` | Avoids unverified timing versus promising Friday delivery despite the explicit prohibition. |
| `jcv2-pair-11` | `semantic_preservation` | Preserves per-workspace enablement versus changing the control to organization-wide only. |
| `jcv2-pair-12` | `multi_constraint_near_miss` | Keeps the rollback plan untested versus changing only that status to tested. |

Within each pair, the original prompt and detailed rubric are identical. Case IDs and
variant identifiers use neutral `a` and `b` values, and compliant placement alternates, so
position does not reveal the construction intent.

## Files and provenance

- `dataset/v1/data.jsonl` is the authoritative 24-case challenge set. Each record includes
  the case ID, pair ID, challenge family, original prompt, frozen response, rubric, neutral
  variant ID, and construction metadata that is private from the judge prompt.
- `configs/materialize-controlled-responses.json` materializes those authored responses into
  a standard run artifact without calling a model. Its sole scorer is intentionally skipped;
  it makes no quality judgment.
- `artifacts/controlled-responses.run.json` is the frozen annotation and future-judge input.
- `artifacts/controlled-responses.summary.json` confirms all 24 authored responses were
  materialized and all quality-scoring results were skipped.
- `configs/judge-sonnet.json` replays that artifact and is present for a future opt-in run.
- `annotations/human-reference.json` does not exist until independent annotation begins.

The dataset checksum is
`a29d5d5d7308c2bf2296fb8ae5407c97daeacd2f86811c10ac7eeb253ca49a6f`.
Construction metadata is never interpreted as a human label.

## Human annotation

Start or resume the single-annotator workflow with:

```bash
evalrel annotate experiments/judge-challenge-v2/artifacts/controlled-responses.run.json \
  --output experiments/judge-challenge-v2/annotations/human-reference.json \
  --experiment judge-challenge-v2 \
  --annotator alwin
```

The workflow displays one neutral case ID, prompt, frozen response, and rubric at a time. It
does not display explicit pair metadata, the sibling response, or the construction label.
Labels are checkpointed separately and do not modify the frozen response artifact.

## Future blinded judge run

No Sonnet call has been made for this experiment. When intentionally run later, the
configuration makes one clean, tool-free Sonnet invocation per frozen response, with one
attempt and concurrency two. The runtime prompt payload contains only `input`,
`candidate_output`, and `rubric`. It contains no case or pair identifier, variant metadata,
construction intent, human annotation, deterministic result, or sibling response.

The future command is intentionally not executed during challenge-set construction:

```bash
evalrel run experiments/judge-challenge-v2/configs/judge-sonnet.json \
  --output experiments/judge-challenge-v2/artifacts/judge-sonnet.run.json \
  --summary-output experiments/judge-challenge-v2/artifacts/judge-sonnet.summary.json
```

After independent annotation and the future judge run, generate analysis with:

```bash
evalrel annotation-report \
  --candidate-run experiments/judge-challenge-v2/artifacts/controlled-responses.run.json \
  --annotations experiments/judge-challenge-v2/annotations/human-reference.json \
  --judge-run experiments/judge-challenge-v2/artifacts/judge-sonnet.run.json \
  --judge-scorer sonnet_challenge_judge \
  --slice-dimension challenge_family \
  --output experiments/judge-challenge-v2/artifacts/judge-sonnet.analysis.json
```

The report supports paired-sample accuracy, Cohen's kappa, confusion counts, false positives
and negatives, accuracy conditioned on human PASS and FAIL, per-family accuracy with sample
sizes, invalid-judge-output accounting, disagreement rationales, and raw invalid-output
evidence. It uses only the independent human annotation artifact as the reference.

## Limits

The challenge set is deliberately balanced and contrastive, so its label distribution is
not representative of natural model traffic. Paired authorship may make distinctions cleaner
than organic failures. A single future judge sample per case cannot estimate stochastic
stability, and one annotator is not ground truth or consensus.
The fixed annotation order and repeated prompts can still allow a human annotator to
recognize paired cases; v2 does not claim that the human workflow is pair-blinded.
