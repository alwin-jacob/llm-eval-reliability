# judge-challenge-v2

This is a controlled judge challenge set for testing whether a judge detects subtle
instruction-following near misses. It is not a benchmark of candidate-model quality. All 24
responses are project-authored experiment material; none is presented as an organically
generated model response.

The design has 12 underlying tasks with two frozen variants each. One variant was authored
to satisfy the shared rubric and one was authored to violate exactly one important
requirement while remaining fluent and mostly correct. The 12/12 construction balance is a
design invariant, not a human reference. Alwin completed one independent annotation pass
over the frozen responses; those labels are neither ground truth nor multi-annotator
consensus.

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
- `artifacts/controlled-responses.run.json` is the frozen annotation and judge input.
- `configs/judge-sonnet.json` is the configuration used to replay the frozen artifact to the
  Sonnet judge.
- `annotations/human-reference.json` contains Alwin's 24 independent PASS/FAIL labels.
- `artifacts/judge-sonnet.run.json` preserves all 24 raw Sonnet judgments and provider
  metadata; its summary and derived analysis use adjacent filenames.

The dataset checksum is
`a29d5d5d7308c2bf2296fb8ae5407c97daeacd2f86811c10ac7eeb253ca49a6f`.
Construction metadata was excluded from the judge and is compared with the human labels
only as a separate dataset-validation check.

## Human annotation

The single-annotator workflow was completed with:

```bash
evalrel annotate experiments/judge-challenge-v2/artifacts/controlled-responses.run.json \
  --output experiments/judge-challenge-v2/annotations/human-reference.json \
  --experiment judge-challenge-v2 \
  --annotator alwin
```

The workflow displays one neutral case ID, prompt, frozen response, and rubric at a time. It
does not display explicit pair metadata, the sibling response, or the construction label.
Labels are checkpointed separately and do not modify the frozen response artifact.

The artifact identifies annotator `alwin`, annotator count one, and no ground-truth,
adjudication, or consensus claim. It contains 12 PASS and 12 FAIL labels with no missing
cases. The human labels match the authorial construction intent on 24/24 cases (accuracy
1.0, Cohen's kappa 1.0). That is a dataset-validation observation; construction intent was
not used as the Sonnet reference and is not called ground truth.

## Blinded Sonnet judge run

The completed run made one clean, tool-free Sonnet invocation per frozen response, with one
attempt and concurrency two. The runtime prompt payload contained only `input`,
`candidate_output`, and `rubric`. It contained no case or pair identifier, variant metadata,
construction intent, human annotation, deterministic result, or sibling response.

The exact command was:

```bash
evalrel run experiments/judge-challenge-v2/configs/judge-sonnet.json \
  --output experiments/judge-challenge-v2/artifacts/judge-sonnet.run.json \
  --summary-output experiments/judge-challenge-v2/artifacts/judge-sonnet.summary.json
```

The derived analysis was generated without additional model calls:

```bash
evalrel annotation-report \
  --candidate-run experiments/judge-challenge-v2/artifacts/controlled-responses.run.json \
  --annotations experiments/judge-challenge-v2/annotations/human-reference.json \
  --judge-run experiments/judge-challenge-v2/artifacts/judge-sonnet.run.json \
  --judge-scorer sonnet_challenge_judge \
  --slice-dimension challenge_family \
  --output experiments/judge-challenge-v2/artifacts/judge-sonnet.analysis.json
```

## Measured result

The run started at `2026-08-16T18:29:47.417082Z` and made exactly 24 Sonnet calls. All 24
returned a strict, parseable verdict on the first attempt: 12 PASS, 12 FAIL, and 0 INVALID.
Against Alwin's 24 labels, accuracy was 1.0 and Cohen's kappa was 1.0. The confusion matrix
was 12 human-PASS/judge-PASS, 12 human-FAIL/judge-FAIL, and zero off-diagonal cases. PASS-case
accuracy was 12/12; FAIL-case accuracy was 12/12. There were no false positives, false
negatives, invalid judgments, or disagreement case IDs.

| Challenge family | n | Human PASS/FAIL | Sonnet PASS/FAIL | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `permission_vs_obligation` | 2 | 1/1 | 1/1 | 1.0 |
| `uncertainty_overclaim` | 2 | 1/1 | 1/1 | 1.0 |
| `causal_overclaim` | 2 | 1/1 | 1/1 | 1.0 |
| `exception_scope` | 2 | 1/1 | 1/1 | 1.0 |
| `quantifier_shift` | 2 | 1/1 | 1/1 | 1.0 |
| `numerical_boundary` | 2 | 1/1 | 1/1 | 1.0 |
| `negation_scope` | 2 | 1/1 | 1/1 | 1.0 |
| `conditionality` | 2 | 1/1 | 1/1 | 1.0 |
| `unsupported_detail` | 2 | 1/1 | 1/1 | 1.0 |
| `instruction_priority` | 2 | 1/1 | 1/1 | 1.0 |
| `semantic_preservation` | 2 | 1/1 | 1/1 | 1.0 |
| `multi_constraint_near_miss` | 2 | 1/1 | 1/1 | 1.0 |

Every family contains only one PASS and one FAIL case. Family kappas are 1.0 but are not
useful estimates at n=2.

The Claude CLI reported 10,530 input tokens, 2,720 output tokens, 13,250 total tokens, zero
cache tokens, and USD 0.090481 of provider usage accounting across n=24. The cost field is
not evidence of a separate bill. Judge-scorer latency had mean 3,309.404 ms, p50 3,087.629
ms, p95 5,089.714 ms, and range 2,719.115–5,454.474 ms. TTFT was available for all 24 calls,
with mean 2,392.667 ms, p50 2,206 ms, p95 3,210 ms, and range 1,806–3,856 ms. The complete
run elapsed 39,756.915 ms with concurrency two. The requested alias was `sonnet`; all 24
responses reported canonical model `claude-sonnet-5`.

All PASS rationales named the preserved condition, and all FAIL rationales named the changed
semantic requirement. Because there were no disagreements or invalid outputs, the error and
disagreement-rationale lists are empty. The rationales do not reveal formatting tolerance,
semantic-over-format bias, rubric misinterpretation, or inconsistent family strictness in
this sample; the experiment deliberately excluded trivial formatting failures and has only
two cases per family, so absence of an observed error is not evidence that those failure
modes are absent generally.

## Limits

The challenge set is deliberately balanced and contrastive, so its label distribution is
not representative of natural model traffic. Paired authorship may make distinctions cleaner
than organic failures. One judgment per case cannot estimate stochastic stability, and one
independent human annotator is not ground truth or consensus. Twelve families with n=2 each
cannot support comparative family-level reliability claims. The fixed annotation order and
repeated prompts can still allow a human annotator to recognize paired cases; v2 does not
claim that the human workflow is pair-blinded.
