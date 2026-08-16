# Experiment artifacts

`controlled-responses.run.json` and its summary are deterministic materializations of the 24
project-authored challenge responses. Their provider metadata identifies a controlled
experiment fixture and records that no model call occurred. They are not candidate-model
benchmark outputs.

`judge-sonnet.run.json` preserves the 24 raw Sonnet judgments and scorer-side provider
metadata. `judge-sonnet.summary.json` is the standard replay summary; its top-level candidate
usage is empty because the candidate responses were replayed. Actual judge tokens, provider
usage accounting, latency, TTFT, invalid-output accounting, agreement, construction-label
validation, and per-family results are derived from scorer evidence in
`judge-sonnet.analysis.json`.

The controlled-response artifact and human annotation artifact are immutable inputs. Any
replication must use distinct output filenames rather than overwrite the frozen run.
