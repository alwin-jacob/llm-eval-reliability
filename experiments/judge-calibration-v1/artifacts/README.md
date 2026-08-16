# Experiment artifacts

`candidate-haiku.run.json` and `candidate-haiku.summary.json` record the completed 16-case
Haiku candidate phase. The earlier `smoke-haiku-structured-01` pair records the single setup
call and is not part of that 16-case sample. `judge-sonnet.run.json` preserves 16 raw judge
outputs and provider metadata; one is invalid JSON and remains unmodified.

`judge-sonnet.summary.json` is the standard replay-run summary. Its top-level usage describes
the replayed Haiku response objects, not the Sonnet scorer calls. The true judge-side tokens,
provider usage accounting, timing, and agreement against the assisted/operator reference are
derived from raw scorer evidence in `judge-sonnet.analysis.json`. The analysis artifact uses
legacy `human` field names from the annotation schema; they do not imply independent human
calibration.

Raw run artifacts are immutable inputs to annotation and judge replay. Derived
reports and any replication must use distinct filenames rather than overwrite these files.
