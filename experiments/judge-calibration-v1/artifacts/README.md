# Experiment artifacts

`candidate-haiku.run.json` and `candidate-haiku.summary.json` record the completed 16-case
Haiku candidate phase. The earlier `smoke-haiku-structured-01` pair records the single setup
call and is not part of that 16-case sample. No human annotation or judge artifact exists.

Raw run artifacts are immutable inputs to annotation and future judge replay. Derived
reports and any replication must use distinct filenames rather than overwrite these files.
