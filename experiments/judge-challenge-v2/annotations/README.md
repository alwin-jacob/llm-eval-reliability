# Independent human-reference annotation

`human-reference.json` contains Alwin's completed annotation pass over all 24 frozen
responses: 12 PASS and 12 FAIL with no missing labels. It remains separate from controlled
responses and authorial construction metadata. Its reference is one independent human
annotator, not ground truth, adjudication, or multi-annotator consensus.

The dataset manifest's `unlabeled` value records the dataset's frozen construction-time
state. Post-freeze annotations belong in this separate artifact and do not mutate the
controlled dataset or response run.
