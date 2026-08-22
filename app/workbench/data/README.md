# Workbench teaching fixture

The public Workbench ships one frozen, entirely synthetic teaching fixture:

- `simcausal_train.csv`: 500 simulated rows and 12 variables; it contains no
  patient, participant, or astronaut records.
- `ground_truth_edges.csv`: the 27-edge answer-key DAG used only for evaluation.
- `true_total_effects.json`: the supplied teaching answer key for total effects
  on `Outcome`.

The original generator and its complete data-generating specification were not
preserved with this fixture. Consequently, these files support a fixed teaching
demonstration, but they are not a byte-for-byte regenerable research result and
must not be represented as one. Their SHA-256 checksums at public release are:

```text
simcausal_train.csv    0EA465E6596397A92DDDFF0615807B8C093FFA97666004107887DD96B381752E
ground_truth_edges.csv EF6265DF9E4DEC5F119810669742A175596BFE8F0A081D9AFDB53275F94C24B3
true_total_effects.json 24FE4BAA5C386D290651C868204D82F15533B3223C821035A68F19649CC1097F
```

## Public-data boundary

Do not add clinical row-level extracts, samples, train/test splits, or other
records governed by a data-use agreement to this directory or elsewhere in the
public repository. In particular, `sample_train.csv`, `train.impute.csv`, and
`test.impute.csv` are local-only inputs and are excluded by `.gitignore`.

A clinical edge list contains no individual rows, but it still requires an
explicit provenance statement and permission for public distribution before it
can be committed.
