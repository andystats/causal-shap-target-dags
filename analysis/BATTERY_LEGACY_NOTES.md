# Legacy M1–M5 battery notes

## The August 13 pilot is unverified

The local `analysis/output/battery_v1/stage_results.json` is a single-seed
pilot generated on 2026-08-13. It is a useful historical record, but it is
**not a stage-worthy result artifact** and must not be presented as a
clean-clone reproduction. The legacy run did not record its Python or package
versions, Git revision, or input hashes; it used a machine-specific import
path, current-directory input aliases, Unix-only `SIGALRM` timeouts, and an
arbitrary cycle-breaking conversion from CPDAGs to DAGs. Its elapsed times are
also machine-specific.

The local `run_stages.py` is retained only as provenance for that JSON. Do not
use or publish it as the supported runner. The local
`renal_edges_mapped.csv` is a derived copy of the 75 committed source-aligned
edges; the supported runner reconstructs it from the source graph and
crosswalk instead of treating it as an input.

The legacy pilot's observations remain hypotheses to retest: toy LiNGAM placed
the `ClinicVisit` collider in the learned adjustment set, and renal PC recovered
the skeleton much better than edge directions while placing two mediators in
the learned adjustment set. “GES clean on both” referred only to the learned
adjustment set; renal GES did not recover the full directed graph or the frozen
effect exactly.

## Supported portable runner

Run from any working directory after installing the repository's discovery
dependencies:

```powershell
py -3.13 analysis/run_m1_m5_battery.py --example all
```

For a CI or local smoke run, keep generated files outside the checked output:

```powershell
py -3.13 analysis/run_m1_m5_battery.py --example toy --output-dir $env:TEMP\causal-shap-battery
```

The runner reads only committed synthetic inputs:

- toy data, edges, and total-effect truth from
  `app/bundles/toy_chain_fork_collider/`;
- renal clean-v3 data from `analysis/output/source_aligned_clean/`;
- renal source edges and the source-to-variable crosswalk from the committed
  DAG-validation outputs; and
- the 50,000-draw renal intervention truth from
  `analysis/output/shap_nephrolithiasis_clean_v3/`.

It writes `battery_results.json`, a deterministic scientific payload, and
`battery_run_metadata.json`, which separately records input hashes,
environment, Git state, wall-clock timings, and the scientific-payload hash.
With a renal run it also writes the reconstructed `renal_edges_mapped.csv`.
Canonical output is not replaced after an error or timeout; `--allow-partial`
is for explicitly diagnostic output only.

Version 2 uses a lexicographically deterministic Dor–Tarsi consistent
extension of each discovered CPDAG. It preserves compelled directions and does
not add new unshielded colliders. Because this corrects the legacy arbitrary
cycle-breaking semantics, version-2 results must be regenerated and reviewed
rather than assumed to equal the legacy JSON.

The renal exposure is currently `altered_gravity`, for which frozen
intervention truth exists. Clean-v3 has no mission-duration variable, so it
cannot answer a mission-duration exposure question without a separate,
prospectively specified simulation.
