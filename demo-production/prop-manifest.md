# Demo and Video Prop Manifest

Status values: **ready**, **adapt**, **create**, or **blocked**.

| ID | Prop | Purpose | Status | Canonical source or target |
| --- | --- | --- | --- | --- |
| P01 | ACIC many-mediator dataset | Dramatic pedagogic stress test | ready | `app/bundles/acic_proxy_stress_test/data.csv` |
| P02 | Wrong-lever opening graphic | Cold-open proxy-versus-intervention contrast | create | Derive from ACIC standard-SHAP ranking and DAG |
| P03 | Guided app home screen | Establish mission, dataset, and target | ready | `app/app.py` |
| P04 | Clean v3 dataset card | Provenance, rows, outcome rate, target | ready | `analysis/output/source_aligned_clean/renal_stone_source_aligned_clean_v3.csv` |
| P05 | Full NASA renal DAG | Living-DAG reveal and closing | ready | `analysis/output/dag_validation/nasa_source_dag.png` |
| P06 | Nephrolithiasis ancestor DAG | Show 28 ancestors and hop distance | ready | `analysis/output/shap_nephrolithiasis_clean_v3/ancestor_importance_maps.png` plus target table |
| P07 | Structural-simulation pipeline | DAG → equations → data → truth | create | Use canonical R scripts and a simple vector graphic |
| P08 | Intervention contrast table | Show truth was frozen before SHAP | ready | `analysis/output/shap_nephrolithiasis_clean_v3/intervention_contrasts.csv` |
| P09 | Predictive-signal diagnostic | Explain low AUC and learner ceiling | ready | `analysis/output/shap_nephrolithiasis_clean_v3/predictive_signal_diagnostics.png` |
| P10 | ACIC rank-movement view | Dramatic ordinary-versus-causal reveal | adapt | Existing Python app rank table and comparison plot |
| P11 | Distance-concentration curves | Primary proximity-bias result | ready | `analysis/output/shap_nephrolithiasis_clean_v3/distance_concentration_curves.png` |
| P12 | `Prediction ≠ intervention` title card | Recurrent visual motif | create | 16:9 and square variants |
| P13 | Astronaut/risk-team framing image | Humanize the mission | create/license | Prefer NASA public-domain image with attribution log |
| P14 | Frozen-truth stamp/overlay | Make prespecification visible | create | Small transparent SVG/PNG |
| P15 | Oracle-versus-learner reveal animation | AUC scene | create | Animate P09 bars in sequence |
| P16 | Stress-test label | Prevent pedagogic data being mistaken for NASA result | create | Persistent lower-third |
| P17 | Feature-importance comparison | Show proximal over-credit at node level | ready | `analysis/output/shap_nephrolithiasis_clean_v3/feature_importance_comparison.png` |
| P18 | Four-panel ancestor importance map | Compare truth, TreeSHAP, matched ordinary, DAG-asymmetric | ready | `analysis/output/shap_nephrolithiasis_clean_v3/ancestor_importance_maps.png` |
| P19 | Paired-bootstrap interval overlay | Matched-control reveal | ready/adapt | `paired_bootstrap_summary.csv`; create compact forest plot |
| P20 | Ordering-only animation | Explain current asymmetric method | create | Topological permutation entering left-to-right |
| P21 | Intervention-propagation animation | Explain structural method | create | Tested engine in `app/causal_shap/structural_value.py`; animation still needed |
| P22 | Guided/lab mode end screen | Invite reproducible exploration | ready | `app/app.py` |
| P23 | Paper end card and QR code | Citation and access | blocked | Final title, DOI, deployment URL |

## Dataset bundle for the app

The app should ship with a versioned local manifest and no hidden downloads:

- `acic_proxy_stress_test` — current ACIC dataset, DAG, and true total effects.
- `nasa_renal_clean_v3` — source-aligned clean data and all fixed analysis
  manifests.
- `nasa_renal_nasa_like_v4` — sparse/selected data, held until the clean method
  is locked.
- `nasa_loss_mission_objectives` — create after the structural method is ready.
- Optional `pedagogic_amplified_signal` — only if explicitly labeled and kept
  out of primary scientific claims.

Each bundle needs: data CSV or Parquet; DAG edges and nodes; variable map; target
metadata; intervention contrasts; truth; train/test/evaluation/background row
manifests; model file; attribution outputs; metric outputs; and a SHA-256
manifest.

## Recording package

- 1920×1080 app layout with browser zoom locked.
- Cursor-highlight preset and keyboard shortcuts documented.
- Clean browser profile with notifications disabled.
- Voiceover WAV, screen capture, and face-camera tracks recorded separately.
- Static fallback images for every live computation.
- Captions/SRT, transcript, figure alt text, and color-blind-safe palettes.
- Attribution ledger for NASA images, papers, icons, fonts, and music.
