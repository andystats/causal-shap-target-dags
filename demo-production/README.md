# Causal SHAP Paper Demo and Video Kit

This folder is the production source of truth for the interactive demo and its
companion video. The intended experience is not a glossy claim that causal SHAP
always wins. It is a guided scientific reveal:

1. Predictive importance points toward variables near the outcome.
2. The DAG changes the question from "what predicts?" to "where could we act?"
3. A dramatic mediator/proxy stress test shows why causal structure can matter.
4. The source-aligned NASA case is subtler and exposes an important negative
   result: causal ordering alone is not enough.
5. A small structural intervention-propagating prototype shows the intended next
   method and stays labeled exploratory until scaled and bootstrapped.

## Recommended format

- Main paper companion: 7–9 minutes, screen-led with short voiceover.
- Conference cut: 3 minutes, retaining the matched-background falsification.
- Social teaser: 60–90 seconds, ending at "prediction is not intervention."
- App: one guided path for the recorded story plus a separate lab mode for
  exploration.

## Two-dataset dramatic structure

- **Act I — pedagogic stress test.** Use the existing ACIC many-mediator/proxy
  dataset to make the attribution failure visually obvious. Clearly label it as
  a designed stress test.
- **Act II — source-aligned NASA analysis.** Use clean v3 to show the exact NASA
  topology, standardized intervention truth, modest discrimination, and the
  matched-background null result. This is the credible scientific center.

Do not strengthen the source-aligned coefficients merely to create a larger
visual win. If a stronger demonstration regime is added, name it
`pedagogic_amplified_signal` and keep it separate from the primary analysis.

## Files

- `video-script.md` — timed narration, screen actions, and edit notes.
- `prop-manifest.md` — every app, dataset, figure, overlay, and recording asset.
- `python-app-plan.md` — audit of the existing Python Shiny app and the target
  publication architecture.
- `../app/` — implemented deterministic Python Shiny app, local result bundles,
  and tested structural attribution engine.

## Production gate

Record only after the app can reproduce checked-in precomputed results without
network access and after every number shown on screen traces to a CSV or model
artifact. Live recomputation is optional; reproducible playback is mandatory.
