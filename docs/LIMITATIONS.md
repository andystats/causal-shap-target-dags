# Limitations and Guardrails

1. The data are synthetic and the structural coefficients are not NASA effect
   estimates.
2. The primary source graph still requires domain review for version choice,
   actionability labels, and intervention cost/difficulty.
3. A held-out AUC near 0.68 limits prediction-level separation, although the true
   structural score shows that this is mostly a simulation ceiling.
4. DAG-constrained ordering and structural intervention propagation answer
   different questions; they should not share the label “Causal SHAP” without a
   precise value-function definition.
5. The encouraging structural result is currently a 32×32×32 prototype without
   paired bootstrap uncertainty or a simulation-seed grid.
6. The pedagogic mediator/proxy example is intentionally dramatic and is not
   primary scientific evidence.
7. Intervention-target ranking is not a treatment recommendation. Feasibility,
   safety, timing, cost, and domain constraints are outside the current engine.
