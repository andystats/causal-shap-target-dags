# Video Script and Storyboard

Working title: **When Feature Importance Points to the Wrong Lever**

Target length: approximately 8 minutes. The narrator may be on camera for the
opening and closing; the analytic middle should be screen-led.

## Scene 1 — Cold open: the wrong lever (0:00–0:35)

**Picture:** Full-screen ordinary-SHAP ranking from the dramatic ACIC stress
test. Highlight a downstream proxy or late mediator. Cut to a simple DAG with an
upstream intervention node and the highlighted downstream node.

**Narration:**

> The model is doing exactly what we asked. It found the features that best
> predict the outcome. But what if the most predictive feature is not a useful
> place to intervene? Feature importance can identify the smoke alarm when the
> decision-maker needs the source of the fire.

**On-screen text:** `Prediction ≠ intervention`

**Props:** P01, P02, P12.

## Scene 2 — Give the user a mission (0:35–1:10)

**Picture:** Open the app in guided mode. Dataset card says "NASA renal-stone
risk"; target says `Nephrolithiasis`. The user presses **Start with prediction**.

**Narration:**

> Imagine you are supporting a human-spaceflight risk team. Your target is
> nephrolithiasis. You have a predictive model and a ranked list of features.
> The practical question is not only what predicts risk. It is where the system
> offers a plausible lever.

**On-screen action:** Show target, dataset provenance, prevalence, and the
eligible pre-outcome feature count. Do not show causal results yet.

**Props:** P03, P04, P13.

## Scene 3 — Reveal the Living DAG (1:10–1:55)

**Picture:** Animate or dissolve from the plain feature list to the NASA
SA-07566 DAG, then isolate the 28 ancestors of `Nephrolithiasis`. Color nodes by
directed hop distance.

**Narration:**

> This is not a correlation diagram inferred after the fact. The simulation is
> built from NASA's published renal-stone DAG code. The graph tells us which
> variables are upstream, which are mediators, and which lie immediately beside
> the outcome. For this target, 28 ancestors span one to six directed hops.

**On-screen text:** `51 nodes · 75 edges · structural kappa 1.000`

**Props:** P05, P06.

## Scene 4 — Hide the answer inside a simulated world (1:55–2:35)

**Picture:** Show the structural-simulation pipeline: DAG → equations → 10,000
records → intervention truth. Briefly display the frozen intervention contrasts.

**Narration:**

> Because the data come from a known structural model, we can calculate an
> answer ordinary observational data rarely give us: the standardized total
> effect of intervening on every ancestor. Binary variables move from zero to
> one; continuous variables move from the observed first to third quartile. We
> freeze that truth before looking at SHAP.

**On-screen text:** `Truth first. Attribution second.`

**Props:** P07, P08, P14.

## Scene 5 — The AUC surprise (2:35–3:20)

**Picture:** Show the predictive-signal diagnostic bars. Reveal the oracle first,
then the fitted learners.

**Narration:**

> Our first learner was XGBoost, using all 28 eligible ancestors. Its held-out
> AUC was only 0.684. That initially looked suspicious for a parametric
> DAG-generated dataset. But the true structural risk score itself reaches only
> about 0.701. A correctly specified logistic regression on the two direct
> parents also reaches 0.701. The low ceiling is in the data-generating process:
> modest coefficients, noisy intermediate nodes, a Bernoulli outcome, and an
> event rate under nine percent. A more flexible learner cannot recover signal
> the simulation never created.

**On-screen text:** `Oracle AUC 0.701 · XGBoost AUC 0.684`

**Props:** P09, P15.

## Scene 6 — The dramatic stress test (3:20–4:15)

**Picture:** Switch the app to **Pedagogic mediator/proxy stress test**. Run
ordinary SHAP, then press **Use known DAG**. Animate rank movement and proxy mass.

**Narration:**

> Before the NASA case, here is the deliberately dramatic version. Downstream
> proxies are highly predictive but have no intervention effect by construction.
> Ordinary SHAP rewards them. When the attribution procedure uses the known
> causal structure, importance moves toward upstream causes. This is the
> intuition the app should make immediately visible.

**On-screen label throughout:** `Designed stress test — not the primary NASA result`

**Props:** P01, P10, P16.

## Scene 7 — Return to the honest NASA result (4:15–5:25)

**Picture:** Switch back to clean v3. Show the distance-concentration curves and
the three/four-panel ancestor importance map.

**Narration:**

> The source-aligned NASA result is subtler. Interventional truth places only
> 42 percent of its importance within two hops. Every predictive attribution
> method places more than 81 percent there. Exact TreeSHAP looks slightly worse;
> DAG-asymmetric SHAP looks slightly better. But that comparison changes both
> ordering and the missing-feature distribution.

**On-screen action:** Pause with TreeSHAP, matched ordinary SHAP, and
DAG-asymmetric SHAP all visible.

**Props:** P11, P17, P18.

## Scene 8 — The matched-background reveal (5:25–6:30)

**Picture:** Dim TreeSHAP. Highlight matched unrestricted ordinary SHAP and
DAG-asymmetric SHAP. Display their identical background and permutation counts,
then show paired-bootstrap intervals.

**Narration:**

> So we ran the fair control: the same model, the same 64 people, the same 128
> background records, the same output scale, and the same number of
> permutations. The only difference is whether permutations must respect the
> DAG. The apparent advantage disappears. PBI is 1.051 for both, POA is 0.210
> for both, and every paired interval includes zero. Causal ordering alone is not
> enough.

**On-screen text:** `A falsification, not a failure`

**Props:** P11, P19.

## Scene 9 — What causal SHAP must do next (6:30–7:25)

**Picture:** Show a coalition intervention at an upstream node. Animate the
change propagating through descendants before the prediction is evaluated.
Contrast this with merely changing the order in which features enter.

**Narration:**

> The next method has to do more than respect order. It must propagate an
> intervention through the structural system. If hydration changes, urine
> concentration and chemistry should change with it before the model is queried.
> That is the intervention-propagating value function we will implement next,
> first for nephrolithiasis and then for loss of mission objectives.

**On-screen text:** `Order features` → `Propagate interventions`

**Props:** P20, P21.

## Scene 10 — Close: the promise of Living DAGs (7:25–8:00)

**Picture:** Return to the full DAG with candidate intervention nodes emphasized.
End on the app home screen with **Guided story** and **Explore the lab** buttons.

**Narration:**

> The lesson is not that feature importance is useless. It answers a predictive
> question. Living DAGs let us ask a consequential one: where in this system
> might action change what happens next? The value of the demo is that users can
> watch those questions separate—and see when a proposed causal method genuinely
> earns the difference.

**End card:** paper title, authors, repository/DOI placeholder, QR code.

**Props:** P05, P22, P23.

## Short-cut versions

### Three-minute conference cut

Use Scenes 1, 3, 5, 7, 8, and 10. Keep the oracle-AUC comparison and the
matched-background result; these are the credibility anchors.

### Ninety-second teaser

Use Scene 1, a five-second DAG reveal, the dramatic stress-test rank movement,
then the line: "In the source-aligned case, the fair control erased the win."
End with `Prediction ≠ intervention` and the paper title.
