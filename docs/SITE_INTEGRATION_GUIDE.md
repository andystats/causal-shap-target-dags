# Site and illustration integration guide

## One public argument

The canonical website is `site/index.qmd`, rendered by Quarto and published on
GitHub Pages. The root `index.html` is only a redirect. Do not create a second
site or duplicate the narrative in JavaScript.

Use this order when changing the public story:

1. settle the claim in [`PROJECT_NARRATIVE.md`](PROJECT_NARRATIVE.md);
2. verify its support in [`RESEARCH_REFERENCES.md`](RESEARCH_REFERENCES.md);
3. update `site/index.qmd`;
4. update `site/theme.scss` only when the visual system needs it;
5. render, inspect the output, and run the repository checks.

## Narrative hierarchy

The public page should always read as an expansion of the archived [Tao RWD
ACIC 2026 Causal SHAP project](https://www.tao-rwd.com/acic-2026/causal-shap):

1. **Prediction** — what is likely to happen?
2. **Predictive attribution** — what did the fitted model use?
3. **Structural attribution** — what propagates after `do(a)`?
4. **Intervention target** — what is mutable and feasible?
5. **Recommendation candidate** — which feasible action is worth testing under
   cost and uncertainty?

The repository currently supplies quantitative evidence through rung 3. Rungs 4
and 5 are the decision target, not an implemented or validated result. Preserve
that boundary in headings, captions, and calls to action.

## ACIC visual inheritance

The live Tao archive—not the local conference pointer—is the design source of
truth. Preserve its quiet editorial grammar:

- warm bone page field and paper panels;
- brown-black ink with muted warm-gray secondary copy;
- cyan for the primary causal argument, teal for causes, and orange for
  warnings or leaked downstream credit;
- Fraunces display headings with Inter body copy;
- yellow highlighter used sparingly for one key phrase;
- centered editorial widths, rounded evidence panels, fine rules, and soft
  shadows;
- hand-drawn black causal arrows with the same cyan, teal, and orange accents.

The site variables in `site/theme.scss` are the canonical palette:

| Role | Value |
|---|---|
| Bone | `#f4ede0` |
| Paper | `#fbf7ef` |
| Ink | `#1a1814` |
| Muted ink | `#6b6258` |
| Cyan | `#0077a8` |
| Teal | `#00897b` |
| Orange | `#e07020` |
| Highlighter | `#ffef45` |

## Hand-drawn illustration plan

The archive already contains the original explanatory drawings. Link to those
canonical files rather than redrawing or duplicating them:

- `https://www.tao-rwd.com/img/acic-2026/WizardOfShap.png`
- `https://www.tao-rwd.com/img/acic-2026/SimpleTruth.jpg`
- `https://www.tao-rwd.com/img/acic-2026/SimpleWhatSHapSees.jpg`

Only one new illustration is needed. Store it in
`site/assets/illustrations/` when it is ready.

| Slot | Filename | Suggested canvas | Purpose |
|---|---|---:|---|
| 01 | `intervention-hierarchy.webp` | 1800 × 700 | A hand-drawn staircase from prediction to recommendation, with the current evidence boundary at structural attribution. |

Art direction:

- loose black ink with the archive's cyan, teal, and orange accents;
- imperfect circles, arrows, short margin notes, and plenty of white space;
- no paragraph text baked into the image;
- transparent or white background;
- WebP preferred, PNG acceptable, ideally below 350 KB;
- no logos or additional visual system.

## Replacing a placeholder

Search `site/index.qmd` for `HAND-DRAWN INSERT 01`. Replace the whole
placeholder `<figure>` with:

```html
<figure class="final-illustration hierarchy-sketch">
  <img
    src="assets/illustrations/intervention-hierarchy.webp"
    alt="Five ascending steps move from prediction through attribution and structural propagation to feasible intervention and recommendation; a marker at the third step shows where current evidence stops.">
  <figcaption>The hierarchy from model explanation to a recommendation candidate.</figcaption>
</figure>
```

Then add only the small image rules needed for `.final-illustration` to
`site/theme.scss`. Keep the placeholder recoverable in Git history.

## Content discipline

- Keep only the three public numbers needed for the argument: 46%/0% in the
  teaching trap, ordering-only τ 0.528, and structural τ 0.794 with top-five
  recovery 1.00.
- Say “NASA-topology simulation,” never “NASA effect.”
- Call the structural result promising and provisional.
- Use “recommendation candidate,” not “recommendation” or “causal promise.”
- Link outward to `docs/` for methods, provenance, detailed tables, references,
  and reproducibility.
- Do not add named future tools to the public hierarchy unless they become part
  of a validated rung.

## Render and verify

From the repository root:

```bash
quarto render site
.venv/bin/python -m unittest discover -s app/tests -v
.venv/bin/python -m causal_shap.build validate
Rscript analysis/validate_outputs.R
git diff --check
```

The rendered artifact should contain one substantive HTML page. Check the first
viewport, the three archived illustrations, the five hierarchy rungs, the one
new-art placeholder, the narrow-screen equations, all anchors, and the
established/provisional distinction before publishing.
