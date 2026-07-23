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

The public page should always read as an expansion of the ACIC 2026 poster:

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

Borrow the poster's visual grammar rather than treating it as a generic brand:

- white working surface with pale blue-gray framing;
- strong cobalt/blue for causes and the core argument;
- orange-red for leaked downstream credit and warnings;
- black causal arrows and loose handwritten annotations;
- large sans-serif statements, thin rules, and almost no decorative chrome;
- flat diagrams and open space instead of cards, gradients, shadows, or
  dashboard controls.

The site variables in `site/theme.scss` are the canonical palette:

| Role | Value |
|---|---|
| Cobalt | `#0647a6` |
| Poster blue | `#2477b8` |
| Orange | `#cf5a00` |
| Red | `#c9362b` |
| Blue-gray field | `#dfe5ec` |
| Ink | `#131820` |

## Hand-drawn illustration plan

Store final art in `site/assets/illustrations/`.

| Slot | Filename | Suggested canvas | Purpose |
|---|---|---:|---|
| 01 | `truth-vs-shap.webp` | 1600 × 1200 | Extend the poster's “Truth vs. what SHAP thinks” panel: an upstream cause fades while a mediator swells with credit. |
| 02 | `intervention-hierarchy.webp` | 1800 × 700 | A hand-drawn staircase from prediction to recommendation, with the current evidence boundary at structural attribution. |

Art direction:

- loose black ink with the poster's blue and orange-red accents;
- imperfect circles, arrows, short margin notes, and plenty of white space;
- no paragraph text baked into the image;
- transparent or white background;
- WebP preferred, PNG acceptable, ideally below 350 KB;
- no logos or additional visual system.

## Replacing a placeholder

Search `site/index.qmd` for `HAND-DRAWN INSERT 01` or `HAND-DRAWN INSERT 02`.
Replace the whole placeholder `<figure>` with:

```html
<figure class="final-illustration hero-sketch">
  <img
    src="assets/illustrations/truth-vs-shap.webp"
    alt="An upstream cause fades while the downstream mediator expands with predictive credit, although the causal path continues through both nodes to the outcome.">
  <figcaption>Truth versus what SHAP thinks.</figcaption>
</figure>
```

For slot 02, use this alt-text starting point:

> Five ascending steps move from prediction through attribution and structural
> propagation to feasible intervention and recommendation; a marker at the
> third step shows where current evidence stops.

Then add only the small image rules needed for `.final-illustration` to
`site/theme.scss`. Keep the placeholder brief recoverable in Git history.

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
viewport, the five hierarchy rungs, both placeholders, the narrow-screen
equations, all anchors, and the established/provisional distinction before
publishing.
