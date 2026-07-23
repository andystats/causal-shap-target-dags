# Site and illustration integration guide

## Canonical source of truth

Use this order when changing the public story:

1. settle the claim and wording in
   [`PROJECT_NARRATIVE.md`](PROJECT_NARRATIVE.md);
2. add or verify supporting sources in
   [`RESEARCH_REFERENCES.md`](RESEARCH_REFERENCES.md);
3. update the canonical Quarto Pages source under `site/`;
4. render and visually inspect the site;
5. only then port the change to the root static companion if that prototype is
   being actively maintained.

The current GitHub Pages workflow deploys `site/_site`, not the root
`index.html`.

## Narrative-to-file map

| Narrative element | Canonical web file |
|---|---|
| Elevator pitch and five-node demonstration | `site/index.qmd` |
| Markov proof and SHAP-variant distinctions | `site/why-it-happens.qmd` |
| Teaching result, NASA ordering null, structural prototype | `site/evidence.qmd` |
| ACIC continuity and annotated published foundation | `site/research-lineage.qmd` |
| Methods and provenance | `site/about.qmd` plus `docs/` |
| Rebuild commands | `site/reproducibility.qmd` |
| Navigation and page order | `site/_quarto.yml` |
| Editorial design and placeholder styles | `site/theme.scss` |

The six-rung workflow, app, cheatsheets, and glossary are supporting material.
Keep them under the **Lab notebook** menu so they do not compete with the single
demonstration.

## Hand-drawn illustration plan

Store final artwork in:

```text
site/assets/illustrations/
```

Recommended files:

| Slot | Filename | Suggested canvas | Purpose |
|---|---|---:|---|
| 01 | `splat-and-lever.webp` | 1600 × 1200 | A paint splat sticks to the last node while a hand reaches upstream for the lever. |
| 02 | `screening-curtain.webp` | 1200 × 1600 | The mediator closes a curtain for prediction while an intervention arrow still travels through it. |
| 03 | `reachable-lever.webp` | 1800 × 600 | A hand selects the effective, feasible, robust lever from several upstream candidates. |

Art direction:

- loose black or deep-green ink;
- one coral wash, with optional muted yellow for mediators;
- no labels baked into the art—the surrounding HTML supplies accessible text;
- transparent or warm-paper background;
- WebP preferred, PNG acceptable; target less than 350 KB per image;
- preserve the slight imperfection and margin-note character of a real sketch.

## Replacing a placeholder in Quarto

Search `site/index.qmd` for `HAND-DRAWN SLOT 01`, `02`, or `03`. Replace the
corresponding placeholder `<figure>` with:

```html
<figure class="illustration-final hero-illustration">
  <img
    src="assets/illustrations/splat-and-lever.webp"
    alt="A paint splat clings to the mediator nearest the outcome while a hand reaches back to an upstream intervention lever.">
  <figcaption>The splat and the lever.</figcaption>
</figure>
```

Use these alt texts as starting points:

- **Slot 01:** “A paint splat clings to the node nearest the outcome while a
  hand reaches back to an upstream intervention lever.”
- **Slot 02:** “A mediator screens the upstream cause from a prediction lens,
  while a causal arrow continues from the cause through the mediator to the
  outcome.”
- **Slot 03:** “A hand chooses one reachable upstream lever after filtering
  candidates for effect, feasibility, cost, and uncertainty.”

Add the following beside the existing placeholder rules in `site/theme.scss`
when the first final illustration arrives:

```scss
.illustration-final {
  margin: 0;
  align-self: center;
}
.illustration-final img {
  display: block;
  width: 100%;
  height: auto;
  max-height: 620px;
  object-fit: contain;
}
.illustration-final figcaption {
  margin-top: 0.55rem;
  color: var(--muted);
  font-size: 0.75rem;
}
```

Do not delete the placeholder copy until the final image has been viewed at
desktop and mobile widths. It is the art brief and should remain recoverable in
Git history.

## Reusing the artwork in the root static companion

The same physical files can serve the no-build root page. Paths differ because
the HTML lives one directory higher:

```html
<img
  src="site/assets/illustrations/splat-and-lever.webp"
  alt="A paint splat clings to the mediator while a hand reaches upstream for the intervention lever.">
```

Update `site/styles.css` for presentation and leave `site/app.js` responsible
only for interaction. Do not duplicate scientific copy inside JavaScript unless
the interaction truly requires it; when it does, keep `valorCopy` and
`methodStages` synchronized with the canonical narrative.

## Reference placement

- Put a source link at the first substantive claim it supports.
- Prefer primary proceedings or publisher pages over secondary summaries.
- Use links, not long quotations.
- Keep the detailed annotation in `docs/RESEARCH_REFERENCES.md`; website copy
  should state only what the reader needs for the argument.
- When describing this project's contribution, cite Heskes for the do-based
  value function and describe our contribution as **target-recovery validation**.

## Render and review

From the repository root:

```bash
quarto render site
python3 -m http.server 8765 --directory site/_site
```

Review at minimum:

1. the first viewport at desktop and phone widths;
2. all three illustration crops and alt text;
3. the ordinary-SHAP versus intervention-truth bars;
4. equation overflow on a narrow screen;
5. navigation and every new internal link;
6. the distinction between established and provisional evidence.

Then run:

```bash
.venv/bin/python -m unittest discover -s app/tests -v
.venv/bin/python -m causal_shap.build validate
git diff --check
```

Commit and push only after the site renders without warnings and these checks
pass.
