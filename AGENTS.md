# Repository working agreement

## Branch and publishing convention

- Work directly on `main` unless Andy explicitly requests a branch or pull
  request.
- At the start of a task, switch to `main`, fetch `origin`, and reconcile remote
  changes without overwriting them. Never force-push `main`.
- After each coherent repository change, run the relevant checks, create a terse
  commit, and push `main` to `origin` unless Andy explicitly asks to leave the
  work uncommitted.
- Do not open an `agent/*` branch or draft pull request by default in this
  repository.
- Attribute commits only to the humans responsible. Never add automated-assistant
  authors, `Co-authored-by` trailers, or tool branding to commit metadata.

## Public-site convention

- The only canonical public site is the single-page Quarto project under
  `site/`; the deployment workflow publishes `site/_site`.
- Keep the public page as a minimalist hierarchy from prediction through
  recommendation. Put methodological depth, references, provenance, and
  limitations in `docs/`, not in additional public microsites.
- The root `index.html` is only a lightweight redirect to GitHub Pages. Do not
  grow it into a second web surface.
- Settle narrative and references in `docs/PROJECT_NARRATIVE.md` and
  `docs/RESEARCH_REFERENCES.md` before changing the public page.

## Scientific guardrails

- Keep teaching stress tests separate from source-aligned simulations.
- Preserve null and diagnostic results; do not tune the data-generating process
  to manufacture a stronger story.
- Say “NASA-topology simulation,” not “NASA effect,” when coefficients and
  outcomes are synthetic.
- Treat structural attribution as a prototype input to action selection, not as
  a guaranteed recommendation.
