const header = document.querySelector('[data-header]');
const navToggle = document.querySelector('.nav-toggle');
const nav = document.querySelector('.site-nav');

const setHeaderState = () => header?.classList.toggle('is-scrolled', window.scrollY > 12);
setHeaderState();
window.addEventListener('scroll', setHeaderState, { passive: true });

navToggle?.addEventListener('click', () => {
  const open = nav?.classList.toggle('is-open');
  navToggle.setAttribute('aria-expanded', String(Boolean(open)));
});

nav?.querySelectorAll('a').forEach((link) => link.addEventListener('click', () => {
  nav.classList.remove('is-open');
  navToggle?.setAttribute('aria-expanded', 'false');
}));

const revealObserver = 'IntersectionObserver' in window
  ? new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      });
    }, { rootMargin: '0px 0px -7% 0px', threshold: 0.08 })
  : null;

document.querySelectorAll('.reveal').forEach((element) => {
  if (revealObserver) revealObserver.observe(element);
  else element.classList.add('is-visible');
});

const valorVisual = document.querySelector('[data-valor-view]');
const valorCaption = document.querySelector('[data-valor-caption]');
const valorCopy = {
  shap: '<span>Traditional SHAP</span> The mediators nearest the outcome can absorb credit that belongs to the upstream intervention.',
  causal: '<span>Causal view</span> Set the upstream node, propagate its consequences forward, and credit the full response carried through the mediator chain.'
};

document.querySelectorAll('.valor-button').forEach((button) => {
  button.addEventListener('click', () => {
    const view = button.dataset.valor;
    if (!valorVisual || !valorCopy[view]) return;
    valorVisual.dataset.valorView = view;
    valorCaption.innerHTML = valorCopy[view];
    document.querySelectorAll('.valor-button').forEach((candidate) => {
      const active = candidate === button;
      candidate.classList.toggle('is-active', active);
      candidate.setAttribute('aria-pressed', String(active));
    });
  });
});

const methodStages = {
  shap: {
    label: 'Traditional SHAP',
    copy: 'A faithful explanation of this fitted model on this case. It can identify predictive mediators or treatment markers without establishing where intervention should begin.',
    href: '#blindspot',
    link: 'See the valor-stealing example'
  },
  causal: {
    label: 'Structural Causal SHAP',
    copy: 'Intervene on a DAG node, propagate the change through its descendants, and measure the resulting model response. This is the layer that returns indirect effect upstream.',
    href: '#evidence',
    link: 'Inspect the renal benchmark'
  },
  lumawarp: {
    label: 'LumaWarp',
    copy: 'Use nonlinear interrelation as a searchlight for deeper nodes that ordinary attribution may miss when credit pools on mediators. Candidates still require DAG and domain review.',
    href: '#stack',
    link: 'Read the depth-signal guardrail'
  },
  dice: {
    label: 'DiCE',
    copy: 'Generate diverse counterfactual changes on the causally screened, mutable part of the graph. The output is a set of possible moves rather than another importance ranking.',
    href: '#stack',
    link: 'See the recourse layer'
  },
  cost: {
    label: 'Cost-sensitive DiCE',
    copy: 'Rank feasible counterfactuals using decision-specific burden, time, reversibility, resource use, and implementation difficulty to form an actionable intervention portfolio.',
    href: '#program',
    link: 'See the claim under test'
  }
};

const orientationCard = document.querySelector('[data-method-stage]');
const explainerLabel = document.querySelector('.map-explainer-label');
const explainerCopy = document.querySelector('.map-explainer-copy');
const explainerLink = document.querySelector('.map-explainer .text-link');

document.querySelectorAll('.stage-button').forEach((button) => {
  button.addEventListener('click', () => {
    const stage = button.dataset.stage;
    const content = methodStages[stage];
    if (!content || !orientationCard) return;
    orientationCard.dataset.methodStage = stage;
    document.querySelectorAll('.stage-button').forEach((candidate) => {
      const active = candidate === button;
      candidate.classList.toggle('is-active', active);
      candidate.setAttribute('aria-pressed', String(active));
    });
    explainerLabel.textContent = content.label;
    explainerCopy.textContent = content.copy;
    explainerLink.href = content.href;
    explainerLink.innerHTML = `${content.link} <span aria-hidden="true">→</span>`;
  });
});

const dialog = document.querySelector('.figure-dialog');
const dialogImage = dialog?.querySelector('.dialog-media img');
const dialogTitle = dialog?.querySelector('#dialog-title');
const dialogCaption = dialog?.querySelector('.dialog-copy p:last-child');
let dialogTrigger = null;

const closeDialog = () => {
  if (dialog?.open) dialog.close();
};

document.querySelectorAll('.figure-button').forEach((button) => {
  button.addEventListener('click', () => {
    if (!dialog || !dialogImage || !dialogCaption || !dialogTitle) return;
    dialogTrigger = button;
    dialogImage.src = button.dataset.image;
    dialogImage.alt = button.dataset.alt || '';
    dialogTitle.textContent = button.dataset.alt || 'Evidence figure';
    dialogCaption.textContent = button.dataset.caption || '';
    dialog.showModal();
    document.body.classList.add('dialog-open');
  });
});

dialog?.querySelector('.dialog-close')?.addEventListener('click', closeDialog);
dialog?.addEventListener('click', (event) => {
  if (event.target === dialog) closeDialog();
});
dialog?.addEventListener('close', () => {
  document.body.classList.remove('dialog-open');
  dialogImage?.removeAttribute('src');
  dialogTrigger?.focus();
});
