"""Named seed constants continuing the repo's dated-integer register.

Every stochastic build step draws its seed from this module so the register
stays collision-free. Historical seeds already burned into committed artifacts:
R pipeline 20260710-20260720, Python structural builds 20260723/20260724,
ACIC pedagogic bundle 20260725/20260726. New entries continue from 20260727.
"""

from __future__ import annotations

SEED_TEACHING_TOY = 20260727
SEED_TEACHING_LADDER = 20260728
SEED_DISCOVERY = 20260729
SEED_VALIDATION_TRUTH = 20260730
SEED_CVAE_TRAINING = 20260731
SEED_VALIDATION_SCENARIOS = 20260801
SEED_FIGURES = 20260802
