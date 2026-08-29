"""What an intervention is allowed to be, and what it costs to do it.

Attribution ranks nodes by influence. Acting on them is a different question:
a node may be immovable, unethical, or simply too expensive to be worth its
effect. This module holds the feasibility and price half of that decision, kept
deliberately separate from the benefit half in ``policy`` so that a cost sheet
can be reviewed by a domain expert without reading any causal code.

Costs are charged on the nodes an actor *directly* assigns, never on the
downstream changes the SCM propagates: moving one upstream lever should not be
billed again for every descendant it happens to shift.

An action is a mapping from node name to a **shift** applied to that node's
factual value, so a cost of ``fixed + unit * |shift|`` prices "how far you moved
it" rather than "where it ended up". Bounds on absolute attainable values are a
later extension; ``min_shift``/``max_shift`` bound the movement itself.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

ILLUSTRATIVE = "ILLUSTRATIVE - not domain-reviewed"


@dataclass(frozen=True)
class ActionSpec:
    """Whether a node can be moved, how far, and what moving it costs."""

    node: str
    manipulable: bool = False
    min_shift: float = 0.0
    max_shift: float = 0.0
    fixed_cost: float = 0.0
    unit_cost: float = 0.0
    reversible: bool = True
    latency_periods: int = 0
    ethical_note: str = ""

    def __post_init__(self) -> None:
        if self.min_shift > self.max_shift:
            raise ValueError(
                f"{self.node}: min_shift {self.min_shift} exceeds max_shift {self.max_shift}"
            )
        if self.fixed_cost < 0 or self.unit_cost < 0:
            raise ValueError(f"{self.node}: costs must be non-negative")
        if self.latency_periods < 0:
            raise ValueError(f"{self.node}: latency_periods must be non-negative")

    def allows(self, shift: float) -> bool:
        return self.min_shift <= shift <= self.max_shift

    def shift_cost(self, shift: float) -> float:
        """Cost of moving this node by ``shift``; untouched nodes are free."""
        if shift == 0.0:
            return 0.0
        return self.fixed_cost + self.unit_cost * abs(shift)


@dataclass(frozen=True)
class CostModel:
    """A reviewed-or-not sheet of what may be moved and what it costs."""

    specs: Mapping[str, ActionSpec]
    budget: float | None = None
    currency: str = "arbitrary units"
    provenance: str = ILLUSTRATIVE

    def __post_init__(self) -> None:
        for name, spec in self.specs.items():
            if name != spec.node:
                raise ValueError(f"Cost model key {name!r} disagrees with spec {spec.node!r}")
        if self.budget is not None and self.budget < 0:
            raise ValueError("Budget must be non-negative")

    def manipulable_nodes(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, spec in self.specs.items() if spec.manipulable))

    def validate_action(self, action: Mapping[str, float]) -> None:
        """Raise if any node is unknown, not manipulable, or moved too far."""
        for node, shift in action.items():
            spec = self.specs.get(node)
            if spec is None:
                raise ValueError(f"No cost specification for node: {node}")
            if shift == 0.0:
                continue
            if not spec.manipulable:
                raise ValueError(f"{node} is not manipulable")
            if not spec.allows(shift):
                raise ValueError(
                    f"{node}: shift {shift} is outside the allowed range "
                    f"[{spec.min_shift}, {spec.max_shift}]"
                )

    def cost(self, action: Mapping[str, float]) -> float:
        self.validate_action(action)
        return sum(self.specs[node].shift_cost(shift) for node, shift in action.items())

    def within_budget(self, action: Mapping[str, float]) -> bool:
        return self.budget is None or self.cost(action) <= self.budget

    def with_budget(self, budget: float | None) -> "CostModel":
        """Same sheet at a different budget, for sweeping the constraint."""
        return CostModel(
            specs=dict(self.specs),
            budget=budget,
            currency=self.currency,
            provenance=self.provenance,
        )

    @classmethod
    def from_csv(
        cls,
        path: Path,
        *,
        budget: float | None = None,
        currency: str = "arbitrary units",
        provenance: str = ILLUSTRATIVE,
    ) -> "CostModel":
        """Read a cost sheet with one row per node.

        Required column ``node``; every other ``ActionSpec`` field is optional
        and falls back to its default, so a sheet may list only what it knows.
        """
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Cost sheet {path} has no rows")
        specs: dict[str, ActionSpec] = {}
        for row in rows:
            node = (row.get("node") or "").strip()
            if not node:
                raise ValueError(f"Cost sheet {path} has a row with no node name")
            specs[node] = ActionSpec(
                node=node,
                manipulable=_as_bool(row.get("manipulable"), default=False),
                min_shift=_as_float(row.get("min_shift"), default=0.0),
                max_shift=_as_float(row.get("max_shift"), default=0.0),
                fixed_cost=_as_float(row.get("fixed_cost"), default=0.0),
                unit_cost=_as_float(row.get("unit_cost"), default=0.0),
                reversible=_as_bool(row.get("reversible"), default=True),
                latency_periods=int(_as_float(row.get("latency_periods"), default=0.0)),
                ethical_note=(row.get("ethical_note") or "").strip(),
            )
        return cls(specs=specs, budget=budget, currency=currency, provenance=provenance)


def _as_float(raw: str | None, *, default: float) -> float:
    text = (raw or "").strip()
    return default if not text else float(text)


def _as_bool(raw: str | None, *, default: bool) -> bool:
    text = (raw or "").strip().lower()
    if not text:
        return default
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"Cannot read {raw!r} as a boolean")
