"""crucible.validation.gate — an audited, un-overridable pass/fail gate.

A :class:`Gate` collects sub-criteria (:class:`GateCheck`) and derives its
verdict *only* by AND-ing every HARD check. There is no setter for the overall
result and no override flag, so a failing hard check can never be talked past by
a later "but the strategy makes sense" argument — the reason this exists at all
is to stop a well-reasoned narrative from waiving a statistical fact. Soft checks
ride along in the same audit trail for context but never gate.

A :class:`Gauntlet` is an ordered run of named gates (crucible's REAL / STRONG /
DURABLE / GENERAL); it passes only if every gate passes. All capital-free — a
gate reasons over trade-log statistics, never over an equity curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class GateCheck:
    """One sub-criterion inside a gate.

    `hard` checks gate the verdict; `soft` checks are informational and recorded
    in the audit trail but never affect pass/fail.
    """
    name: str
    passed: bool
    value: Any = None
    threshold: Any = None
    hard: bool = True
    detail: str = ""


@dataclass
class Gate:
    """An audited AND-gate for one trial of edge validation.

    The verdict (:attr:`passed`) is derived exclusively by AND-ing every hard
    check. There is no way to set it directly and no override parameter, so a
    failed hard check cannot be silently waived. Soft checks are kept for context.
    """
    name: str
    checks: List[GateCheck] = field(default_factory=list)

    def add(self, name: str, passed: bool, value: Any = None,
            threshold: Any = None, hard: bool = True, detail: str = "") -> GateCheck:
        check = GateCheck(name=name, passed=bool(passed), value=value,
                          threshold=threshold, hard=hard, detail=detail)
        self.checks.append(check)
        return check

    @property
    def passed(self) -> bool:
        """True only if every hard check passed. No hard checks -> not a pass."""
        hard = [c for c in self.checks if c.hard]
        return bool(hard) and all(c.passed for c in hard)

    @property
    def failed_checks(self) -> List[GateCheck]:
        return [c for c in self.checks if c.hard and not c.passed]

    def audit_report(self) -> str:
        """Full audit trail: every sub-criterion with its value, threshold, and
        result, then the derived verdict — so a reader can verify by inspection
        that no failing hard check was overridden."""
        width = 62
        lines = ["=" * width, f"GATE: {self.name}", "=" * width]
        for c in self.checks:
            tag = "PASS" if c.passed else ("FAIL" if c.hard else "WARN")
            kind = "HARD" if c.hard else "soft"
            val = "" if c.value is None else f" value={_fmt(c.value)}"
            thr = "" if c.threshold is None else f" threshold={_fmt(c.threshold)}"
            lines.append(f"  [{tag}] ({kind}) {c.name}{val}{thr}")
            if c.detail:
                lines.append(f"         {c.detail}")
        lines.append("-" * width)
        n_hard = sum(1 for c in self.checks if c.hard)
        n_pass = sum(1 for c in self.checks if c.hard and c.passed)
        lines.append(f"  Hard checks passed: {n_pass}/{n_hard} "
                     f"(verdict = AND of all hard checks)")
        lines.append(f"  VERDICT: {'PASS' if self.passed else 'FAIL'}")
        if not self.passed and self.failed_checks:
            lines.append(f"  Failing: {', '.join(c.name for c in self.failed_checks)}")
        lines.append("=" * width)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializable audit record for reports/persistence."""
        return {
            "name": self.name,
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "value": c.value,
                 "threshold": c.threshold, "hard": c.hard, "detail": c.detail}
                for c in self.checks
            ],
        }


@dataclass
class Gauntlet:
    """An ordered run of named gates. Passes only if *every* gate passes — the
    same un-overridable AND at the gate level that :class:`Gate` applies to its
    checks, so an early failure can't be redeemed by a strong later gate."""
    gates: List[Gate] = field(default_factory=list)

    def add(self, gate: Gate) -> Gate:
        self.gates.append(gate)
        return gate

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(g.passed for g in self.gates)

    @property
    def failed_gates(self) -> List[Gate]:
        return [g for g in self.gates if not g.passed]

    def audit_report(self) -> str:
        parts = [g.audit_report() for g in self.gates]
        verdict = "PASS" if self.passed else "FAIL"
        summary = [f"GAUNTLET VERDICT: {verdict}  "
                   f"({sum(g.passed for g in self.gates)}/{len(self.gates)} gates passed)"]
        if self.failed_gates:
            summary.append("  Failing gates: " +
                           ", ".join(g.name for g in self.failed_gates))
        return "\n\n".join(parts + ["\n".join(summary)])

    def to_dict(self) -> dict:
        return {"passed": self.passed, "gates": [g.to_dict() for g in self.gates]}


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)
