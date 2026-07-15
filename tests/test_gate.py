import pytest

from crucible.validation import Gate, GateCheck, Gauntlet


def test_hard_checks_and_gate():
    g = Gate("REAL")
    g.add("a", True)
    g.add("b", True)
    assert g.passed is True
    g.add("c", False)
    assert g.passed is False
    assert [c.name for c in g.failed_checks] == ["c"]


def test_soft_check_never_gates():
    g = Gate("STRONG")
    g.add("hard_ok", True)
    g.add("soft_fail", False, hard=False)
    assert g.passed is True                 # a failing soft check does not gate
    assert g.failed_checks == []            # failed_checks is hard-only


def test_no_hard_checks_is_not_a_pass():
    g = Gate("EMPTY")
    assert g.passed is False                # nothing was actually proven
    g.add("only_soft", True, hard=False)
    assert g.passed is False


def test_add_returns_the_check_and_records_fields():
    g = Gate("STRONG")
    c = g.add("expectancy_ci_lower", True, value=0.12, threshold=0.0, detail="point=0.20")
    assert isinstance(c, GateCheck)
    assert (c.value, c.threshold, c.hard, c.detail) == (0.12, 0.0, True, "point=0.20")


def test_to_dict_and_audit_report():
    g = Gate("DURABLE")
    g.add("wfe_band", True, value=0.62, threshold=(0.30, 1.00))
    g.add("fold_dispersion", False, value=3.1, threshold=2.0)
    d = g.to_dict()
    assert d["name"] == "DURABLE"
    assert d["passed"] is False
    assert len(d["checks"]) == 2
    report = g.audit_report()
    assert "GATE: DURABLE" in report
    assert "VERDICT: FAIL" in report
    assert "fold_dispersion" in report


def test_gauntlet_passes_only_if_every_gate_passes():
    a = Gate("REAL"); a.add("x", True)
    b = Gate("STRONG"); b.add("y", True)
    gauntlet = Gauntlet()
    gauntlet.add(a)
    gauntlet.add(b)
    assert gauntlet.passed is True

    c = Gate("DURABLE"); c.add("z", False)
    gauntlet.add(c)
    assert gauntlet.passed is False
    assert [g.name for g in gauntlet.failed_gates] == ["DURABLE"]
    assert "GAUNTLET VERDICT: FAIL" in gauntlet.audit_report()


def test_empty_gauntlet_is_not_a_pass():
    assert Gauntlet().passed is False
