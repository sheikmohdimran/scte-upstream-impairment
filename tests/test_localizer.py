"""Step 6 — graph-theory localizer."""

from uil.domain.classification import Classification
from uil.domain.labels import ImpairmentLabel
from uil.localizer.graph_localizer import GraphLocalizer, Topology


def _amp_class(amp_id: str, status: str) -> Classification:
    label = ImpairmentLabel.CPD if status == "impaired" else ImpairmentLabel.Clean
    return Classification(deviceType="AMP", measurementId=f"m-{amp_id}", status=status,
                          **{"class": label}, confidence=0.9, ampId=amp_id)


def _topo() -> Topology:
    return Topology.from_amp_list("RPD-1", "P1", [
        {"ampId": "A1", "parentId": None},
        {"ampId": "A2", "parentId": "A1"},
        {"ampId": "A3", "parentId": "A2"},
        {"ampId": "A4", "parentId": "A1"},
    ])


def test_single_branch_localizes_to_span_below_clean_boundary() -> None:
    classifications = [
        _amp_class("A1", "clean"),
        _amp_class("A2", "impaired"),
        _amp_class("A3", "impaired"),
        _amp_class("A4", "clean"),
    ]
    res = GraphLocalizer().localize("RPD-1", "P1", ImpairmentLabel.CPD, classifications, _topo())
    assert res.localizationStatus == "localized"
    # A1 is the clean upstream boundary feeding the impaired A2/A3 cluster.
    assert any(d.ampId == "A1" for d in res.cleanBoundaryDevices)
    assert {d.ampId for d in res.supportingDevices} == {"A2", "A3"}


def test_missing_measurement_lowers_confidence() -> None:
    # A3 not measured -> uncertain -> low_confidence + re-measure suggestion
    classifications = [
        _amp_class("A1", "clean"),
        _amp_class("A2", "impaired"),
        _amp_class("A4", "clean"),
    ]
    res = GraphLocalizer().localize("RPD-1", "P1", ImpairmentLabel.CPD, classifications, _topo())
    assert any(d.ampId == "A3" for d in res.uncertainDevices)
    assert res.localizationStatus == "low_confidence"
    assert res.recommendedNextAction and res.recommendedNextAction.action == "re-measure"


def test_conflicting_labels_escalate() -> None:
    # Two different impairment types among impaired amps -> common-point analysis unreliable.
    classifications = [
        _amp_class("A1", "clean"),
        Classification(deviceType="AMP", measurementId="m-A2", status="impaired",
                       **{"class": ImpairmentLabel.CPD}, confidence=0.9, ampId="A2"),
        Classification(deviceType="AMP", measurementId="m-A3", status="impaired",
                       **{"class": ImpairmentLabel.Ingress}, confidence=0.88, ampId="A3"),
        _amp_class("A4", "clean"),
    ]
    res = GraphLocalizer().localize("RPD-1", "P1", ImpairmentLabel.CPD, classifications, _topo())
    assert res.localizationStatus == "low_confidence"
    assert res.recommendedNextAction and res.recommendedNextAction.action == "escalate"
    assert {d.ampId for d in res.supportingDevices} == {"A2", "A3"}


def test_flat_topology_without_parent_links_is_handled() -> None:
    flat = Topology.from_amp_list("RPD-1", "P1", [
        {"ampId": "A1", "distanceFromRpdMeters": 100},
        {"ampId": "A2", "distanceFromRpdMeters": 200},
    ])
    classifications = [
        _amp_class("A1", "clean"),
        _amp_class("A2", "impaired"),
    ]
    res = GraphLocalizer().localize("RPD-1", "P1", ImpairmentLabel.CPD, classifications, flat)
    assert res.localizationStatus in {"localized", "low_confidence"}


# ── New topology scenarios ─────────────────────────────────────────────────────

def _linear_topo(n: int) -> Topology:
    """Linear chain: A1 -> A2 -> ... -> An (no branching)."""
    amps = [{"ampId": f"A{i}", "parentId": f"A{i-1}" if i > 1 else None}
            for i in range(1, n + 1)]
    return Topology.from_amp_list("RPD-1", "P1", amps)


def _star_topo(n_leaves: int) -> Topology:
    """Star: A1(root) -> A2, A3, ..., A{n_leaves+1} (all direct children of A1)."""
    amps = [{"ampId": "A1", "parentId": None}] + [
        {"ampId": f"A{i}", "parentId": "A1"} for i in range(2, n_leaves + 2)
    ]
    return Topology.from_amp_list("RPD-1", "P1", amps)


def test_linear_chain_fault_at_end_common_point() -> None:
    """A1→A2→A3→A4→A5, fault at A3/A4/A5 — LCA is A3, cleanBoundary=[A2]."""
    topo = _linear_topo(5)
    classifications = [
        _amp_class("A1", "clean"),
        _amp_class("A2", "clean"),
        _amp_class("A3", "impaired"),
        _amp_class("A4", "impaired"),
        _amp_class("A5", "impaired"),
    ]
    res = GraphLocalizer().localize("RPD-1", "P1", ImpairmentLabel.CPD, classifications, topo)
    assert res.localizationStatus == "localized"
    # LCA of {A3,A4,A5} on linear path = A3; A2 is the clean boundary
    assert {d.ampId for d in res.cleanBoundaryDevices} == {"A2"}
    assert {d.ampId for d in res.supportingDevices} == {"A3", "A4", "A5"}


def test_all_amps_impaired_localizes_to_root() -> None:
    """A1(root)→[A2,A3], all three impaired — common point = A1, branch at A1."""
    topo = Topology.from_amp_list("RPD-1", "P1", [
        {"ampId": "A1", "parentId": None},
        {"ampId": "A2", "parentId": "A1"},
        {"ampId": "A3", "parentId": "A1"},
    ])
    classifications = [
        _amp_class("A1", "impaired"),
        _amp_class("A2", "impaired"),
        _amp_class("A3", "impaired"),
    ]
    res = GraphLocalizer().localize("RPD-1", "P1", ImpairmentLabel.CPD, classifications, topo)
    assert res.localizationStatus == "localized"
    assert res.likelySourceLocation is not None
    # Common point is A1 (the root); all three amps see the fault
    assert {d.ampId for d in res.supportingDevices} == {"A1", "A2", "A3"}


def test_single_amp_segment_impaired_localizes_to_device() -> None:
    """Only one amp in the segment, and it is impaired."""
    topo = Topology.from_amp_list("RPD-1", "P1", [{"ampId": "A1", "parentId": None}])
    classifications = [_amp_class("A1", "impaired")]
    res = GraphLocalizer().localize("RPD-1", "P1", ImpairmentLabel.CPD, classifications, topo)
    assert res.localizationStatus == "localized"
    assert res.likelySourceLocation.locationType == "device"
    assert res.likelySourceLocation.downstreamBoundaryDevice.ampId == "A1"


def test_wide_star_single_leaf_impaired() -> None:
    """A1→[A2,A3,A4,A5], only A2 impaired — single device, cleanBoundary=[A1]."""
    topo = _star_topo(n_leaves=4)       # A1, A2, A3, A4, A5
    classifications = [
        _amp_class("A1", "clean"),
        _amp_class("A2", "impaired"),
        _amp_class("A3", "clean"),
        _amp_class("A4", "clean"),
        _amp_class("A5", "clean"),
    ]
    res = GraphLocalizer().localize("RPD-1", "P1", ImpairmentLabel.CPD, classifications, topo)
    assert res.localizationStatus == "localized"
    assert res.likelySourceLocation.locationType == "device"
    assert res.likelySourceLocation.downstreamBoundaryDevice.ampId == "A2"
    assert any(d.ampId == "A1" for d in res.cleanBoundaryDevices)


def test_all_amps_clean_returns_low_confidence() -> None:
    """No impaired amps measured (even though RPD sees fault) — re-measure suggested."""
    classifications = [
        _amp_class("A1", "clean"),
        _amp_class("A2", "clean"),
        _amp_class("A3", "clean"),
        _amp_class("A4", "clean"),
    ]
    res = GraphLocalizer().localize("RPD-1", "P1", ImpairmentLabel.CPD, classifications, _topo())
    assert res.localizationStatus == "low_confidence"
    assert res.recommendedNextAction and res.recommendedNextAction.action == "re-measure"


def test_leaf_only_fault_localizes_to_device() -> None:
    """Deepest amp in a linear chain is the only impaired device."""
    topo = _linear_topo(4)     # A1→A2→A3→A4
    classifications = [
        _amp_class("A1", "clean"),
        _amp_class("A2", "clean"),
        _amp_class("A3", "clean"),
        _amp_class("A4", "impaired"),
    ]
    res = GraphLocalizer().localize("RPD-1", "P1", ImpairmentLabel.CPD, classifications, topo)
    assert res.localizationStatus == "localized"
    assert res.likelySourceLocation.locationType == "device"
    assert res.likelySourceLocation.downstreamBoundaryDevice.ampId == "A4"
    assert any(d.ampId == "A3" for d in res.cleanBoundaryDevices)
