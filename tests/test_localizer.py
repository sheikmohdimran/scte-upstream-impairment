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
