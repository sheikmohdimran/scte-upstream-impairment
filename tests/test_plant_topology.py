"""Topology parsing + passive-aware localization tests.

Validates the data-package parser (:mod:`uil.localizer.plant_topology`) against BOTH real
CableLabs sample plants AND synthetic shapes not present in either sample, to prove the
parser generalizes to the *format* rather than overfitting the two example files.
"""

import json
from pathlib import Path

import pytest

from uil.domain.classification import Classification
from uil.domain.labels import ImpairmentLabel
from uil.domain.refs import AmpDeviceRef, PlantDeviceRef
from uil.localizer.graph_localizer import GraphLocalizer, Topology
from uil.localizer.plant_topology import parse_data_package

_FIX = Path(__file__).resolve().parent / "fixtures"
_REAL_FILES = [
    _FIX / "network-topology-example-1.json",
    _FIX / "network-topology-example-2.json",
]


def _clf(amp_id: str, status: str, klass: ImpairmentLabel) -> Classification:
    return Classification(
        deviceType="AMP", measurementId=f"m-{amp_id}", status=status,
        **{"class": klass}, confidence=0.95, ampId=amp_id,
    )


# ── Real sample files ────────────────────────────────────────────────────────
@pytest.mark.parametrize("path", _REAL_FILES, ids=[p.name for p in _REAL_FILES])
def test_real_topology_parses_into_rooted_graph(path: Path) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    parsed = parse_data_package(doc)
    nodes = parsed["nodes"]

    assert parsed["rootId"] is not None
    amps = [n for n in nodes.values() if n.deviceType == "AMP"]
    roots = [n for n in nodes.values() if n.deviceType == "RPD"]
    assert len(roots) == 1
    assert len(amps) > 0

    # No orphans: every non-root node's parent resolves inside the graph.
    orphans = [
        n.id for n in nodes.values()
        if n.deviceType != "RPD" and (n.parentId is None or n.parentId not in nodes)
    ]
    assert orphans == []

    # Connectors/leaves are collapsed away — only amps/passives/root remain.
    assert all(n.deviceType in {"AMP", "RPD"} or n.deviceType.startswith("Rf") for n in nodes.values())
    assert not any(n.deviceType in {"RfPort", "RfCable", "device"} for n in nodes.values())


@pytest.mark.parametrize("path", _REAL_FILES, ids=[p.name for p in _REAL_FILES])
def test_real_topology_every_amp_path_reaches_root(path: Path) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    topo = Topology.from_data_package("RPD-1", "1", doc)
    root_id = parse_data_package(doc)["rootId"]
    for amp_id in topo.amp_ids:
        path_ids = topo.path_to_rpd(amp_id)
        assert path_ids, f"{amp_id} has empty path"
        # last hop on the path is the root (path stops at node whose parent is None)
        assert topo.amps[path_ids[-1]].parentId is None
        assert path_ids[-1] == root_id


@pytest.mark.parametrize("path", _REAL_FILES, ids=[p.name for p in _REAL_FILES])
def test_real_topology_localizes_single_impaired_amp(path: Path) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    topo = Topology.from_data_package("RPD-1", "1", doc)
    amp_ids = topo.amp_ids
    target = amp_ids[0]
    classifications = [_clf(target, "impaired", ImpairmentLabel.CPD)] + [
        _clf(a, "clean", ImpairmentLabel.Clean) for a in amp_ids[1:]
    ]
    result = GraphLocalizer().localize(
        "RPD-1", "1", ImpairmentLabel.CPD, classifications, topo
    )
    assert result.likelySourceLocation is not None
    assert result.impairmentType == ImpairmentLabel.CPD


# ── Synthetic: passive common point ──────────────────────────────────────────
def _pkg(components: list[dict], edges: list[dict]) -> dict:
    return {"resources": [{"data": {"components": components, "edges": edges}}]}


def _splitter_two_amp_pkg() -> dict:
    # FN1 -> port -> cable -> SP1(splitter) -> {branch A: cable->AMP_A, branch B: cable->AMP_B}
    comps = [
        {"id": "FN", "type": "RfSource", "name": "FN1"},
        {"id": "P0", "type": "RfPort", "name": "FN1 P1"},
        {"id": "C0", "type": "RfCable", "length": "100"},
        {"id": "SP", "type": "RfSplitter", "name": "SP1"},
        {"id": "CA", "type": "RfCable", "length": "50"},
        {"id": "AMPA", "type": "RfAmp", "name": "AMP_A"},
        {"id": "CB", "type": "RfCable", "length": "60"},
        {"id": "AMPB", "type": "RfAmp", "name": "AMP_B"},
        {"id": "HOME", "type": "device", "name": "Home1"},
    ]
    edges = [
        {"source": "FN", "target": "P0"},
        {"source": "P0", "target": "C0"},
        {"source": "C0", "target": "SP"},
        {"source": "SP", "target": "CA"},
        {"source": "CA", "target": "AMPA"},
        {"source": "SP", "target": "CB"},
        {"source": "CB", "target": "AMPB"},
        {"source": "AMPA", "target": "HOME"},
    ]
    return _pkg(comps, edges)


def test_common_point_can_be_a_passive_device() -> None:
    topo = Topology.from_data_package("RPD-9", "1", _splitter_two_amp_pkg())
    # Both amps on different branches are impaired -> common point is the splitter SP.
    classifications = [
        _clf("AMPA", "impaired", ImpairmentLabel.CPD),
        _clf("AMPB", "impaired", ImpairmentLabel.CPD),
    ]
    result = GraphLocalizer().localize(
        "RPD-9", "1", ImpairmentLabel.CPD, classifications, topo
    )
    likely = result.likelySourceLocation
    assert likely is not None
    assert likely.locationType == "branch"
    assert isinstance(likely.upstreamBoundaryDevice, PlantDeviceRef)
    assert likely.upstreamBoundaryDevice.deviceType == "RfSplitter"
    assert likely.upstreamBoundaryDevice.name == "SP1"


def test_homes_and_connectors_are_collapsed() -> None:
    parsed = parse_data_package(_splitter_two_amp_pkg())
    types = {n.deviceType for n in parsed["nodes"].values()}
    assert "device" not in types  # subscriber home dropped
    assert "RfPort" not in types and "RfCable" not in types  # connectors collapsed
    assert {"RPD", "AMP", "RfSplitter"} <= types


def test_cable_length_accumulates_into_distance() -> None:
    parsed = parse_data_package(_splitter_two_amp_pkg())
    nodes = parsed["nodes"]
    # AMP_A distance = C0(100) to splitter + CA(50) = 150
    ampa = next(n for n in nodes.values() if n.name == "AMP_A")
    assert ampa.distanceFromRpdMeters == pytest.approx(150.0)


# ── Synthetic generalization shapes ──────────────────────────────────────────
def test_linear_chain_generalizes() -> None:
    comps = [{"id": "FN", "type": "RfSource", "name": "FN1"}]
    edges = []
    prev = "FN"
    for i in range(1, 6):
        cable, amp = f"C{i}", f"A{i}"
        comps += [
            {"id": cable, "type": "RfCable", "length": "10"},
            {"id": amp, "type": "RfAmp", "name": f"AMP{i}"},
        ]
        edges += [{"source": prev, "target": cable}, {"source": cable, "target": amp}]
        prev = amp
    topo = Topology.from_data_package("RPD-1", "1", _pkg(comps, edges))
    assert len(topo.amp_ids) == 5
    # deepest amp path should traverse all 5 amps + root
    deepest = "A5"
    assert len(topo.path_to_rpd(deepest)) == 6  # A5..A1 + FN


def test_empty_or_malformed_document_raises() -> None:
    with pytest.raises(ValueError):
        parse_data_package({"nope": True})


def test_unknown_component_type_is_tolerated_as_connector() -> None:
    comps = [
        {"id": "FN", "type": "RfSource", "name": "FN1"},
        {"id": "X", "type": "RfMysteryDevice"},  # unknown -> pass-through
        {"id": "A1", "type": "RfAmp", "name": "AMP1"},
    ]
    edges = [{"source": "FN", "target": "X"}, {"source": "X", "target": "A1"}]
    topo = Topology.from_data_package("RPD-1", "1", _pkg(comps, edges))
    assert "A1" in topo.amp_ids
    assert topo.amps["A1"].parentId == "FN"  # unknown node collapsed away
