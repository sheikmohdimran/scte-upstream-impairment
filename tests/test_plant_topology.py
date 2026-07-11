"""Topology parsing + passive-aware localization tests.

Validates the data-package parser (:mod:`uil.localizer.plant_topology`) against BOTH real
CableLabs sample plants AND synthetic shapes not present in either sample, to prove the
parser generalizes to the *format* rather than overfitting the two example files.
"""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, RefResolver

from uil.domain.classification import Classification
from uil.domain.labels import ImpairmentLabel
from uil.domain.refs import AmpDeviceRef, RpdDeviceRef
from uil.localizer.graph_localizer import GraphLocalizer, Topology
from uil.localizer.plant_topology import InvalidRpdPortError, parse_data_package
from uil.mcp_server.server import MockMcpServer, Scenario

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
    root_id = parse_data_package(doc)["rootId"]
    root_component = next(
        c for c in doc["resources"][0]["data"]["components"] if c["id"] == root_id
    )
    first_port = next(
        c for c in doc["resources"][0]["data"]["components"]
        if c.get("type") == "RfPort" and c.get("name", "").startswith(root_component["name"])
    )
    topo = Topology.from_data_package(root_id, str(first_port["portId"]), doc)
    for amp_id in topo.amp_ids:
        path_ids = topo.path_to_rpd(amp_id)
        assert path_ids, f"{amp_id} has empty path"
        # last hop on the path is the root (path stops at node whose parent is None)
        assert topo.amps[path_ids[-1]].parentId is None
        assert path_ids[-1] == root_id


@pytest.mark.parametrize("path", _REAL_FILES, ids=[p.name for p in _REAL_FILES])
def test_real_topology_localizes_single_impaired_amp(path: Path) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    root_id = parse_data_package(doc)["rootId"]
    root_name = next(
        c["name"] for c in doc["resources"][0]["data"]["components"] if c["id"] == root_id
    )
    first_port = next(
        c for c in doc["resources"][0]["data"]["components"]
        if c.get("type") == "RfPort" and c.get("name", "").startswith(root_name)
    )
    port_id = str(first_port["portId"])
    topo = Topology.from_data_package(root_id, port_id, doc)
    amp_ids = topo.amp_ids
    target = amp_ids[0]
    classifications = [_clf(target, "impaired", ImpairmentLabel.CPD)] + [
        _clf(a, "clean", ImpairmentLabel.Clean) for a in amp_ids[1:]
    ]
    result = GraphLocalizer().localize(
        root_id, port_id, ImpairmentLabel.CPD, classifications, topo
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
        {"id": "P0", "type": "RfPort", "portId": 1, "name": "FN1 P1"},
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
    topo = Topology.from_data_package("FN", "1", _splitter_two_amp_pkg())
    # Both amps on different branches are impaired -> common point is the splitter SP.
    classifications = [
        _clf("AMPA", "impaired", ImpairmentLabel.CPD),
        _clf("AMPB", "impaired", ImpairmentLabel.CPD),
    ]
    result = GraphLocalizer().localize(
        "FN", "1", ImpairmentLabel.CPD, classifications, topo
    )
    likely = result.likelySourceLocation
    assert likely is not None
    assert likely.locationType == "branch"
    assert isinstance(likely.upstreamBoundaryDevice, RpdDeviceRef)
    assert likely.upstreamBoundaryDevice.rpdId == "FN"
    assert "SP" in likely.description


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
    comps = [
        {"id": "FN", "type": "RfSource", "name": "FN1"},
        {"id": "P0", "type": "RfPort", "portId": 1, "name": "FN1 P1"},
    ]
    edges = [{"source": "FN", "target": "P0"}]
    prev = "P0"
    for i in range(1, 6):
        cable, amp = f"C{i}", f"A{i}"
        comps += [
            {"id": cable, "type": "RfCable", "length": "10"},
            {"id": amp, "type": "RfAmp", "name": f"AMP{i}"},
        ]
        edges += [{"source": prev, "target": cable}, {"source": cable, "target": amp}]
        prev = amp
    topo = Topology.from_data_package("FN", "1", _pkg(comps, edges))
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
        {"id": "P0", "type": "RfPort", "portId": 1, "name": "FN1 P1"},
        {"id": "X", "type": "RfMysteryDevice"},  # unknown -> pass-through
        {"id": "A1", "type": "RfAmp", "name": "AMP1"},
    ]
    edges = [
        {"source": "FN", "target": "P0"},
        {"source": "P0", "target": "X"},
        {"source": "X", "target": "A1"},
    ]
    topo = Topology.from_data_package("FN", "1", _pkg(comps, edges))
    assert "A1" in topo.amp_ids
    assert topo.amps["A1"].parentId == "FN"  # unknown node collapsed away


def _multi_port_pkg() -> dict:
    components = [
        {"id": "RPD-A", "type": "RfSource", "name": "RPD-A"},
        {"id": "A-P1", "type": "RfPort", "portId": 1, "name": "RPD-A Port 1"},
        {"id": "A-P2", "type": "RfPort", "portId": 2, "name": "RPD-A Port 2"},
        {"id": "A1", "type": "RfAmp", "name": "AMP-A1"},
        {"id": "A2", "type": "RfAmp", "name": "AMP-A2"},
        {"id": "RPD-B", "type": "RfSource", "name": "RPD-B"},
        {"id": "B-P1", "type": "RfPort", "portId": 1, "name": "RPD-B Port 1"},
        {"id": "B1", "type": "RfAmp", "name": "AMP-B1"},
    ]
    edges = [
        {"source": "RPD-A", "target": "A-P1"},
        {"source": "A-P1", "target": "A1"},
        {"source": "RPD-A", "target": "A-P2"},
        {"source": "A-P2", "target": "A2"},
        {"source": "RPD-B", "target": "B-P1"},
        {"source": "B-P1", "target": "B1"},
    ]
    return _pkg(components, edges)


def test_data_package_scopes_to_requested_rpd_port() -> None:
    topo = Topology.from_data_package("RPD-A", "2", _multi_port_pkg())
    assert topo.amp_ids == ["A2"]
    assert topo.path_to_rpd("A2")[-1] == "RPD-A"


@pytest.mark.parametrize(
    ("rpd_id", "port_id"),
    [("missing", "1"), ("RPD-A", "99")],
)
def test_data_package_rejects_unknown_rpd_or_port(rpd_id: str, port_id: str) -> None:
    with pytest.raises(InvalidRpdPortError):
        Topology.from_data_package(rpd_id, port_id, _multi_port_pkg())


def test_topology_amp_list_handle_chains_into_measurement() -> None:
    scenario = Scenario(
        rpdId="FN",
        portId="1",
        rpd_label=ImpairmentLabel.CPD,
        amps=[
            {"ampId": "AMPA", "label": "CPD"},
            {"ampId": "AMPB", "label": "CPD"},
        ],
        topology_doc=_splitter_two_amp_pkg(),
    )
    server = MockMcpServer(scenario)
    amp_list = server.getAllAmpsInSegment("FN", "1")
    stored = server.store.get(amp_list["ampListRef"])
    assert stored["ampIds"] == ["AMPA", "AMPB"]

    measured = server.getAmpUpstreamSpectrumMeasurements(ampListRef=amp_list["ampListRef"])
    assert measured["status"] == "success"
    assert measured["measurementCount"] == 2


def test_topology_tool_maps_unknown_port_to_contract_error() -> None:
    scenario = Scenario(
        rpdId="RPD-A",
        portId="1",
        rpd_label=ImpairmentLabel.CPD,
        amps=[],
        topology_doc=_multi_port_pkg(),
    )
    result = MockMcpServer(scenario).getAllAmpsInSegment("RPD-A", "99")
    assert result["status"] == "error"
    assert result["errorCode"] == "INVALID_RPD_PORT"


def test_passive_common_point_output_conforms_to_device_ref_schema() -> None:
    import itertools

    from uil.domain.localization import build_public_localization

    topo = Topology.from_data_package("FN", "1", _splitter_two_amp_pkg())
    internal = GraphLocalizer().localize(
        "FN",
        "1",
        ImpairmentLabel.CPD,
        [
            _clf("AMPA", "impaired", ImpairmentLabel.CPD),
            _clf("AMPB", "impaired", ImpairmentLabel.CPD),
        ],
        topo,
    )
    _counter = itertools.count(1)
    result = build_public_localization(
        "FN", "1", internal, lambda prefix, value: f"{prefix}-{next(_counter):04d}"
    ).model_dump(mode="json", exclude_none=True)

    schema_dir = Path(__file__).resolve().parents[1] / "schemas" / "tools"
    defs = json.loads((schema_dir / "_defs.schema.json").read_text(encoding="utf-8"))
    schema = json.loads(
        (schema_dir / "reference" / "localizeUpstreamSpectrumImpairmentSource.schema.json")
        .read_text(encoding="utf-8")
    )["outputSchema"]
    resolver = RefResolver(
        base_uri="",
        referrer=defs,
        store={"../_defs.schema.json": defs, "_defs.schema.json": defs},
    )
    Draft202012Validator(schema, resolver=resolver).validate(result)


# ── Real-plant, full-scale end-to-end (deterministic; independent oracle) ─────
def _amp_subtree(topo: Topology, fault: str) -> set[str]:
    """Amp ids in fault's downstream subtree (fault ∪ amp-descendants) within `topo`."""
    kids: dict[str, list[str]] = {}
    for nid, n in topo.amps.items():
        if n.parentId:
            kids.setdefault(n.parentId, []).append(nid)
    seen: set[str] = set()
    out: set[str] = set()
    stack = [fault]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if topo.amps[cur].deviceType == "AMP":
            out.add(cur)
        stack.extend(kids.get(cur, []))
    return out


def _amp_ancestors(topo: Topology, amp: str) -> set[str]:
    """Measured (AMP) ancestors of `amp` on the path toward the RPD root."""
    out: set[str] = set()
    cur = topo.amps[amp].parentId
    seen: set[str] = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        node = topo.amps.get(cur)
        if node is None:
            break
        if node.deviceType == "AMP":
            out.add(cur)
        cur = node.parentId
    return out


def _pick_fault_amp(topo: Topology) -> str:
    """Deterministically pick a discriminating fault amp.

    Prefer an amp that has BOTH a measured (AMP) ancestor (so a clean upstream boundary exists)
    and amp-descendants (a branch fault), then the largest such subtree; fall back gracefully.
    Stable tiebreak on amp id keeps the choice reproducible.
    """
    def key(a: str) -> tuple:
        anc = len(_amp_ancestors(topo, a)) > 0
        desc = len(_amp_subtree(topo, a)) - 1  # exclude self
        return (anc, desc > 0, desc, a)

    return max(topo.amp_ids, key=key)


# Every real RPD-port segment across both plants (plant-1: 74 amps; plant-2: 47 amps).
_REAL_SEGMENTS = [
    ("network-topology-example-1.json", "1", 56),
    ("network-topology-example-1.json", "2", 2),
    ("network-topology-example-1.json", "3", 16),
    ("network-topology-example-2.json", "1", 2),
    ("network-topology-example-2.json", "2", 20),
    ("network-topology-example-2.json", "3", 20),
    ("network-topology-example-2.json", "4", 5),
]


@pytest.mark.parametrize(
    ("fname", "port", "expected_amps"),
    _REAL_SEGMENTS,
    ids=[f"{f.split('-')[-1].split('.')[0]}-p{p}" for f, p, _ in _REAL_SEGMENTS],
)
def test_real_plant_e2e_localizes_per_port(fname: str, port: str, expected_amps: int) -> None:
    """Full 6-step chain over EVERY real RPD-port segment of both CableLabs plants.

    Independent (non-circular) oracle: a fault is planted at a deterministically chosen amp, so
    that amp and its amp-descendants are impaired while upstream / other-branch amps stay clean.
    The localized boundary must equal the planted subtree and exclude the clean upstream amps.
    This closes the "full real plant, per-amp injection" gap for both plants (74 + 47 amps).
    """
    from uil.agent.orchestrator import Orchestrator

    root = "0000000001"
    doc = json.loads((_FIX / fname).read_text(encoding="utf-8"))
    topo = Topology.from_data_package(root, port, doc)
    amp_ids = topo.amp_ids
    assert len(amp_ids) == expected_amps

    fault = _pick_fault_amp(topo)
    impaired = _amp_subtree(topo, fault)
    clean_upstream = _amp_ancestors(topo, fault)
    assert fault in impaired
    assert clean_upstream.isdisjoint(impaired)

    scenario = Scenario(
        rpdId=root, portId=port, rpd_label=ImpairmentLabel.CPD,
        amps=[{"ampId": a, "label": "CPD" if a in impaired else "Clean"} for a in amp_ids],
        topology_doc=doc,
    )
    server = MockMcpServer(scenario)
    result = Orchestrator(server, scenario_name=f"real-{fname}-p{port}").run()

    assert result.status == "localized"
    assert [c.tool for c in result.trace.calls] == [
        "getRPDSpectrumMeasurements",
        "analyzeSpectrumMeasurements",
        "getAllAmpsInSegment",
        "getAmpUpstreamSpectrumMeasurements",
        "analyzeSpectrumMeasurements",
        "localizeUpstreamSpectrumImpairmentSource",
    ]
    # Tool 3 exposed only the count; the amp ids stayed behind the handle.
    t3 = next(c for c in result.trace.calls if c.tool == "getAllAmpsInSegment")
    assert t3.result["ampCount"] == expected_amps
    assert set(t3.result.keys()) <= {"status", "ampListRef", "ampCount"}

    loc = server.resolve_localization(result.localization)
    supporting = {d["ampId"] for d in loc["supportingDevices"]}
    assert supporting == impaired                       # exactly the planted subtree
    if clean_upstream:                                  # discriminating boundary present
        assert supporting.isdisjoint(clean_upstream)    # clean upstream not implicated


