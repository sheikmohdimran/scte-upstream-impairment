"""In-process mock MCP server: the 5 reference-surface tools (Steps 2 + 7).

Returns dicts shaped exactly like each tool's reference ``outputSchema``
(``oneOf(success | error[ | partial_success])``). Raw spectra live in the
:class:`HandleStore`; the SLM only ever receives handles + counts.

Fault injection (Step 7) lets tests force ``error`` / ``partial_success`` paths so the
orchestrator's recovery logic — the paper's real value-prop — can be exercised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from uil.classifier.rule_classifier import RuleClassifier
from uil.domain.labels import ImpairmentLabel
from uil.localizer.graph_localizer import GraphLocalizer, Topology
from uil.mcp_server.handle_store import HandleStore
from uil.sim.spectrum_simulator import SpectrumSimulator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FaultInjection:
    rpd_unreachable: bool = False
    rpd_stale_only: bool = False
    topology_unavailable: bool = False
    amp_failed_ports: set[str] = field(default_factory=set)  # ampIds that fail to measure
    analyze_fails: bool = False


@dataclass
class Scenario:
    rpdId: str
    portId: str
    rpd_label: ImpairmentLabel
    amps: list[dict]  # each: {ampId, parentId, label}
    faults: FaultInjection = field(default_factory=FaultInjection)


class MockMcpServer:
    def __init__(self, scenario: Scenario, seed: int = 42) -> None:
        self.scn = scenario
        self.store = HandleStore()
        self.sim = SpectrumSimulator(seed=seed)
        self.clf = RuleClassifier()
        self.localizer = GraphLocalizer()
        self._labels = {a["ampId"]: ImpairmentLabel(a.get("label", "Clean")) for a in scenario.amps}

    # ---- Tool 1 ---------------------------------------------------------
    def getRPDSpectrumMeasurements(self, rpdId: str, portId: str, numBins: int = 256,
                                   startFrequencyHz: int = 5_000_000, stopFrequencyHz: int = 85_000_000) -> dict:
        if self.scn.faults.rpd_unreachable:
            return {"status": "error", "errorCode": "DEVICE_UNREACHABLE", "message": f"{rpdId}/{portId} unreachable"}
        if self.scn.faults.rpd_stale_only:
            return {"status": "error", "errorCode": "STALE_DATA_ONLY"}
        spectrum = self.sim.generate(self.scn.rpd_label)
        ts = _now()
        meas_id = self.store.put("meas", {
            "deviceType": "RPD", "rpdId": rpdId, "portId": portId,
            "timestamp": ts, "spectrum": spectrum,
        })
        return {
            "status": "success",
            "measurementRef": {
                "measurementId": meas_id, "deviceType": "RPD",
                "measurementType": "upstream_spectrum", "timestamp": ts,
                "rpdId": rpdId, "portId": portId,
            },
        }

    # ---- Tool 2 / 5 -----------------------------------------------------
    def analyzeSpectrumMeasurements(self, measurementRefs: list[dict] | None = None,
                                    measurementSetRef: str | None = None) -> dict:
        if self.scn.faults.analyze_fails:
            return {"status": "error", "errorCode": "CLASSIFICATION_FAILED"}
        if (measurementRefs is None) == (measurementSetRef is None):
            return {"status": "error", "errorCode": "INVALID_MEASUREMENT_REF",
                    "message": "exactly one of measurementRefs|measurementSetRef required"}

        raw_items: list[dict] = []
        if measurementSetRef is not None:
            if not self.store.has(measurementSetRef):
                return {"status": "error", "errorCode": "INVALID_MEASUREMENT_REF"}
            raw_items = list(self.store.get(measurementSetRef))
        else:
            for ref in measurementRefs:  # type: ignore[union-attr]
                mid = ref.get("measurementId")
                if not mid or not self.store.has(mid):
                    return {"status": "error", "errorCode": "INVALID_MEASUREMENT_REF"}
                raw_items.append(self.store.get(mid))

        classifications = []
        for item in raw_items:
            c = self.clf.classify_spectrum(
                item["spectrum"],
                device_type=item["deviceType"],
                measurement_id=item.get("_mid", item.get("measurementId", "?")),
                rpd_id=item.get("rpdId"), port_id=item.get("portId"), amp_id=item.get("ampId"),
            )
            classifications.append(c)

        set_ref = self.store.put("classset", classifications)
        impaired = [c for c in classifications if c.status == "impaired"]
        classes_present = sorted({c.klass.value for c in classifications})
        return {
            "status": "success",
            "classificationSetRef": set_ref,
            "classificationCount": len(classifications),
            "impairedCount": len(impaired),
            "classesPresent": classes_present,
        }

    # ---- Tool 3 ---------------------------------------------------------
    def getAllAmpsInSegment(self, rpdId: str, portId: str) -> dict:
        if self.scn.faults.topology_unavailable:
            return {"status": "error", "errorCode": "TOPOLOGY_UNAVAILABLE"}
        if not self.scn.amps:
            return {"status": "error", "errorCode": "EMPTY_SEGMENT"}
        amp_records = [{
            "ampId": a["ampId"], "parentId": a.get("parentId"),
            "children": a.get("children", []),
            "distanceFromRpdMeters": a.get("distanceFromRpdMeters"),
        } for a in self.scn.amps]
        amp_list_ref = self.store.put("amplist", {"rpdId": rpdId, "portId": portId, "amps": amp_records})
        return {
            "status": "success",
            "segmentId": f"seg-{rpdId}-{portId}",
            "ampListRef": amp_list_ref,
            "ampCount": len(amp_records),
            "topologyTimestamp": _now(),
        }

    # ---- Tool 4 ---------------------------------------------------------
    def getAmpSpectrumMeasurements(self, ampIds: list[str] | None = None, ampListRef: str | None = None,
                                   numBins: int = 256, startFrequencyHz: int = 5_000_000,
                                   stopFrequencyHz: int = 85_000_000) -> dict:
        if (ampIds is None) == (ampListRef is None):
            return {"status": "error", "errorCode": "INVALID_AMP_LIST_REF",
                    "message": "exactly one of ampIds|ampListRef required"}
        if ampListRef is not None:
            if not self.store.has(ampListRef):
                return {"status": "error", "errorCode": "INVALID_AMP_LIST_REF"}
            ids = [a["ampId"] for a in self.store.get(ampListRef)["amps"]]
        else:
            ids = list(ampIds)  # type: ignore[arg-type]

        measurements, failed = [], []
        for amp_id in ids:
            if amp_id in self.scn.faults.amp_failed_ports:
                failed.append({"ampId": amp_id, "errorCode": "DEVICE_UNREACHABLE"})
                continue
            if amp_id not in self._labels:
                failed.append({"ampId": amp_id, "errorCode": "INVALID_AMP_ID"})
                continue
            spectrum = self.sim.generate(self._labels[amp_id])
            ts = _now()
            mid = self.store.put("meas", {
                "deviceType": "AMP", "ampId": amp_id, "portId": "0",
                "timestamp": ts, "spectrum": spectrum,
            })
            # echo the measurementId into the stored payload so analyze can label it
            self.store.get(mid)["_mid"] = mid
            measurements.append(mid)

        if not measurements:
            return {"status": "error", "errorCode": "AMP_MEASUREMENTS_UNAVAILABLE"}

        set_ref = self.store.put("measset", [self.store.get(mid) for mid in measurements])
        result = {
            "status": "partial_success" if failed else "success",
            "measurementSetRef": set_ref,
            "measurementCount": len(measurements),
            "failedCount": len(failed),
        }
        if failed:
            result["failedDevicesRef"] = self.store.put("failed", failed)
        return result

    # ---- Tool 6 ---------------------------------------------------------
    def localizeUpstreamSpectrumImpairmentSource(self, rpdId: str, portId: str, impairmentType: str,
                                                 classificationSetRefs: list[str]) -> dict:
        invalid = [r for r in classificationSetRefs if not self.store.has(r)]
        if invalid:
            return {"status": "error", "errorCode": "INVALID_CLASSIFICATION_SET_REF",
                    "invalidClassificationSetRefs": invalid}
        pooled = []
        for r in classificationSetRefs:
            pooled.extend(self.store.get(r))
        if not any(c.status == "impaired" for c in pooled):
            return {"status": "error", "errorCode": "NO_IMPAIRMENT_CONFIRMED"}

        amp_records = None
        for handle in self.store._data:  # find the topology we built in tool 3
            if handle.startswith("amplist"):
                amp_records = self.store.get(handle)["amps"]
                break
        if amp_records is None:
            return {"status": "error", "errorCode": "TOPOLOGY_UNAVAILABLE"}

        topo = Topology.from_amp_list(rpdId, portId, amp_records)
        result = self.localizer.localize(rpdId, portId, ImpairmentLabel(impairmentType), pooled, topo)
        return result.model_dump(mode="json", exclude_none=True)
