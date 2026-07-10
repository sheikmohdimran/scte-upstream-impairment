"""In-process mock MCP server: the 5 reference-surface tools (Steps 2 + 7).

Returns dicts shaped exactly like each tool's reference ``outputSchema``
(``oneOf(success | error[ | partial_success])``). Raw spectra live in the
:class:`HandleStore`; the SLM only ever receives handles + counts.

Fault injection (Step 7) lets tests force ``error`` / ``partial_success`` paths so the
orchestrator's recovery logic — the paper's real value-prop — can be exercised.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from uil.classifier.cnn_classifier import CnnClassifier
from uil.classifier.rule_classifier import RuleClassifier
from uil.domain.labels import ImpairmentLabel
from uil.domain.spectrum_sample import DeviceSpecification, ImpairmentType
from uil.localizer.graph_localizer import GraphLocalizer, Topology
from uil.mcp_server.handle_store import HandleStore
from uil.sim.spectrum_sample_generator import SpectrumSampleGenerator, _LABEL_TO_IMPAIRMENT_TYPE
from uil.sim.spectrum_simulator import SpectrumSimulator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FaultInjection:
    rpd_unreachable: bool = False
    rpd_stale_only: bool = False
    rpd_measurement_unavailable_once: bool = False
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
    severity: float = 1.0      # run-level severity; fixed for this run's lifetime
    max_devices: int = 50      # parameterised cap for getDeviceSpectrumSamples
    topology_doc: dict | None = None  # optional CableLabs RF plant data-package (T3 real shape)


class MockMcpServer:
    def __init__(self, scenario: Scenario, seed: int = 42, classifier=None,
                 use_cnn_path: bool = False) -> None:
        self.scn          = scenario
        self.store        = HandleStore()
        self.sim          = SpectrumSimulator(seed=seed)   # always — legacy T1/T4 path
        self.sample_gen   = SpectrumSampleGenerator()      # always — T7/T8 + CNN path
        self.use_cnn_path = use_cnn_path
        if use_cnn_path:
            if classifier is not None and not isinstance(classifier, CnnClassifier):
                raise ValueError(
                    "use_cnn_path=True requires CnnClassifier; "
                    f"got {type(classifier).__name__}"
                )
            self.clf = CnnClassifier()
        else:
            # Default to RuleClassifier so the synthetic-data tests remain stable.
            # Pass classifier=CnnClassifier() to use the v1 CNN on real HFC data.
            self.clf = classifier if classifier is not None else RuleClassifier()
        self.localizer = GraphLocalizer()
        self._labels   = {a["ampId"]: ImpairmentLabel(a.get("label", "Clean")) for a in scenario.amps}
        self._rpd_measurement_attempts = 0

    # ---- Tool 1 ---------------------------------------------------------
    def getRPDSpectrumMeasurements(self, rpdId: str, portId: str, numBins: int = 256,
                                   startFrequencyHz: int = 5_000_000, stopFrequencyHz: int = 85_000_000) -> dict:
        self._rpd_measurement_attempts += 1
        if self.scn.faults.rpd_unreachable:
            return {"status": "error", "errorCode": "DEVICE_UNREACHABLE", "message": f"{rpdId}/{portId} unreachable"}
        if self.scn.faults.rpd_stale_only:
            return {"status": "error", "errorCode": "STALE_DATA_ONLY"}
        if self.scn.faults.rpd_measurement_unavailable_once and self._rpd_measurement_attempts == 1:
            return {"status": "error", "errorCode": "MEASUREMENT_UNAVAILABLE"}
        if self.use_cnn_path:
            imp_type = _LABEL_TO_IMPAIRMENT_TYPE[self.scn.rpd_label]
            spec     = DeviceSpecification(deviceId=f"rpd-{rpdId}", deviceType="RPD",
                                           impairments=[imp_type])
            sample   = self.sample_gen.generate(spec, run_severity=self.scn.severity)
            payload  = {"deviceType": "RPD", "rpdId": rpdId, "portId": portId,
                        "timestamp": sample.timestamp, "snapshots": sample.snapshots}
        else:
            spectrum = self.sim.generate(self.scn.rpd_label)
            payload  = {"deviceType": "RPD", "rpdId": rpdId, "portId": portId,
                        "timestamp": _now(), "spectrum": spectrum}
        ts      = payload["timestamp"]
        meas_id = self.store.put("meas", payload)
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
            mid = item.get("_mid", item.get("measurementId", "?"))
            if "snapshots" in item:
                # CNN path: native 8x200 linear format — no conversion needed
                c = self.clf.classify_snapshots(
                    np.array(item["snapshots"], dtype=np.float32),
                    device_type=item["deviceType"],
                    measurement_id=mid,
                    rpd_id=item.get("rpdId"), port_id=item.get("portId"), amp_id=item.get("ampId"),
                )
            else:
                # Legacy path: RawSpectrum (256-bin dBmV)
                c = self.clf.classify_spectrum(
                    item["spectrum"],
                    device_type=item["deviceType"],
                    measurement_id=mid,
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
        if self.scn.topology_doc is not None:
            # Real CableLabs plant data-package: store the doc; localize parses it.
            topo = Topology.from_data_package(rpdId, portId, self.scn.topology_doc)
            amp_count = len(topo.amp_ids)
            if amp_count == 0:
                return {"status": "error", "errorCode": "EMPTY_SEGMENT"}
            amp_list_ref = self.store.put(
                "amplist", {"rpdId": rpdId, "portId": portId, "doc": self.scn.topology_doc}
            )
            return {
                "status": "success",
                "segmentId": f"seg-{rpdId}-{portId}",
                "ampListRef": amp_list_ref,
                "ampCount": amp_count,
                "topologyTimestamp": _now(),
            }
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
            if self.use_cnn_path:
                imp_type = _LABEL_TO_IMPAIRMENT_TYPE[self._labels[amp_id]]
                spec     = DeviceSpecification(deviceId=f"amp-{amp_id}", deviceType="AMP",
                                               impairments=[imp_type])
                sample   = self.sample_gen.generate(spec, run_severity=self.scn.severity)
                payload  = {"deviceType": "AMP", "ampId": amp_id, "portId": "0",
                            "timestamp": sample.timestamp, "snapshots": sample.snapshots}
            else:
                spectrum = self.sim.generate(self._labels[amp_id])
                payload  = {"deviceType": "AMP", "ampId": amp_id, "portId": "0",
                            "timestamp": _now(), "spectrum": spectrum}
            mid = self.store.put("meas", payload)
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
        topology_doc = None
        for handle in self.store._data:  # find the topology we built in tool 3
            if handle.startswith("amplist"):
                stored = self.store.get(handle)
                if "doc" in stored:
                    topology_doc = stored["doc"]
                else:
                    amp_records = stored["amps"]
                break
        if topology_doc is not None:
            topo = Topology.from_data_package(rpdId, portId, topology_doc)
        elif amp_records is not None:
            topo = Topology.from_amp_list(rpdId, portId, amp_records)
        else:
            return {"status": "error", "errorCode": "TOPOLOGY_UNAVAILABLE"}

        result = self.localizer.localize(rpdId, portId, ImpairmentLabel(impairmentType), pooled, topo)
        return result.model_dump(mode="json", exclude_none=True)

    # ---- Tool 7 ---------------------------------------------------------
    def getDeviceSpectrumSamples(self, devices: list[dict]) -> dict:
        """Generate spectrum samples for RPD/amp devices (T7 — new tool).

        Input device IDs must start with 'rpd-' or 'amp-'.
        Impairment labels use existing ImpairmentLabel CamelCase values.
        Run-level severity comes from Scenario.severity.
        """
        if len(devices) > self.scn.max_devices:
            return {"status": "error", "errorCode": "TOO_MANY_DEVICES",
                    "message": f"Requested {len(devices)}, limit is {self.scn.max_devices}"}
        specs = []
        for d in devices:
            did = d.get("deviceId", "")
            if did.startswith("rpd-"):
                dtype = "RPD"
            elif did.startswith("amp-"):
                dtype = "AMP"
            else:
                return {"status": "error", "errorCode": "INVALID_DEVICE_ID",
                        "message": f"deviceId '{did}' must start with 'rpd-' or 'amp-'"}
            try:
                impairments = [
                    _LABEL_TO_IMPAIRMENT_TYPE[ImpairmentLabel(lbl)]
                    for lbl in d.get("impairments", ["Clean"])
                ]
            except ValueError as exc:
                return {"status": "error", "errorCode": "INVALID_IMPAIRMENT",
                        "message": str(exc)}
            specs.append(DeviceSpecification(
                deviceId=did, deviceType=dtype,
                impairments=impairments,
                severity=d.get("severity"),
            ))
        result = self.sample_gen.generate_group(
            specs, run_severity=self.scn.severity, max_devices=self.scn.max_devices
        )
        ref = self.store.put("sampleset", result)
        return {
            "status":      "success",
            "sampleSetRef": ref,
            "deviceCount": result.deviceCount,
            "runSeverity": result.runSeverity,
        }

    # ---- Tool 8 ---------------------------------------------------------
    def getSignalMetrics(self, modemId: str, windowSec: int = 60) -> dict:
        """Derive scalar upstream RF metrics for a modem (T8 — new tool).

        modemId must start with 'rpd-' or 'amp-'. Generates a clean
        spectrum baseline deterministically; scalars are also deterministic.
        """
        if modemId.startswith("rpd-"):
            dtype = "RPD"
        elif modemId.startswith("amp-"):
            dtype = "AMP"
        else:
            return {"status": "error", "errorCode": "INVALID_DEVICE_ID",
                    "message": f"modemId '{modemId}' must start with 'rpd-' or 'amp-'"}
        spec   = DeviceSpecification(
            deviceId=modemId, deviceType=dtype, impairments=[ImpairmentType.clean]
        )
        sample = self.sample_gen.generate(spec, run_severity=self.scn.severity)
        ref    = self.store.put("sigmet", sample)
        # Deterministic scalar telemetry — same inputs always give same values
        key = f"{modemId}:{windowSec}:{self.scn.severity:.4f}"
        rng = np.random.default_rng(
            int.from_bytes(hashlib.md5(key.encode()).digest()[:4], "big")
        )
        tx   = float(38.0 + rng.uniform(0, 14))
        snr  = float(35.0 + rng.uniform(0, 10))
        uncr = float(min(rng.exponential(0.001), 1.0))
        return {
            "status":             "success",
            "rawArtifactRef":     ref,
            "observationSummary": (
                f"Modem {modemId}: TX={tx:.1f} dBmV, DS-SNR={snr:.1f} dB, "
                f"uncorr={uncr:.4f} ({windowSec}s, severity={self.scn.severity:.2f})"
            ),
            "upstreamTxPower":    tx,
            "downstreamSnr":      snr,
            "uncorrectablesRate": uncr,
        }
