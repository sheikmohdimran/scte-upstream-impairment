"""Demo: CNN path + T7/T8 new tools.

Shows all three new flows introduced alongside the physics-based
SpectrumSampleGenerator and CnnClassifier integration:

  Flow B — CNN path (use_cnn_path=True):  6-step localization with native
            8×200 linear spectra and the v1 CNN classifier.
  Flow C — T7 getDeviceSpectrumSamples:   batch spectrum generation for
            RPD and smart-amp devices.
  Flow D — T8 getSignalMetrics:           scalar RF metrics for a modem.

Run from the repo root:
  PYTHONPATH=src python examples/demo_new_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from uil.agent.orchestrator import Orchestrator
from uil.domain.labels import ImpairmentLabel
from uil.mcp_server.server import MockMcpServer, Scenario


# ── helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def _cnn_scenario(label: ImpairmentLabel, severity: float = 0.8) -> Scenario:
    return Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=label,
        amps=[
            {"ampId": "A1", "parentId": None,  "label": "Clean"},
            {"ampId": "A2", "parentId": "A1",  "label": label.value},
            {"ampId": "A3", "parentId": "A2",  "label": label.value},
            {"ampId": "A4", "parentId": "A1",  "label": "Clean"},
        ],
        severity=severity,
    )


# ── Flow B: CNN path ──────────────────────────────────────────────────────────

_section("Flow B — CNN path  (use_cnn_path=True, SpectrumSampleGenerator + CnnClassifier)")

for label, name in [
    (ImpairmentLabel.CPD,                  "CPD"),
    (ImpairmentLabel.Ripple,               "Ripple / MicroReflection"),
    (ImpairmentLabel.NarrowbandInterference,"NarrowbandInterference"),
    (ImpairmentLabel.ImpulseNoise,         "ImpulseNoise"),
]:
    server = MockMcpServer(_cnn_scenario(label, severity=0.8), use_cnn_path=True)
    result = Orchestrator(server, scenario_name=f"cnn-{name}").run()

    print(f"\n  Injected  : {name}")
    print(f"  Data      : 8×200 linear (SpectrumSampleGenerator)")
    print(f"  Classifier: CnnClassifier (us_impairment_cnn_v1.pt, F1=0.9322)")
    for c in result.trace.calls:
        mark = "✓" if c.outcome == "success" else "~"
        print(f"    {mark} step {c.step}: {c.tool:<44} [{c.outcome}]")
    if result.localization:
        loc = result.localization
        print(f"  Detected  : {loc.get('impairmentType')}")
        print(f"  Status    : {loc.get('localizationStatus')}  "
              f"(confidence={loc.get('confidence')})")
        if loc.get("likelySourceLocation"):
            print(f"  Source    : {loc['likelySourceLocation']['description']}")
    else:
        print(f"  Status    : {result.status}")


# ── Flow C: T7 getDeviceSpectrumSamples ───────────────────────────────────────

_section("Flow C — T7 getDeviceSpectrumSamples")

scenario = Scenario(
    rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
    amps=[{"ampId": "A1", "parentId": None, "label": "Clean"}],
    severity=0.75,   # run-level severity — applies to devices with no per-device override
)
server = MockMcpServer(scenario)

devices = [
    {"deviceId": "rpd-001", "impairments": ["CPD"],                   "severity": 0.8},
    {"deviceId": "rpd-002", "impairments": ["Ripple"]},                          # uses run severity 0.75
    {"deviceId": "rpd-003", "impairments": ["CPD", "Ripple"],         "severity": 0.6},  # multi-label
    {"deviceId": "amp-001", "impairments": ["NarrowbandInterference"], "severity": 0.9},
    {"deviceId": "amp-002", "impairments": ["Clean"]},
    {"deviceId": "amp-003", "impairments": ["ImpulseNoise"],           "severity": 0.5},
]

ref_out = server.getDeviceSpectrumSamples(devices=devices)
print(f"\n  Reference surface (what SLM/agent sees):")
print(f"    status       = {ref_out['status']}")
print(f"    sampleSetRef = {ref_out['sampleSetRef']}   ← opaque handle, no spectra")
print(f"    deviceCount  = {ref_out['deviceCount']}")
print(f"    runSeverity  = {ref_out['runSeverity']}")

print(f"\n  Raw surface (behind the handle — 8×200 linear spectra):")
result = server.store.get(ref_out["sampleSetRef"])
for s in result.samples:
    mat = np.array(s.snapshots)
    print(f"    {s.deviceId:<14}  {s.deviceType}  imp={s.impairments}  "
          f"sev={s.severity:.2f}  shape={mat.shape}  "
          f"floor≈{mat.mean():.1f}  peak={mat.max():.0f}")


# ── Flow D: T8 getSignalMetrics ────────────────────────────────────────────────

_section("Flow D — T8 getSignalMetrics")

scenario = Scenario(
    rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.Clean,
    amps=[{"ampId": "A1", "parentId": None, "label": "Clean"}],
    severity=1.0,
)
server = MockMcpServer(scenario)

for modem_id, window in [
    ("rpd-north-001",   60),
    ("rpd-south-042",  300),
    ("amp-site-a-lvl2", 60),
]:
    out    = server.getSignalMetrics(modemId=modem_id, windowSec=window)
    sample = server.store.get(out["rawArtifactRef"])
    print(f"\n  {modem_id}  window={window}s")
    print(f"    {out['observationSummary']}")
    print(f"    rawArtifactRef = {out['rawArtifactRef']}  "
          f"← 8×200 spectrum (powerUnit={sample.powerUnit})")

print()
print("  ✓  Severity effect on scalars:")
for sev in [0.1, 0.5, 1.0]:
    sc = Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.Clean,
        amps=[{"ampId": "A1", "parentId": None, "label": "Clean"}],
        severity=sev,
    )
    r = MockMcpServer(sc).getSignalMetrics(modemId="rpd-001", windowSec=60)
    print(f"    severity={sev:.1f}  TX={r['upstreamTxPower']:.1f} dBmV  "
          f"SNR={r['downstreamSnr']:.1f} dB  uncorr={r['uncorrectablesRate']:.5f}")

print()
