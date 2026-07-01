# SpectrumSampleGenerator — Implementation Plan
# Date: 2026-07-01

## Goal

Replace the 8-label generic stub in `SpectrumSimulator` with physics-based generation
(`SpectrumSampleGenerator`), integrate `CnnClassifier` on its native 8×200 linear format
via `use_cnn_path=True`, and add two new MCP tools:
- **T7** `getDeviceSpectrumSamples` — batch spectrum generation for RPD/amp devices
- **T8** `getSignalMetrics` — scalar RF metrics for a modem (CableLabs ticket)

---

## Files

### Create (new)
| File | Purpose |
|------|---------|
| `src/uil/domain/spectrum_sample.py` | `ImpairmentType`, `DeviceSpecification`, `SpectrumSample`, `GroupSpectrumResult` |
| `src/uil/sim/spectrum_sample_generator.py` | Physics-based generator, bridge table, `generate()`, `generate_group()` |
| `schemas/tools/reference/getDeviceSpectrumSamples.schema.json` | T7 reference schema |
| `schemas/tools/raw/getDeviceSpectrumSamples.schema.json` | T7 raw schema |
| `schemas/tools/reference/getSignalMetrics.schema.json` | T8 reference schema |
| `schemas/tools/raw/getSignalMetrics.schema.json` | T8 raw schema |
| `tests/test_spectrum_sample_generator.py` | 10 tests |
| `tests/test_signal_metrics.py` | 6 tests |
| `tests/test_cnn_path.py` | 5 E2E tests for `use_cnn_path=True` |

### Modify (existing)
| File | What changes |
|------|-------------|
| `schemas/tools/_defs.schema.json` | Add `generatorImpairmentType` + `spectrumSample` defs |
| `src/uil/classifier/cnn_classifier.py` | Add `_LOG_MAX_NATIVE` constant + `classify_snapshots()` method |
| `src/uil/mcp_server/server.py` | `Scenario` fields, `MockMcpServer.__init__`, T1/T4/T2/T5 branches, T7/T8 methods |

---

## Phase 1 — Domain models
**`src/uil/domain/spectrum_sample.py`**

```python
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field

class ImpairmentType(str, Enum):
    clean              = "clean"
    cpd                = "cpd"
    ingress_narrowband = "ingress_narrowband"
    ingress_burst      = "ingress_burst"
    impulse_noise      = "impulse_noise"
    micro_reflection   = "micro_reflection"
    amplitude_tilt     = "amplitude_tilt"

class DeviceSpecification(BaseModel):        # internal model — no prefix validator
    deviceId:    str
    deviceType:  Literal["RPD", "AMP"]       # set explicitly by T7/T8 method
    impairments: list[ImpairmentType] = [ImpairmentType.clean]
    severity:    Optional[float] = Field(default=None, ge=0.0, le=1.0)

class SpectrumSample(BaseModel):             # camelCase — follows RawSpectrum convention
    deviceId:         str
    deviceType:       Literal["RPD", "AMP"]
    impairments:      list[str]              # ImpairmentType.value strings (snake_case)
    severity:         float
    snapshots:        list[list[float]]      # 8 rows x 200 cols, linear power
    startFrequencyHz: int = 5_000_000
    stopFrequencyHz:  int = 85_000_000       # band edge, matches existing simulator
    numBins:          int = 200
    numSnapshots:     int = 8
    powerUnit:        Literal["linear"] = "linear"
    timestamp:        str

class GroupSpectrumResult(BaseModel):
    runSeverity:  float
    deviceCount:  int
    samples:      list[SpectrumSample]
    generatedAt:  str
```

---

## Phase 2 — Generator
**`src/uil/sim/spectrum_sample_generator.py`**

**Constraints:**
- All 7 generators inlined verbatim from
  `netsys_aiml/modelZoo/notebooks/docsis_pnm_us_impairment_cnn/data_generator.py`
  — no cross-repo import
- Constants inlined exactly:
  ```python
  T_SNAPS    = 8
  NUM_BINS   = 200
  F_START_HZ = 5_000_000
  BIN_HZ     = 400_000
  MAX_VAL    = 32767.0
  FLOOR_VAL  = 8.0
  ```
- Severity always explicit — remove `if severity is None: severity = rng.uniform(...)` branches
- `_rng_for()` uses `md5(key.encode()).digest()[:4]` → `np.random.default_rng(int.from_bytes(..., "big"))`
- `_synthesize()`: floor + additive deltas from each generator, clipped to `MAX_VAL`
- `generate_group()` raises `ValueError` if `len(specs) > max_devices`

**Bridge table (ImpairmentLabel → ImpairmentType):**
```python
from uil.domain.labels import ImpairmentLabel
from uil.domain.spectrum_sample import ImpairmentType

_LABEL_TO_IMPAIRMENT_TYPE: dict[ImpairmentLabel, ImpairmentType] = {
    ImpairmentLabel.Clean:                  ImpairmentType.clean,
    ImpairmentLabel.CPD:                    ImpairmentType.cpd,
    ImpairmentLabel.Ingress:                ImpairmentType.ingress_burst,
    ImpairmentLabel.ImpulseNoise:           ImpairmentType.impulse_noise,
    ImpairmentLabel.WidebandNoise:          ImpairmentType.ingress_burst,
    ImpairmentLabel.NarrowbandInterference: ImpairmentType.ingress_narrowband,
    ImpairmentLabel.Ripple:                 ImpairmentType.micro_reflection,
    ImpairmentLabel.Suckout:               ImpairmentType.amplitude_tilt,
    ImpairmentLabel.UnknownImpairment:      ImpairmentType.clean,
}
```

**Class skeleton:**
```python
class SpectrumSampleGenerator:
    def generate(self, spec: DeviceSpecification, run_severity: float = 1.0) -> SpectrumSample:
        eff_sev = spec.severity if spec.severity is not None else run_severity
        rng = self._rng_for(spec.deviceId, spec.impairments, eff_sev)
        matrix = self._synthesize(spec.impairments, eff_sev, rng)
        return SpectrumSample(
            deviceId=spec.deviceId, deviceType=spec.deviceType,
            impairments=[i.value for i in spec.impairments],
            severity=eff_sev, snapshots=matrix.tolist(), timestamp=_now())

    def generate_group(self, specs, run_severity=1.0, max_devices=50) -> GroupSpectrumResult:
        if len(specs) > max_devices:
            raise ValueError(f"Device count {len(specs)} exceeds limit {max_devices}")
        samples = [self.generate(s, run_severity) for s in specs]
        return GroupSpectrumResult(runSeverity=run_severity, deviceCount=len(samples),
                                   samples=samples, generatedAt=_now())

    @staticmethod
    def _rng_for(device_id, impairments, severity) -> np.random.Generator:
        key = f"{device_id}:{'|'.join(sorted(i.value for i in impairments))}:{severity:.4f}"
        seed = int.from_bytes(hashlib.md5(key.encode()).digest()[:4], "big")
        return np.random.default_rng(seed)

    def _synthesize(self, impairments, severity, rng) -> np.ndarray:
        floor = _gen_floor(rng)
        if impairments == [ImpairmentType.clean]:
            return floor
        combined = floor.copy()
        for imp in impairments:
            combined += _GENERATORS[imp](rng, severity) - floor   # additive delta only
        return np.clip(combined, 0.0, MAX_VAL).astype(np.float32)
```

---

## Phase 3 — CnnClassifier extension
**`src/uil/classifier/cnn_classifier.py`** — add two things only:

```python
# Add alongside existing _LOG_MAX:
_LOG_MAX_NATIVE: float = math.log10(32767.0)
# CRITICAL: matches data_generator.py LOG_MAX exactly.
# Different from _LOG_MAX = log10(2000.0) which is kept for classify_spectrum() compat.
```

```python
def classify_snapshots(
    self,
    snapshots: np.ndarray,      # shape (8, 200), linear power — native training format
    *,
    device_type: str,
    measurement_id: str,
    rpd_id: Optional[str] = None,
    port_id: Optional[str] = None,
    amp_id: Optional[str] = None,
) -> Classification:
    """Classify native 8x200 linear snapshots — no conversion needed."""
    self._load()
    # Skips steps 1-3 of _preprocess() (dBmV->linear, resample 256->200, replicate x8)
    log_mat = np.log10(np.maximum(snapshots, 1.0)) / _LOG_MAX_NATIVE
    z = ((log_mat - self._mean) / (self._std + 1e-8)).astype(np.float32)
    x = self._torch.from_numpy(z[np.newaxis, np.newaxis]).to(self._device)
    with self._torch.no_grad():
        proba = self._torch.sigmoid(self._model(x)).cpu().numpy()[0]
    # active-class detection, label mapping, Classification construction
    # — identical to classify_spectrum() from this point forward
    ...
```

---

## Phase 4 — Server
**`src/uil/mcp_server/server.py`**

### 4a. Extend Scenario (backward-compatible defaults)
```python
@dataclass
class Scenario:
    rpdId:       str
    portId:      str
    rpd_label:   ImpairmentLabel
    amps:        list[dict]
    faults:      FaultInjection = field(default_factory=FaultInjection)
    severity:    float = 1.0    # run-level; fixed for run lifetime
    max_devices: int   = 50     # parameterised cap for T7
```

### 4b. MockMcpServer.__init__
```python
def __init__(self, scenario, seed=42, classifier=None, use_cnn_path=False):
    self.scn          = scenario
    self.store        = HandleStore()
    self.sim          = SpectrumSimulator(seed=seed)     # always — legacy T1/T4
    self.sample_gen   = SpectrumSampleGenerator()        # always — T7/T8 + CNN path
    self.use_cnn_path = use_cnn_path
    if use_cnn_path:
        if classifier is not None and not isinstance(classifier, CnnClassifier):
            raise ValueError("use_cnn_path=True requires CnnClassifier")
        self.clf = CnnClassifier()
    else:
        self.clf = classifier if classifier is not None else RuleClassifier()
    self.localizer = GraphLocalizer()
    self._labels   = {a["ampId"]: ImpairmentLabel(a.get("label","Clean")) for a in scenario.amps}
    self._rpd_measurement_attempts = 0
```

### 4c. T1 getRPDSpectrumMeasurements — CNN branch
```python
if self.use_cnn_path:
    imp_type = _LABEL_TO_IMPAIRMENT_TYPE[self.scn.rpd_label]
    spec = DeviceSpecification(deviceId=f"rpd-{rpdId}", deviceType="RPD",
                               impairments=[imp_type])
    sample = self.sample_gen.generate(spec, run_severity=self.scn.severity)
    payload = {"deviceType":"RPD","rpdId":rpdId,"portId":portId,
               "timestamp":sample.timestamp, "snapshots":sample.snapshots}
else:
    spectrum = self.sim.generate(self.scn.rpd_label)
    payload  = {"deviceType":"RPD","rpdId":rpdId,"portId":portId,
                "timestamp":_now(), "spectrum":spectrum}
meas_id = self.store.put("meas", payload)
```

### 4d. T4 getAmpSpectrumMeasurements — CNN branch (per amp)
Same pattern using `_LABEL_TO_IMPAIRMENT_TYPE[self._labels[amp_id]]`.

### 4e. T2/T5 analyzeSpectrumMeasurements — format auto-detection
```python
if "snapshots" in item:
    c = self.clf.classify_snapshots(np.array(item["snapshots"]),
                                    device_type=item["deviceType"],
                                    measurement_id=item.get("_mid","?"), ...)
else:
    c = self.clf.classify_spectrum(item["spectrum"],
                                   device_type=item["deviceType"],
                                   measurement_id=item.get("_mid","?"), ...)
```

### 4f. T7 getDeviceSpectrumSamples
```python
def getDeviceSpectrumSamples(self, devices: list[dict]) -> dict:
    if len(devices) > self.scn.max_devices:
        return {"status":"error","errorCode":"TOO_MANY_DEVICES",
                "message":f"Requested {len(devices)}, limit {self.scn.max_devices}"}
    specs = []
    for d in devices:
        did = d.get("deviceId","")
        if   did.startswith("rpd-"): dtype = "RPD"
        elif did.startswith("amp-"): dtype = "AMP"
        else: return {"status":"error","errorCode":"INVALID_DEVICE_ID",
                      "message":f"deviceId '{did}' must start with 'rpd-' or 'amp-'"}
        try:
            impairments = [_LABEL_TO_IMPAIRMENT_TYPE[ImpairmentLabel(lbl)]
                           for lbl in d.get("impairments",["Clean"])]
        except ValueError as e:
            return {"status":"error","errorCode":"INVALID_IMPAIRMENT","message":str(e)}
        specs.append(DeviceSpecification(deviceId=did, deviceType=dtype,
                                         impairments=impairments,
                                         severity=d.get("severity")))
    result = self.sample_gen.generate_group(specs, run_severity=self.scn.severity,
                                            max_devices=self.scn.max_devices)
    ref = self.store.put("sampleset", result)
    return {"status":"success","sampleSetRef":ref,
            "deviceCount":result.deviceCount,"runSeverity":result.runSeverity}
```

### 4g. T8 getSignalMetrics
```python
def getSignalMetrics(self, modemId: str, windowSec: int = 60) -> dict:
    if   modemId.startswith("rpd-"): dtype = "RPD"
    elif modemId.startswith("amp-"): dtype = "AMP"
    else: return {"status":"error","errorCode":"INVALID_DEVICE_ID",
                  "message":f"modemId '{modemId}' must start with 'rpd-' or 'amp-'"}
    spec   = DeviceSpecification(deviceId=modemId, deviceType=dtype,
                                  impairments=[ImpairmentType.clean])
    sample = self.sample_gen.generate(spec, run_severity=self.scn.severity)
    ref    = self.store.put("sigmet", sample)
    key    = f"{modemId}:{windowSec}:{self.scn.severity:.4f}"
    rng    = np.random.default_rng(
        int.from_bytes(hashlib.md5(key.encode()).digest()[:4], "big"))
    tx   = float(38.0 + rng.uniform(0, 14))
    snr  = float(35.0 + rng.uniform(0, 10))
    uncr = float(min(rng.exponential(0.001), 1.0))
    return {"status":"success","rawArtifactRef":ref,
            "observationSummary":
                f"Modem {modemId}: TX={tx:.1f} dBmV, DS-SNR={snr:.1f} dB, "
                f"uncorr={uncr:.4f} ({windowSec}s, severity={self.scn.severity:.2f})",
            "upstreamTxPower":tx,"downstreamSnr":snr,"uncorrectablesRate":uncr}
```

---

## Phase 5 — Schemas

### _defs.schema.json additions
```json
"generatorImpairmentType": {
  "type": "string",
  "enum": ["clean","cpd","ingress_narrowband","ingress_burst",
           "impulse_noise","micro_reflection","amplitude_tilt"]
},
"spectrumSample": {
  "type": "object",
  "properties": {
    "deviceId":         { "type": "string" },
    "deviceType":       { "type": "string", "enum": ["RPD","AMP"] },
    "impairments":      { "type": "array", "minItems": 1,
                          "items": { "$ref": "#/$defs/generatorImpairmentType" } },
    "severity":         { "type": "number", "minimum": 0, "maximum": 1 },
    "snapshots":        { "type": "array", "minItems": 8, "maxItems": 8,
                          "items": { "type": "array", "minItems": 200, "maxItems": 200,
                                     "items": { "type": "number" } } },
    "startFrequencyHz": { "type": "integer" },
    "stopFrequencyHz":  { "type": "integer" },
    "numBins":          { "type": "integer", "const": 200 },
    "numSnapshots":     { "type": "integer", "const": 8 },
    "powerUnit":        { "type": "string", "const": "linear" },
    "timestamp":        { "type": "string", "format": "date-time" }
  },
  "required": ["deviceId","deviceType","impairments","severity","snapshots",
               "startFrequencyHz","stopFrequencyHz","numBins","numSnapshots",
               "powerUnit","timestamp"]
}
```

### schemas/tools/reference/getDeviceSpectrumSamples.schema.json
```json
{
  "name": "getDeviceSpectrumSamples",
  "description": "Generate synthetic upstream spectrum samples for RPD or smart amp devices.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "devices": {
        "type": "array", "minItems": 1,
        "items": {
          "type": "object",
          "properties": {
            "deviceId":    { "type": "string",
                             "description": "Must start with 'rpd-' or 'amp-'." },
            "impairments": { "type": "array", "minItems": 1,
                             "items": { "$ref": "../_defs.schema.json#/$defs/impairmentLabel" },
                             "default": ["Clean"] },
            "severity":    { "type": "number", "minimum": 0, "maximum": 1 }
          },
          "required": ["deviceId"],
          "additionalProperties": false
        }
      }
    },
    "required": ["devices"],
    "additionalProperties": false
  },
  "outputSchema": {
    "oneOf": [
      {
        "type": "object",
        "properties": {
          "status":      { "const": "success" },
          "sampleSetRef":{ "type": "string" },
          "deviceCount": { "type": "integer", "minimum": 0 },
          "runSeverity": { "type": "number", "minimum": 0, "maximum": 1 }
        },
        "required": ["status","sampleSetRef","deviceCount","runSeverity"]
      },
      {
        "allOf": [
          { "$ref": "../_defs.schema.json#/$defs/errorResponse" },
          { "properties": { "errorCode": {
              "enum": ["TOO_MANY_DEVICES","INVALID_DEVICE_ID",
                       "INVALID_IMPAIRMENT","PERMISSION_DENIED"] } } }
        ]
      }
    ]
  }
}
```

### schemas/tools/raw/getDeviceSpectrumSamples.schema.json
```json
{
  "name": "getDeviceSpectrumSamples",
  "description": "Raw output: inline spectrum samples per device (8x200 linear power).",
  "outputSchema": {
    "oneOf": [
      {
        "type": "object",
        "properties": {
          "status":      { "const": "success" },
          "runSeverity": { "type": "number" },
          "deviceCount": { "type": "integer" },
          "samples":     { "type": "array",
                           "items": { "$ref": "../_defs.schema.json#/$defs/spectrumSample" } }
        },
        "required": ["status","runSeverity","deviceCount","samples"]
      },
      {
        "allOf": [
          { "$ref": "../_defs.schema.json#/$defs/errorResponse" },
          { "properties": { "errorCode": {
              "enum": ["TOO_MANY_DEVICES","INVALID_DEVICE_ID",
                       "INVALID_IMPAIRMENT","PERMISSION_DENIED"] } } }
        ]
      }
    ]
  }
}
```

### schemas/tools/reference/getSignalMetrics.schema.json
```json
{
  "name": "getSignalMetrics",
  "description": "Derive scalar upstream RF metrics for a modem over an observation window.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "modemId":   { "type": "string",
                     "description": "Must start with 'rpd-' or 'amp-'." },
      "windowSec": { "type": "integer", "minimum": 1, "maximum": 86400,
                     "default": 60 }
    },
    "required": ["modemId"],
    "additionalProperties": false
  },
  "outputSchema": {
    "oneOf": [
      {
        "type": "object",
        "properties": {
          "status":             { "const": "success" },
          "rawArtifactRef":     { "type": "string" },
          "observationSummary": { "type": "string" },
          "upstreamTxPower":    { "type": "number" },
          "uncorrectablesRate": { "type": "number", "minimum": 0, "maximum": 1 },
          "downstreamSnr":      { "type": "number" }
        },
        "required": ["status","rawArtifactRef","observationSummary",
                     "upstreamTxPower","uncorrectablesRate","downstreamSnr"]
      },
      {
        "allOf": [
          { "$ref": "../_defs.schema.json#/$defs/errorResponse" },
          { "properties": { "errorCode": {
              "enum": ["DEVICE_UNREACHABLE","MODEM_OFFLINE","INVALID_DEVICE_ID",
                       "WINDOW_TOO_LARGE","PERMISSION_DENIED"] } } }
        ]
      }
    ]
  }
}
```

### schemas/tools/raw/getSignalMetrics.schema.json
```json
{
  "name": "getSignalMetrics",
  "description": "Raw output: full spectrum sample + scalar RF metrics.",
  "outputSchema": {
    "oneOf": [
      {
        "type": "object",
        "properties": {
          "status":             { "const": "success" },
          "modemId":            { "type": "string" },
          "windowSec":          { "type": "integer" },
          "upstreamTxPower":    { "type": "number" },
          "downstreamSnr":      { "type": "number" },
          "uncorrectablesRate": { "type": "number", "minimum": 0, "maximum": 1 },
          "observationSummary": { "type": "string" },
          "spectrum":           { "$ref": "../_defs.schema.json#/$defs/spectrumSample" }
        },
        "required": ["status","modemId","windowSec","upstreamTxPower","downstreamSnr",
                     "uncorrectablesRate","observationSummary","spectrum"]
      },
      {
        "allOf": [
          { "$ref": "../_defs.schema.json#/$defs/errorResponse" },
          { "properties": { "errorCode": {
              "enum": ["DEVICE_UNREACHABLE","MODEM_OFFLINE","INVALID_DEVICE_ID",
                       "WINDOW_TOO_LARGE","PERMISSION_DENIED"] } } }
        ]
      }
    ]
  }
}
```

---

## Phase 6 — Tests

### tests/test_spectrum_sample_generator.py (10 tests)
| Test | Assertion |
|------|-----------|
| `test_all_7_types_produce_8x200` | parametrize all `ImpairmentType` → shape (8, 200) |
| `test_deterministic_same_inputs` | two calls, same inputs → identical `snapshots` |
| `test_different_severity_differs` | `severity=0.2` vs `0.9` → different `snapshots` |
| `test_per_device_severity_overrides_run` | `spec.severity=0.1`, `run_severity=1.0` → uses `0.1` |
| `test_group_preserves_order_and_count` | 3 devices → `deviceCount==3`, `samples[1].deviceId==B` |
| `test_group_max_devices_enforced` | 51 specs, `max_devices=50` → `ValueError` |
| `test_snapshots_in_valid_range` | all values in `[0, 32767]` |
| `test_label_bridge_cpd` | `_LABEL_TO_IMPAIRMENT_TYPE[ImpairmentLabel.CPD] == ImpairmentType.cpd` |
| `test_label_bridge_all_9_labels` | no `KeyError` for any `ImpairmentLabel` |
| `test_schema_conformance_t7_reference` | `Draft202012Validator(t7_ref_schema).validate(server.getDeviceSpectrumSamples(...))` |

### tests/test_signal_metrics.py (6 tests)
| Test | Assertion |
|------|-----------|
| `test_deterministic` | same `modemId + windowSec + severity` → identical response |
| `test_severity_changes_output` | `severity=0.1` vs `0.9` → different `rawArtifactRef` contents |
| `test_reference_hides_snapshots` | response dict has no `"snapshots"` key |
| `test_raw_artifact_resolvable_8x200` | `store.get(ref).snapshots` → 8 rows, 200 cols |
| `test_field_ranges` | `upstreamTxPower ∈ [35,55]`, `downstreamSnr ∈ [30,50]`, `uncorrectablesRate ∈ [0,1]` |
| `test_schema_conformance_t8_reference` | `Draft202012Validator(t8_ref_schema).validate(server.getSignalMetrics(...))` |

### tests/test_cnn_path.py (5 tests)
| Test | Assertion |
|------|-----------|
| `test_cnn_path_t1_stores_snapshots_key` | `store.get(meas_id)` has `"snapshots"` not `"spectrum"` |
| `test_cnn_path_t2_routes_to_classify_snapshots` | `analyzeSpectrumMeasurements` returns `status=="success"` |
| `test_cnn_path_reference_surface_hides_snapshots` | T1 reference output has no `"snapshots"` key |
| `test_cnn_path_full_orchestrator` | `Orchestrator(cnn_server).run().status != "failed"` |
| `test_cnn_path_wrong_classifier_raises` | `MockMcpServer(scenario, use_cnn_path=True, classifier=RuleClassifier())` → `ValueError` |

---

## Verification Sequence

```bash
# Step 1 — after Phase 1+2
pytest tests/test_spectrum_sample_generator.py -v          # 10 pass

# Step 2 — after Phase 3
pytest tests/test_cnn_path.py::test_cnn_path_t2_routes_to_classify_snapshots -v   # 1 quick smoke

# Step 3 — after Phase 4
pytest tests/ -q --ignore=tests/test_langgraph_agent.py    # 32 original still pass
pytest tests/test_signal_metrics.py -v                     # 6 pass

# Step 4 — after Phase 5
pytest tests/test_schema_conformance.py -v                 # 4 new schemas auto-discovered + valid

# Step 5 — after Phase 6
pytest tests/test_cnn_path.py -v                           # 5 pass

# Step 6 — final
pytest tests/ -q --ignore=tests/test_langgraph_agent.py    # 53 pass (32 + 21 new)
python examples/demo.py                                     # "localized" — no regression
```

---

## End-to-End Flows

### Flow A — Legacy (use_cnn_path=False, default)
```
Scenario → MockMcpServer(use_cnn_path=False)
T1: SpectrumSimulator.generate() → RawSpectrum(256-bin dBmV) → store "spectrum" key
T2: "spectrum" key detected → RuleClassifier.classify_spectrum()
T3: topology → ampListRef
T4: SpectrumSimulator.generate() per amp → "spectrum" key
T5: "spectrum" key → RuleClassifier.classify_spectrum() per amp
T6: GraphLocalizer → localizationStatus="localized"
```

### Flow B — CNN path (use_cnn_path=True)
```
Scenario(severity=0.8) → MockMcpServer(use_cnn_path=True) → CnnClassifier forced
T1: _LABEL_TO_IMPAIRMENT_TYPE[label] → SpectrumSampleGenerator.generate()
    → ndarray(8,200) linear → store "snapshots" key
T2: "snapshots" key detected → CnnClassifier.classify_snapshots()
    → log10(max(x,1)) / log10(32767) → z-score → CNN → Classification
T3/T4/T5: same pattern per amp
T6: GraphLocalizer → localizationStatus
```

### Flow C — New tools (T7/T8, any use_cnn_path)
```
MockMcpServer.sample_gen always created

T7: getDeviceSpectrumSamples(devices=[{deviceId:"rpd-001", impairments:["CPD"]}])
    prefix check → deviceType="RPD"
    ImpairmentLabel("CPD") → _LABEL_TO_IMPAIRMENT_TYPE → ImpairmentType.cpd
    SpectrumSampleGenerator.generate_group(specs, run_severity=scn.severity)
    store.put("sampleset", GroupSpectrumResult)
    ← {status, sampleSetRef, deviceCount, runSeverity}  ← NO snapshots exposed

T8: getSignalMetrics(modemId="rpd-001", windowSec=60)
    SpectrumSampleGenerator.generate(clean_spec, run_severity)
    store.put("sigmet", SpectrumSample)
    md5 seed → scalar metrics
    ← {status, rawArtifactRef, observationSummary, upstreamTxPower, downstreamSnr, uncorrectablesRate}
```

---

## 11 Resolved Constraints (implementation checklist)

| # | Constraint |
|---|-----------|
| 1 | `_LABEL_TO_IMPAIRMENT_TYPE` covers all 9 `ImpairmentLabel` values; `Suckout→amplitude_tilt`, `UnknownImpairment→clean` |
| 2 | `rpd-*`/`amp-*` prefix validation only in T7/T8 methods — not in `DeviceSpecification` |
| 3 | `classify_snapshots()` skips dBmV→linear, resample 256→200, replicate ×8; starts at log10 |
| 4 | `_LOG_MAX_NATIVE = log10(32767.0)` matches `data_generator.py LOG_MAX`; separate from `_LOG_MAX = log10(2000.0)` |
| 5 | `use_cnn_path=True` + non-`CnnClassifier` → `ValueError` |
| 6 | T7/T8 input uses `impairmentLabel` CamelCase (existing enum); mapped to `ImpairmentType` snake_case internally |
| 7 | `self.sample_gen = SpectrumSampleGenerator()` always in `__init__`; not gated by `use_cnn_path` |
| 8 | `SpectrumSample` and `DeviceSpecification` use camelCase field names (follows `RawSpectrum` convention) |
| 9 | `stopFrequencyHz = 85_000_000` (band edge, not last bin center `84_600_000`) |
| 10 | `_defs.schema.json` gets `generatorImpairmentType` (7 snake_case) + `spectrumSample` defs |
| 11 | All schema `$ref` use `"../_defs.schema.json#/$defs/..."` relative path — matches existing `_defs_resolver()` |
