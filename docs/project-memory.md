# Project Memory — SCTE / vCMTS Upstream Impairment Localization

> Exported from the agent's working repository memory. Durable notes on the design contract,
> current implementation state, assumptions, and pending items. See `corrected-plan.md` for the
> full design plan and `operations/open-decisions.md` for the live decision tracker.

Location: `/home/sdp/wrkdir/scte`
Goal: Agentic (SLM-orchestrated) upstream impairment localization for cable plant. Paper due July 2026.

## Ground truth = MCP tool contract (OneDrive_2_6-17-2026.zip)

5 MCP tools, fixed call sequence:
1. `getRPDSpectrumMeasurements(rpdId, portId, numBins?=256, start=5MHz, stop=85MHz)` -> measurementRef
2. `analyzeSpectrumMeasurements(measurementRefs[] | measurementSetRef)` -> classificationSetRef, classificationCount, impairedCount, classesPresent[] (classify RPD)
3. `getAllAmpsInSegment(rpdId, portId)` -> segmentId, ampListRef, ampCount (+raw topology)
4. `getAmpSpectrumMeasurements(ampIds[] | ampListRef, ...)` -> measurementSetRef, measurementCount, failedCount, failedDevicesRef?; partial_success
5. `analyzeSpectrumMeasurements(measurementSetRef)` -> classificationSetRef (amps)
6. `localizeUpstreamSpectrumImpairmentSource(rpdId, portId, impairmentType, classificationSetRefs[]=[RPD set + amp set])` -> localizationStatus, likelySourceLocation, candidateLocations[], supporting/clean/uncertain Devices[], recommendedNextAction (can loop -> re-measure)

## Key design facts

- Reference surface: SLM never sees raw spectra; large data = handles+counts.
- Raw payload = rawSpectrum {startFrequencyHz, stopFrequencyHz, numBins, traces:{maxHold,minHold,average}[numBins]}.
- Impairment labels (9): Clean, CPD, Ingress, ImpulseNoise, WidebandNoise, NarrowbandInterference, Ripple, Suckout, UnknownImpairment. Primary anomaly = CPD = raised upstream noise floor.
- Device identity: RPD = rpdId+portId; AMP = ampId+portId.
- NOT recursive drill-down. Measure ALL amps in leg at once, classify all, localize via graph theory over topology (common-point analysis).
- Classifier = CNN multi-class over spectrum (reuse Bhaskar's prior full-band CNN, 7-class). LLM does NOT classify.
- LLM/SLM role = orchestrate tool sequence, handle errors/partial_success/missing labels, re-measure decisions, human handoff on multi-anomaly.
- Each tool output = oneOf by status incl error + partial_success; per-tool error code enums.

## POC scaffold and phases (2026-06)

- Python interp: `/home/sdp/vllm-env/bin/python` (pydantic 2.11, numpy, jsonschema, pytest).
- Run tests: `cd upstream-impairment && python -m pytest -q`. Demo: `examples/demo.py`.
- Structure: `src/uil/{domain,sim,classifier,localizer,mcp_server,agent}`; schemas under `schemas/tools/`.
- Phase 3 (LangGraph + pluggable OpenAI-compatible SLM) done; fine-tuning skipped. Deterministic
  Orchestrator stays as grading oracle. Llama-3.2-1B tool-calling weak without fine-tuning;
  recommend 7-8B via vLLM.
- Localizer flags conflicting/multi-anomaly labels -> low_confidence + escalate.

## 2026-07: new CableLabs samples (OneDrive_2_7-9-2026/) + build — 132 pass, 2 skip

Three sample groups: Network Topology Examples (2 json), Spectrum Capture Examples (2 json),
Alarms (`alarms.json`). Spectrum/CNN changes owned separately. Real capture contract for
coordination: 5-184 MHz, 180 pts @1 MHz, dBuV (`raw_byte*0.5`), maxhold-only, `rawValueHex`.

### Alarm trigger gate — DONE

- `src/uil/agent/trigger.py` `AlarmTrigger`. Rule: **call** iff `entity.type=="rpdPort"` AND
  `alarmType` in {highUpstreamFecErrors, highUpstreamCorrectables, highUpstreamUncorrectables,
  highUpstreamCorrectablesAndUncorrectables}. Emits `getRPDSpectrumMeasurements(rpdId,portId)`.
- Perfect on all 26 eval cases (`tests/test_trigger_eval.py`, fixture
  `tests/fixtures/alarms_eval.json`). Hard negatives: upstream-FEC on cableModem (neg-13/14),
  non-FEC upstream on rpdPort (neg-04/05/06).
- **Wired (2026-07-10):** `Orchestrator.run_from_alarm(alarm)` gates then runs (`run()` takes
  optional `rpd_id`/`port_id` to seed step 1); noCall -> status `not_triggered`, no tool calls,
  `trigger` field = TriggerDecision. `LangGraphAgent.run_from_alarm` mirrors it (skips SLM invoke
  on noCall). Tests in `tests/test_agent_e2e.py`.

### Audit trace persistence — DONE (regulatory)

- `src/uil/agent/trace_store.py` `TraceStore`: append-only JSONL, **on by default**, recorded by
  both orchestration paths at every terminal outcome (localized/low_confidence/escalated/failed/
  not_triggered).
- **Tamper-evident**: SHA-256 hash chain (`prevHash`/`recordHash`, genesis `0`*64) per file;
  `TraceStore.verify()` returns `(ok, bad_index)` and detects edits/reordering/deletion.
- **Daily rotation**: `tool_call_traces-YYYY-MM-DD.jsonl` (UTC); pass `filename=` to disable
  rotation. Records carry `runId`, `recordedUtc`, `finalStatus`, `toolCallCount`, trigger
  metadata, and the handles-only trace (no raw spectra).
- Config: `UIL_TRACE_DIR` (default `./traces/`), `UIL_TRACE_DISABLE=1` to disable. `conftest`
  disables it for the suite; `tests/test_trace_store.py` re-enables into `tmp_path`. `traces/`
  is gitignored.

### Topology parser — DONE

- `src/uil/localizer/plant_topology.py` `parse_data_package()` + `Topology.from_data_package()`.
- Real shape = Frictionless data-package `resources[0].data.{components[], edges[]}`. Root =
  `RfSource` (FN1); edges DOWNSTREAM (reversed for upstream). `RfAmp` = measured;
  `RfSplitter/RfTap/RfCoupler/RfPowerInserter` = passive common-point candidates; `RfPort/RfCable`
  collapsed (cable length -> distance); `device` (homes) = leaves dropped. file1 = 74 amps/382
  passive, file2 = 47 amps. 0 orphans.
- **GENERALIZED to the data-package FORMAT, not the 2 sample files.** Anti-overfit: tested on both
  real files + synthetic shapes (linear chain, splitter-two-amp star, unknown type tolerated,
  malformed doc raises). Unknown types/keys handled defensively; role sets are module constants.
  CableLabs-specific bindings (C1-C3) quarantined in `plant_topology.py`.
- Generic passive ref: `domain/refs.py` `PlantDeviceRef{deviceType,deviceId,name}`; `DeviceRef`
  union extended. `AmpNode` gained `deviceType`/`name`; `Topology.amp_ids` + `ref_for()`.
  Localizer common point can be passive -> `PlantDeviceRef`.
- Server wired: `Scenario.topology_doc` optional; `getAllAmpsInSegment`/localize use
  `from_data_package` when set (back-compat with amps list).

### Transcript confirmation

- Topology tool EXISTS (Randy: "we'll have the topology stuff"; Chai building it), delivered via
  `getAllAmpsInSegment` as a DB-record handle (not preloaded). So topology-present = DEFAULT path,
  degraded = exception.

### Open clarifications (docs/operations/open-decisions.md C1-C4)

- C1 `rpdId`/`portId` -> graph binding (assume RfSource≈RPD, portId picks RfPort subtree).
- C2 whole-plant vs pre-sliced subgraph at the tool boundary.
- C3 amp id = component `id` vs `name`.
- C4 alarm transport SNMP trap vs Kafka.

### Pending

- Per-`portId` segment slicing.
- Update `schemas/tools/raw/getAllAmpsInSegment.schema.json` to the data-package shape.
- Server e2e over the full real plant needs a per-amp impairment-label injection strategy.
- Coordinate the spectrum/CNN ingestion interface (5-184 MHz / dBuV / maxhold).
