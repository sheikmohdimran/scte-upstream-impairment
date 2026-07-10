# Upstream Impairment Localization - Intel POC

Agentic upstream-impairment localization for vCMTS/DOCSIS upstream networks. The system runs a
fixed 6-step MCP tool chain where the SLM orchestrates tool calls and error handling, while
classification and localization remain deterministic components.

Primary source docs:
- `docs/corrected-plan.md` (the reconciled design plan; moved out of this README)
- `CABLELABS-HANDOFF.md`
- `EXECUTION_STEPS.md`
- `docs/operations/open-decisions.md` (live decision/assumption tracker)
- `docs/project-memory.md` (working notes + current implementation state)

## What This Repository Implements

- A runnable POC for upstream impairment localization with one-shot amplifier measurement per leg.
- An MCP reference surface where the SLM sees handles and counts, not raw spectra arrays.
- A CPD-focused synthetic simulator, bootstrap rule classifier, and graph localizer.
- Two orchestration modes:
	- Deterministic orchestrator as grading oracle.
	- LangGraph SLM agent for tool-calling behavior.

## MCP Tool Chain (Fixed Sequence)

```
1 getRPDSpectrumMeasurements(rpdId, portId)          -> measurementRef
2 analyzeSpectrumMeasurements([measurementRef])      -> classificationSetRef (RPD)
3 getAllAmpsInSegment(rpdId, portId)                 -> ampListRef, ampCount, segmentId
4 getAmpSpectrumMeasurements(ampListRef)             -> measurementSetRef (or partial_success)
5 analyzeSpectrumMeasurements(measurementSetRef)     -> classificationSetRef (amps)
6 localizeUpstreamSpectrumImpairmentSource(...)      -> localized or low_confidence + next action
```

This flow is one-shot over the amplifier leg, not recursive drill-down. It is entered by an
alarm-trigger gate (see "Workflow Entry: Alarm Trigger").

## Data Surfaces: Handles vs Raw

- Reference surface (SLM-facing):
	- `measurementRef`, `measurementSetRef`, `ampListRef`, `classificationSetRef`, counts.
	- Includes explicit success, error, and `partial_success` shapes.
- Raw surface (backend-only):
	- `rawSpectrum` traces (`maxHold`, `minHold`, `average`) with 256 bins over 5-85 MHz.
	- Inline classifier artifacts and topology internals used by server/localizer.

Contract details are documented in `docs/architecture/contract-and-data-surfaces.md`.

## Assumptions And Risks

- Capture defaults are 5-85 MHz, 256 bins, dBmV traces (simulator/legacy path). Real amp
  captures are 5-184 MHz / 180 pts @1 MHz / dBuV / maxhold — spectrum + CNN ingestion is
  owned separately; coordinate on that interface.
- CPD signature is provisional (floor rise, comb spacing, and band extent).
- Topology is now parsed from the CableLabs plant data-package (`components[]` + `edges[]`)
  via `src/uil/localizer/plant_topology.py`; the localizer common point may be a passive
  device (splitter/tap/coupler). Binding assumptions C1–C3 tracked in open-decisions.
- `impairmentType` selection rule for step 6 currently uses dominant non-Clean signal.
- Rule-based classifier is scaffolding; production accuracy depends on CNN integration.

Full tracked assumptions and owner asks remain in `CABLELABS-HANDOFF.md`.

## Workflow Entry: Alarm Trigger

An incoming alarm is gated before any tool call (`src/uil/agent/trigger.py`):

- **Trigger (call)** iff `entity.type == "rpdPort"` **and** `alarmType` is an upstream
  FEC-error type (`highUpstreamFecErrors`, `highUpstreamCorrectables`,
  `highUpstreamUncorrectables`, `highUpstreamCorrectablesAndUncorrectables`). The workflow
  then starts at step 1 seeded with the alarm's `rpdId`/`portId`.
- **No trigger (noCall)** otherwise — e.g. upstream-FEC alarms on a *cable modem*, or
  non-FEC upstream alarms on an RPD port. The orchestrator short-circuits with status
  `not_triggered` and makes zero tool calls.

Entry points: `Orchestrator.run_from_alarm(alarm)` (deterministic oracle) and
`LangGraphAgent.run_from_alarm(alarm)` (SLM, skips model invocation on obvious negatives).
The gate is validated against the CableLabs `eval_CPD` alarm set (26 cases, all passing) in
`tests/test_trigger_eval.py`.

## Audit Trace Persistence (Regulatory)

Every run is durably recorded for audit (`src/uil/agent/trace_store.py`, `TraceStore`):

- **On by default** — both `Orchestrator` and `LangGraphAgent` write a record at every terminal
  outcome (`localized`, `low_confidence`, `escalated`, `failed`, `not_triggered`).
- **Append-only JSONL**, one record per run, with `runId`, `recordedUtc`, `scenario`,
  `finalStatus`, `toolCallCount`, the full tool-call `trace`, and the alarm `trigger` decision.
- **Tamper-evident** — records form a SHA-256 hash chain (`prevHash`/`recordHash`); any edit,
  deletion, or reordering is detectable via `TraceStore.verify()`.
- **Daily rotation** — `tool_call_traces-YYYY-MM-DD.jsonl` (UTC).
- **Handles only** — the audit record never contains raw spectra, only reference-surface
  handles, counts, and decisions.

Configuration:
- `UIL_TRACE_DIR` — output directory (default `./traces/`).
- `UIL_TRACE_DISABLE=1` — explicitly disable persistence (used by the test suite).

Tests: `tests/test_trace_store.py` (recording, append/chain, tamper detection, rotation).

## Open Decisions And Owners

- Irene/Bhaskar:
	- Final CPD spectral signature parameters.
	- CNN reuse/training details and 7-class to 9-label mapping.
	- Confidence calibration semantics for clean vs impaired thresholds.
- Randy (topology tool + trigger **now sampled/confirmed**; bindings still open):
	- C1: how `rpdId`/`portId` binds to the plant graph (assumed `RfSource`≈RPD, `portId`
	  picks an `RfPort` subtree).
	- C2: whole-plant vs pre-sliced per-segment subgraph at the tool boundary.
	- C3: amp identity — component `id` vs `name`.
	- C4: alarm transport (SNMP trap vs Kafka).
	- OpenShift/OCP validation timeline and scale expectations.
- Randy + Irene:
	- Error taxonomy completeness and `partial_success` contract confirmation.
	- Human handoff content requirements for operator actionability.

Decision tracker: `docs/operations/open-decisions.md`.

## Project Layout

| Path | Purpose |
|------|---------|
| `schemas/tools/` | Vendored MCP JSON schemas (reference and raw surfaces) |
| `src/uil/domain/` | Pydantic schema mirrors (incl. `PlantDeviceRef` for passive devices) |
| `src/uil/sim/` | Upstream spectrum simulator |
| `src/uil/classifier/` | Bootstrap rule classifier + CNN path |
| `src/uil/localizer/` | Graph-theory localizer + `plant_topology.py` data-package parser |
| `src/uil/mcp_server/` | Mock MCP server and handle store |
| `src/uil/agent/` | Orchestrator, LangGraph agent, alarm `trigger.py`, `trace_store.py` (audit), traces, handoff |
| `tests/` | Conformance, simulator, localizer, topology, trigger, E2E tests |
| `tests/fixtures/` | Real CableLabs samples: `alarms_eval.json`, `network-topology-example-*.json` |
| `docs/` | `corrected-plan.md`, `project-memory.md`, architecture + operations notes |

## Demo Quickstart

```bash
cd upstream-impairment

make install
make test
make demo

# Optional SLM orchestration demo (OpenAI-compatible endpoint required)
export SLM_BASE_URL=http://localhost:8000/v1
export SLM_MODEL=<served-model-name>
export OPENAI_API_KEY=EMPTY
make agent-demo

# Audit traces are written to ./traces/ by default; redirect or disable:
export UIL_TRACE_DIR=/var/log/uil/traces   # optional
# export UIL_TRACE_DISABLE=1                # optional: turn off persistence
```

Expected deterministic demo output (abridged):

```
Scenario 1: full localization -> localized
Scenario 2: partial_success + escalation -> low_confidence
```

## Current Status

- POC runs end-to-end on synthetic CPD data, entered via the alarm-trigger gate.
- **Test suite: 132 passing, 2 skipped.**
- Implemented since baseline:
	- Alarm-trigger gate (`src/uil/agent/trigger.py`) — perfect on the 26-case `eval_CPD` set;
	  wired into both `Orchestrator.run_from_alarm` and `LangGraphAgent.run_from_alarm`.
	- CableLabs plant data-package topology parser (`plant_topology.py`) generalized to the
	  `components[]` + `edges[]` format (validated on both real sample files + synthetic shapes).
	- Passive-aware localizer: common point can be a splitter/tap/coupler (`PlantDeviceRef`).
	- `getAllAmpsInSegment`/localize accept a real topology doc (`Scenario.topology_doc`).
	- Tamper-evident, daily-rotated audit trace persistence (`trace_store.py`), on by default.
- Next focus:
	- CNN integration behind `analyzeSpectrumMeasurements` (owned separately; 5-184 MHz / dBuV).
	- Update `schemas/tools/raw/getAllAmpsInSegment.schema.json` to the data-package shape.
	- Per-`portId` segment slicing; server e2e over a full real plant (needs label injection).
	- Confirm CableLabs bindings C1–C4 (see `docs/operations/open-decisions.md`).

Execution tracker: `EXECUTION_STEPS.md`.
