# Upstream Impairment Localization - Intel POC

Agentic upstream-impairment localization for vCMTS/DOCSIS upstream networks. The system runs a
fixed 6-step MCP tool chain where the SLM orchestrates tool calls and error handling, while
classification and localization remain deterministic components.

Primary source docs merged into this README:
- `../SCTE-Plan-Corrected.md`
- `CABLELABS-HANDOFF.md`
- `EXECUTION_STEPS.md`

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

This flow is one-shot over the amplifier leg, not recursive drill-down.

## Data Surfaces: Handles vs Raw

- Reference surface (SLM-facing):
	- `measurementRef`, `measurementSetRef`, `ampListRef`, `classificationSetRef`, counts.
	- Includes explicit success, error, and `partial_success` shapes.
- Raw surface (backend-only):
	- `rawSpectrum` traces (`maxHold`, `minHold`, `average`) with 256 bins over 5-85 MHz.
	- Inline classifier artifacts and topology internals used by server/localizer.

Contract details are documented in `docs/architecture/contract-and-data-surfaces.md`.

## Assumptions And Risks

- Capture defaults are 5-85 MHz, 256 bins, dBmV traces.
- CPD signature is provisional (floor rise, comb spacing, and band extent).
- Topology shape from `getAllAmpsInSegment` is assumed to include parent/children links.
- `impairmentType` selection rule for step 6 currently uses dominant non-Clean signal.
- Rule-based classifier is scaffolding; production accuracy depends on CNN integration.

Full tracked assumptions and owner asks remain in `CABLELABS-HANDOFF.md`.

## Open Decisions And Owners

- Irene/Bhaskar:
	- Final CPD spectral signature parameters.
	- CNN reuse/training details and 7-class to 9-label mapping.
	- Confidence calibration semantics for clean vs impaired thresholds.
- Randy:
	- Real `getAllAmpsInSegment` payload shape and sample topology.
	- Trigger contract (SNMP trap or Kafka schema).
	- OpenShift/OCP validation timeline and scale expectations.
- Randy + Irene:
	- Error taxonomy completeness and `partial_success` contract confirmation.
	- Human handoff content requirements for operator actionability.

Decision tracker: `docs/operations/open-decisions.md`.

## Project Layout

| Path | Purpose |
|------|---------|
| `schemas/tools/` | Vendored MCP JSON schemas (reference and raw surfaces) |
| `src/uil/domain/` | Pydantic schema mirrors |
| `src/uil/sim/` | Upstream spectrum simulator |
| `src/uil/classifier/` | Bootstrap rule classifier |
| `src/uil/localizer/` | Graph-theory localizer |
| `src/uil/mcp_server/` | Mock MCP server and handle store |
| `src/uil/agent/` | Deterministic orchestrator, LangGraph agent, traces, handoff |
| `tests/` | Conformance, simulator, localizer, E2E tests |

## Demo Quickstart

```bash
cd /home/sdp/wrkdir/scte/upstream-impairment

make install
make test
make demo

# Optional SLM orchestration demo (OpenAI-compatible endpoint required)
export SLM_BASE_URL=http://localhost:8000/v1
export SLM_MODEL=<served-model-name>
export OPENAI_API_KEY=EMPTY
make agent-demo
```

Expected deterministic demo output (abridged):

```
Scenario 1: full localization -> localized
Scenario 2: partial_success + escalation -> low_confidence
```

## Current Status

- POC baseline is runnable end-to-end on synthetic CPD data.
- Test suite baseline: 29 tests passing.
- Next hardening focus:
	- CNN integration behind `analyzeSpectrumMeasurements`.
	- Topology-shape resilience in localizer.
	- Contract-tight error and `partial_success` handling.

Execution tracker: `EXECUTION_STEPS.md`.
