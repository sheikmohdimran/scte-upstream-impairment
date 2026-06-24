# Upstream Impairment Localization — Intel POC

Agentic (SLM-orchestrated) localization of **upstream** spectrum impairments in a vCMTS/DOCSIS
cable plant. An SLM drives a fixed chain of **5 MCP tools** to capture spectrum at an RPD port,
enumerate and measure every amplifier in the leg **in one shot**, classify each capture, and then
**localize** the impairment source via graph-theory common-point analysis.

See `../SCTE-Plan-Corrected.md` for the full plan and `EXECUTION_STEPS.md` for the build tracker.
The tool contract is vendored under `schemas/tools/` (source of truth).

## Architecture (one-shot, not recursive)

```
1 getRPDSpectrumMeasurements  -> measurementRef
2 analyzeSpectrumMeasurements -> classificationSetRef · RPD   (kept for step 6)
3 getAllAmpsInSegment         -> ampListRef, ampCount
4 getAmpSpectrumMeasurements  -> measurementSetRef  (may be partial_success)
5 analyzeSpectrumMeasurements -> classificationSetRef · amps
6 localizeUpstream…           -> span/device/branch + confidence + recommendedNextAction
```

The SLM only ever holds **handles + counts**; raw spectra (`maxHold/minHold/average` per bin,
5–85 MHz, 256 bins) stay behind the backend handle store. Classification is a CNN in production
(here a transparent rule classifier bootstrap); the SLM never classifies.

## Layout

| Path | Step | What |
|------|------|------|
| `schemas/tools/` | 0 | Vendored JSON schemas (reference + raw) |
| `src/uil/domain/` | 0 | Pydantic mirrors of the schemas |
| `src/uil/sim/` | 3 | Upstream spectrum simulator (CPD-focused) |
| `src/uil/classifier/` | 4 | Bootstrap rule classifier |
| `src/uil/localizer/` | 6 | Graph-theory common-point localizer |
| `src/uil/mcp_server/` | 2,7 | Mock MCP server (5 tools) + handle store + fault injection |
| `src/uil/agent/` | 5,7,8 | Orchestrator + human-handoff + tool-call traces |
| `tests/` | all | Conformance, simulator, localizer, E2E |

## Run

```bash
make install        # installs into /home/sdp/vllm-env
make test           # pytest
make demo           # end-to-end CPD localization with an injected partial_success
```

## Status
First pass lands a runnable skeleton of **all 8 build steps** end-to-end on synthetic data.
Next: real CNN integration, LangGraph + local-SLM orchestration, richer fault models.
Provisional CPD parameters (`config/cpd_signatures.yaml`) await Irene/Bhaskar confirmation.
