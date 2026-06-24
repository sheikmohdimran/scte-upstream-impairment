# CableLabs Handoff — Assumptions, Clarifications & Demo Guide

> **Purpose**: everything Imran (Intel) needs to (a) talk through the POC implementation, (b) run
> the demo live, and (c) procure the open decisions from CableLabs (Randy / Irene) and Intel
> (Bhaskar / Carlos). Architecture source of truth: [`../SCTE-Plan-Corrected.md`](../SCTE-Plan-Corrected.md);
> tool contract vendored under [`schemas/tools/`](schemas/tools/); step tracker in
> [`EXECUTION_STEPS.md`](EXECUTION_STEPS.md).

---

## 0. One-paragraph framing (say this first)

We built a runnable POC of the **agentic upstream-impairment localization** pipeline. An SLM
orchestrates a fixed **6-step MCP tool chain**; it only ever sees **handles + counts**, never raw
spectra. Classification is a (currently mocked) **CNN multi-class** behind
`analyzeSpectrumMeasurements`; localization is **graph-theory common-point** analysis — not LLM
reasoning. Everything runs offline on a **synthetic CPD simulator** behind the same MCP reference
surface, so the simulator can later be swapped for live RPD/CMTS capture **without changing
anything above it**. The whole thing is green: **29 automated tests pass**.

---

## PART A — Assumptions baked into the POC

Each row: what we assumed, where it lives in code/config, and the **risk if CableLabs tells us
otherwise**. These are the things to validate so the paper's numbers are defensible.

### A1. Capture / spectrum format
| # | Assumption | Where | Risk if wrong |
|---|------------|-------|---------------|
| 1 | Upstream band **5–85 MHz**, **256 bins**, per-bin `maxHold`/`minHold`/`average` | [config/capture_defaults.yaml](config/capture_defaults.yaml); `RawSpectrum` in [src/uil/domain/spectrum.py](src/uil/domain/spectrum.py) | Bin count/band drives CNN input shape; mismatch forces retrain + simulator change. |
| 2 | Power units are **dBmV** per bin | simulator + classifier thresholds | Threshold/feature scaling shifts; classifier recalibration. |
| 3 | One capture per device, captured **on demand** (expensive), not streamed | orchestrator flow | Affects trigger design & how often we re-measure. |

### A2. CPD signature (the primary anomaly) — **all provisional**
| # | Assumption (provisional value) | Where | Risk if wrong |
|---|--------------------------------|-------|---------------|
| 4 | CPD = **raised noise floor**, lift ≈ **+8 dB** | [config/cpd_signatures.yaml](config/cpd_signatures.yaml) `cpd.floor_rise_db` | Detection threshold; false-positive/negative rate. |
| 5 | CPD concentrated in **5–42 MHz** low upstream | same, `band_start/stop_hz` | Band-limited features wrong → mislabels. |
| 6 | Discrete **comb products at ~6 MHz** spacing, +6 dB teeth | same, `comb_spacing_hz`, `comb_peak_db` | The "comb" is a key CPD discriminator; wrong spacing weakens the CNN. |
| 7 | Clean floor ≈ **-40 dBmV**, ±1 dB variance | same, `clean.*` | Defines the impaired/clean boundary used everywhere. |
| 8 | Other 8 labels (Ingress, NarrowbandInterference, …) are **stubs** | same | Only CPD is realistic today; multi-label claims need real signatures. |

### A3. Classifier (CNN) — mocked
| # | Assumption | Where | Risk if wrong |
|---|------------|-------|---------------|
| 9 | A **rule-based bootstrap** classifier stands in for the CNN; emits `class` + `confidence` | [src/uil/classifier/rule_classifier.py](src/uil/classifier/rule_classifier.py) | Real accuracy unknown until Bhaskar's CNN is integrated. |
| 10 | Output conforms to `_defs.schema.json#/$defs/classification` (`class`, `confidence`, `observations[]`) | [src/uil/domain/classification.py](src/uil/domain/classification.py) | If the real CNN emits a different shape, the reference contract changes. |
| 11 | Label set is the **9 labels** in `_defs.schema.json#/$defs/impairmentLabel` | [src/uil/domain/labels.py](src/uil/domain/labels.py) | Bhaskar's prior model is ~7-class; mapping to these 9 is unconfirmed. |

### A4. Topology & localizer
| # | Assumption | Where | Risk if wrong |
|---|------------|-------|---------------|
| 12 | `getAllAmpsInSegment` returns `amps[]` with **`parentId` / `children` / `distanceFromRpdMeters`** (a tree) | [src/uil/mcp_server/server.py](src/uil/mcp_server/server.py); `Topology.from_amp_list` in [src/uil/localizer/graph_localizer.py](src/uil/localizer/graph_localizer.py) | If real data is a **flat / distance-ordered list** with no parent links, the graph builder must change. **Highest-impact unknown.** |
| 13 | Upstream flows **leaf → RPD**; a fault is seen by every device on the path to the RPD | localizer docstring/logic | Inverts the common-point math if the model differs. |
| 14 | Single propagating fault per segment; **>1 distinct impaired label ⇒ escalate** | `GraphLocalizer._conflicting` | Real multi-fault segments may need a richer model than "escalate". |
| 15 | `distanceFromRpdMeters` is **captured but unused** today | localizer | Available to tighten span localization once real distances exist. |

### A5. Agent / SLM
| # | Assumption | Where | Risk if wrong |
|---|------------|-------|---------------|
| 16 | `impairmentType` for step 6 = **first/dominant non-Clean** label from the RPD classification set | orchestrator + agent system prompt | If selection should come from amps or a policy, behavior changes. |
| 17 | The SLM is responsible for **off-happy-path** decisions only (retry/partial/escalate), not classification | [src/uil/agent/langgraph_agent.py](src/uil/agent/langgraph_agent.py) `SYSTEM_PROMPT` | Matches the design intent per the transcript. |
| 18 | A **7–8B instruct model** (or a guarded harness) is needed for reliable tool-calling; **fine-tuning skipped** for now | agent module | `Llama-3.2-1B` alone is unreliable; demo uses a scripted model offline. |
| 19 | Transient errors (`STALE_DATA_ONLY`, `MEASUREMENT_UNAVAILABLE`) retry **once**; others escalate | orchestrator + prompt | Retry policy is a guess; confirm operational tolerances. |

### A6. Deployment
| # | Assumption | Where | Risk if wrong |
|---|------------|-------|---------------|
| 20 | POC runs **locally at Intel** (Phases 1–3), later deploys to CableLabs **OCP/OpenShift** for validation | plan §6 Phase 4 | OpenShift access was blocked at last check — confirm timeline. |
| 21 | MCP reference surface is the **only** seam between mock and real | mock server | If real systems can't honor the handle contract, redesign needed. |

---

## PART B — Clarifications to procure from CableLabs / Intel

Grouped by owner, with **what each unblocks**. Bring this list to the next sync.

### B1. From Irene / Bhaskar (signal + model)
1. **CPD spectral signature** — real floor-rise margin (dB), comb spacing & amplitude, band extent.
   → unblocks A2 (simulator realism) and the CNN training set.
2. **CNN reuse details** — input shape & normalization of Bhaskar's full-band CNN vs our 256-bin
   upstream trace; retrain plan; **7-class → 9-label** mapping. → unblocks A3.
3. **Sample real captures** (even a handful of clean + CPD traces) to calibrate the simulator and
   sanity-check the bootstrap classifier. → unblocks A1/A2.
4. **Confidence semantics** — is the CNN `confidence` calibrated, and what threshold separates
   `impaired` vs `clean`? → unblocks A3/A4 boundary logic.

### B2. From Randy (topology + triggers + platform)
5. **`getAllAmpsInSegment` payload** — does the **raw** output carry `parentId`/`children`/
   `distanceFromRpdMeters`, or a flat/distance list? Provide one real example. → unblocks A4 (#12),
   the single highest-impact item.
6. **Trigger contract** — exact **SNMP-trap** payload (and/or **Kafka** telemetry schema) that
   kicks off step 1, and the rule that decides "multi-channel upstream issue." → unblocks Phase 4.
7. **`impairmentType` selection rule** for the localize call (step 6). → unblocks A5 (#16).
8. **OpenShift / OCP access & timeline** for the validation deployment. → unblocks A6 (#20).
9. **Scale expectation** — telemetry cadence and the ~75k-modems/CPU concern; how many concurrent
   localizations must the agent sustain? → informs architecture claims in the paper.

### B3. Contract / scope confirmations (Randy + Irene)
10. **Error taxonomy completeness** — are the per-tool error codes we modeled the full set, and is
    `partial_success` shaped as we assumed (`failedCount` + `failedDevicesRef`)? → confirms the
    reference contract we built to.
11. **Human-handoff expectation** — what must the escalation "Diagnosis Summary" contain for an
    operator to act? → tightens [src/uil/agent/handoff.py](src/uil/agent/handoff.py).
12. **SLM choice & fine-tuning** — is Intel expected to deliver a fine-tuned small model, or is a
    larger off-the-shelf instruct model acceptable for the paper? → unblocks A5 (#18).
13. **Emulated amp data** — until smart-amp hardware exists, is emulating amp captures (resembling
    RPD upstream) acceptable for the July deliverable? → confirms Phase 4 plan.

---

## PART C — Demo walkthrough (talk track)

**Architecture in one diagram** (the fixed sequence the agent runs):

```
1 getRPDSpectrumMeasurements ──▶ 2 analyzeSpectrumMeasurements (RPD)
                                          │ classificationSetRef (kept)
3 getAllAmpsInSegment ──▶ 4 getAmpSpectrumMeasurements (may be partial_success)
                                          │ measurementSetRef
                                 5 analyzeSpectrumMeasurements (amps)
                                          │ classificationSetRef
6 localizeUpstreamSpectrumImpairmentSource ──▶ span / device / branch + confidence
        └─ low_confidence / conflicting ──▶ human handoff (Diagnosis Summary)
```

**Two things to emphasize while presenting:**
1. **The SLM never touches raw spectra** — only handles + counts cross the reference surface. Open
   any tool result in the trace: you'll see `measurementId`, `classificationSetRef`, counts — no
   arrays. This is the central design idea that keeps SLM context small and safe.
2. **The value is the off-happy-path behavior** — the demo injects a `partial_success` (one amp
   leg unreachable). The agent proceeds with what it has, the localizer marks the missing leg
   `uncertain`, confidence drops, and it **escalates with a human-readable summary**. A plain
   script does the happy path; the agent earns its keep on the error path.

**Suggested live sequence (≈3 min):**
1. `make test` → "29 green: schema conformance, simulator, classifier, localizer, agent E2E."
2. `make demo` → show **Scenario 1** (full CPD localization to the A1→A2 branch, confidence 0.8)
   and **Scenario 2** (injected failure → low_confidence → handoff markdown).
3. Open the printed **TRACE** → point at handles-only payloads and the per-step decisions.
4. (If an SLM endpoint is up) `make agent-demo` → same chain, but a **real LLM** chooses each tool
   call via LangGraph. Otherwise explain: the deterministic orchestrator is the **grading oracle**
   the SLM is measured against, and an offline scripted-model test proves the LangGraph loop.

**Talking points for likely questions:**
- *"Is localization the LLM?"* No — deterministic graph common-point analysis
  ([graph_localizer.py](src/uil/localizer/graph_localizer.py)). The LLM only orchestrates.
- *"Is the classifier real?"* It's a transparent **bootstrap rule** standing in for Bhaskar's CNN;
  it's a drop-in behind `analyzeSpectrumMeasurements`.
- *"How do we go live?"* Swap `MockMcpServer` for a real adapter implementing the same 5 tools.
  Nothing above the reference surface changes (Phase 4).

---

## PART D — How to run it

**Environment:** Python 3.12 venv at `/home/sdp/vllm-env/bin/python` (pydantic 2, numpy,
jsonschema, pytest, and the agent extras already installed).

```bash
cd /home/sdp/wrkdir/scte/upstream-impairment

# 1. Tests (fast, fully offline) — expect "29 passed"
make test

# 2. Deterministic demo — two scenarios (full localize + partial_success escalation)
make demo

# 3. (Optional) Real LangGraph + SLM agent — needs an OpenAI-compatible endpoint.
#    Recommended: a 7-8B instruct model served by vLLM (1B tool-calling is unreliable).
export SLM_BASE_URL=http://localhost:8000/v1
export SLM_MODEL=<served-model-name>
export OPENAI_API_KEY=EMPTY
make agent-demo
```

Install on a fresh machine: `make install` (core + dev) or `make install-agent` (adds
langgraph/langchain). The `Makefile` pins the interpreter to the venv.

**Expected `make demo` output (abridged):**
```
Scenario 1: full localization → localized
  impairment=CPD status=localized confidence=0.8
  source: Branch at/under amp A2 (common point of impaired amps ['A2','A3'])
Scenario 2: partial_success + escalation → low_confidence
  [human handoff artifact generated]
```

---

## PART E — Mock vs real (the drop-in seams)

| Component | Today (mock) | Drop-in for production | Seam |
|-----------|--------------|------------------------|------|
| Spectrum source | `SpectrumSimulator` (CPD synthetic) | Live RPD/CMTS capture (or emulated amps) | `MockMcpServer` tool bodies |
| Classifier | `RuleClassifier` (floor-lift features) | Bhaskar's CNN | `analyzeSpectrumMeasurements` |
| Orchestrator | deterministic `Orchestrator` (oracle) | LangGraph `LangGraphAgent` + real SLM | identical 5-tool calls |
| MCP transport | in-process Python | FastAPI MCP server on OCP | reference surface (handles+counts) |

The reference surface (handles + counts, the 5 tool signatures, the error/`partial_success`
shapes) is the **stable contract**. As long as a real implementation honors it, Phases 0–3 stay
unchanged — which is exactly what makes Phase 4 a swap, not a rewrite.

---

## Quick-reference: file map

| Area | Path |
|------|------|
| Domain models (schema mirrors) | [src/uil/domain/](src/uil/domain/) |
| CPD simulator | [src/uil/sim/spectrum_simulator.py](src/uil/sim/spectrum_simulator.py) |
| Bootstrap classifier | [src/uil/classifier/rule_classifier.py](src/uil/classifier/rule_classifier.py) |
| Graph localizer | [src/uil/localizer/graph_localizer.py](src/uil/localizer/graph_localizer.py) |
| Mock MCP server + handle store | [src/uil/mcp_server/](src/uil/mcp_server/) |
| Deterministic orchestrator (oracle) | [src/uil/agent/orchestrator.py](src/uil/agent/orchestrator.py) |
| Real LangGraph + SLM agent | [src/uil/agent/langgraph_agent.py](src/uil/agent/langgraph_agent.py) |
| Trace + handoff | [src/uil/agent/trace.py](src/uil/agent/trace.py), [src/uil/agent/handoff.py](src/uil/agent/handoff.py) |
| Provisional configs | [config/cpd_signatures.yaml](config/cpd_signatures.yaml), [config/capture_defaults.yaml](config/capture_defaults.yaml) |
| Step tracker | [EXECUTION_STEPS.md](EXECUTION_STEPS.md) |
