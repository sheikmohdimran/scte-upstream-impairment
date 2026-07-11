# Project Memory — SCTE / vCMTS Upstream Impairment Localization

> Exported from the agent's working repository memory. Durable notes on the design contract,
> current implementation state, assumptions, and pending items. See `corrected-plan.md` for the
> full design plan and `operations/open-decisions.md` for the live decision tracker.

Location: `/home/sdp/wrkdir/scte`
Goal: Agentic (SLM-orchestrated) upstream impairment localization for cable plant. Paper due July 2026.

## MCP contract sources (reconciliation required)

The original contract was vendored from `OneDrive_2_6-17-2026.zip`. The 2026-07-10
`slm-main.zip` contains a newer integration candidate under `mcp-server/shared/`; it has no
`.git` metadata, so treat it as the current CableLabs implementation contract rather than a
provably ordered revision. Align this repo to it before adding more topology behavior.

5 MCP tools, fixed call sequence (aligned to the extracted CableLabs `shared/*.schema.json` on 2026-07-11 — see "MCP contract aligned" below):
1. `getRPDSpectrumMeasurements(rpdId, portId, numBins?=256, start=5MHz, stop=85MHz)` -> measurementRef
2. `analyzeSpectrumMeasurements(measurementRefs[] | measurementSetRef)` -> classificationSetRef, classificationCount, impairedCount, classesPresent[] (classify RPD)
3. `getAllAmpsInSegment(rpdId, portId)` -> `{status, ampListRef, ampCount}` only (topology stays internal server config)
4. `getAmpUpstreamSpectrumMeasurements(ampIds[] | ampListRef, ...)` -> measurementSetRef, measurementCount, failedCount, failedDevicesRef?; partial_success. **Canonical name**; `getAmpSpectrumMeasurements` retained only as a private compatibility alias.
5. `analyzeSpectrumMeasurements(measurementSetRef)` -> classificationSetRef (amps)
6. `localizeUpstreamSpectrumImpairmentSource(rpdId, portId, impairmentType?=CPD, classificationSetRefs[]=[RPD set + amp set])` -> `{status, impairmentType, localizationStatus, confidence, candidateLocations[]}`. Each candidate carries `upstreamBoundaryDevice` + plural `downstreamBoundaryDevices[]`; evidence moves behind `supportingDevicesRef` / `cleanBoundaryDevicesRef` / `uncertainDevicesRef` handles (no inline device arrays, no top-level `likelySourceLocation`/`recommendedNextAction`).

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

## 2026-07: new CableLabs samples and verified fixes — 171 pass, 0 skip

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
  union was initially extended, then corrected after contract verification: passive nodes stay
  internal and `DeviceRef` is again RPD/AMP only. `Topology.ref_for()` walks a passive common
  point upstream to the nearest measurable AMP/RPD boundary; descriptions retain the internal
  passive common-point id.
- `Topology.from_data_package(rpdId, portId, doc)` now strictly resolves the matching `RfSource`
  and direct `RfPort`, then scopes to that port's descendants before connector collapse. Unknown
  sources/ports produce `InvalidRpdPortError`; Tool 3 maps it to `INVALID_RPD_PORT`.
- `ampListRef` now owns a canonical `{rpdId, portId, ampIds}` payload for both synthetic and
  data-package paths, so Tool 4 resolves real-topology handles correctly. Tool 6 reads topology
  directly from the scenario/server configuration instead of scanning for the first amp-list
  handle, eliminating stale/cross-run selection.

### Verified against `slm-main.zip` (2026-07-10)

- Topology tool exists (Randy: "we'll have the topology stuff"; Chai building it). Their server
  loads/caches the whole plant graph from `SLM_TOPOLOGY_PATH` (or a bundled default); the
  data-package is **server configuration, not a tool response**.
- C1/C2: `rpdId` identifies the `RfSource`; `find_rpd_port_node` resolves `portId`; both graph
  tools scope internally to `descendants(rpd_port_node)`. `getAllAmpsInSegment` stores only the
  resulting amp-id list behind `ampListRef` and returns `{status, ampListRef, ampCount}`.
- C3: amp identity is component numeric `id`, not `name`. Scenarios/stores/tools consistently
  use ids. Long ids were observed being hallucinated by the SLM; deterministic argument/handle
  pass-through is required.
- A single RPD exists per branch. Passive components remain structural nodes in the NetworkX
  graph and affect paths, but only RPD/AMP nodes are classified and allowed as public device refs.
- `graph-modeling-utils` is an external internal-PyPI dependency (`>=0.2,<0.3`), not bundled in
  the zip. Their graph algorithms cannot be run independently without that package/access.
- Their bundled `mcp-server/input/*.json` plants are explicitly described as example/test plants,
  not production data. Our OneDrive fixtures are not exact hash matches, so their sensitivity is
  unknown; prefer the declared example/test plants in a shareable suite.

### Scenario corpus available

- Exactly 211 generated overlays over 9 base plants:
  - `multi_branch`: 137
  - `suspect`: 24
  - `single_boundary`: 22
  - `all_impaired`: 15
  - `common_element`: 13
- Seven additional deployment scenarios include experimental `intermittent`, `multi_fault`, and
  `non_coherent` cases; these are not part of the main 211-case generated corpus.
- Overlay shape: `basePlant`, `rpdId`, `portId`, `impairmentType`, `faults`, `overrides`, baked
  `devices`, and an `expect` oracle. Fault propagation is generated coherently as
  `origin ∪ ancestors(origin)`.

### Additional reusable components from `slm-main.zip`

- **FastMCP service:** one HTTP MCP server exposing all five tools at `/mcp`, plus `/healthz`,
  `/readyz`, `/execute`, `/diagnose`, `/tools`, and `/tool-revisions`.
- **Direct adapters for our orchestrator:** `ToolAdapter` (in-process) and `HttpToolAdapter`
  (real HTTP round trip) already implement the interface expected by `uil.agent.Orchestrator`.
  Their `pyproject.toml` pins this repo's old `impl-plan-2026-06-24` revision, so compatibility
  with current `main` must be tested before reuse.
- **Pluggable handle store:** memory or Redis; short opaque refs, deep-copy semantics, UUID ids,
  and native per-key TTL (`SLM_HANDLE_TTL`, default 3600 s). This is production-ready compared
  with our in-process-only `HandleStore`.
- **Telemetry store:** memory or VictoriaMetrics, separating large spectrum time series from
  Redis metadata/handles. Useful when real captures replace mocks.
- **Scenario generator:** creates topology-coherent overlays and expected localization oracles;
  directly addresses our full-plant label-injection gap.
- **Schema-to-tool export:** builds self-contained OpenAI/vLLM tool specs from the reference
  schemas, inlining transitive `$defs`, and adds a `noAction` eval convention.
- **Immutable tool-spec revisions/A-B testing:** content-hashed revisions with parent lineage,
  atomic writes, and `/tools?rev=` serving. Preserve authored JSON key order: their tests found
  that sorting semantically identical tool JSON changed gpt-oss accuracy from 56% to 48%.
- **Readiness and deployment:** topology/Redis/VictoriaMetrics readiness checks, Bearer auth on
  `/mcp`, Kubernetes manifests, and real-service CI. Security caveat: `/execute`, `/diagnose`,
  and revision routes need equivalent auth before production exposure.
- **Evaluation lessons:** vLLM context 4096 was exceeded by one token on some cases; 8192 fixed
  it. Their integration report says graph boundary detection was correct on all 37 tested
  scenarios, while an older hard-assertion harness was wrong on branching topologies.

### Localizer differences requiring a decision

- Their localizer keeps passive nodes structurally but emits only RPD/AMP boundary refs; ours
  matches this (passive nodes internal, public refs RPD/AMP only).
- ~~Their result carries `supportingDevicesRef`, `cleanBoundaryDevicesRef`, and
  `uncertainDevicesRef` handles per candidate; ours returns device arrays inline.~~
  **ADOPTED (2026-07-11):** our public localize output now carries the same per-candidate
  handles; inline device arrays are gone from the wire surface. The internal
  `GraphLocalizer` result keeps the inline evidence; the mock server converts it to the
  handle-backed public contract via `build_public_localization`.
- Their conflict policy demotes contradictory upstream-clean nodes to uncertain and retries;
  ours escalates on multiple impairment labels but does not implement the same topology-conflict
  policy. *(Still a decision — behavioural, not contract.)*
- Their multi-span/common-element result is always `low_confidence`; ours can choose a passive
  common point. Compare semantics with Randy/Irene before merging either approach.
- ~~Their `getAmpUpstreamSpectrumMeasurements` tool name differs from our
  `getAmpSpectrumMeasurements`; adapters currently bridge the name.~~ **RESOLVED (2026-07-11):**
  `getAmpUpstreamSpectrumMeasurements` is now our canonical tool name across schemas, mock
  server, agent tools/prompts and traces; `getAmpSpectrumMeasurements` remains only as a
  private alias so an adapter bridge is no longer required.

### Remaining clarification

- C4 alarm transport (SNMP trap vs Kafka) and stability of alarm field names remains open.

### Pending

- ~~Align our reference/raw schemas and Pydantic output models to the current shared contract;
  specifically resolve handle-vs-inline localization evidence, canonical amp-tool naming, and
  `getAllAmpsInSegment` fields.~~ **DONE (2026-07-11)** — see "MCP contract aligned to CableLabs"
  below. Passive public refs were already corrected to RPD/AMP only.
- Adopt the 211-overlay scenario format/generator for full-plant e2e tests.
- Replace the OneDrive topology fixtures with declared example/test plants.
- Validate their `ToolAdapter`/`HttpToolAdapter` against our current `main` and FastMCP server.
- Compare their graph-modeling localizer with ours when `graph-modeling-utils`/Docker is available.
- Coordinate the spectrum/CNN ingestion interface (5-184 MHz / dBuV / maxhold).

### MCP contract aligned to CableLabs (2026-07-11) — DONE

**Why:** the extracted `slm-main` server on the CableLabs cluster is the real integration
target (Jay swapped our mock MCP server for it, keeping LangGraph as the SLM wrapper). The
SLM's advertised tool names and the server's registered tools must match by name and I/O
schema, or tool calls fail to resolve / responses fail to deserialize. Their
`mcp-server/shared/*.schema.json` is the source of truth; our surface disagreed. This change
makes the same client code work against both our mock and their server.

**What changed (schemas, Pydantic models, mock server, agent tools/prompts, handoff builder,
and tests, together):**

- **Canonical amp tool name.** `getAmpSpectrumMeasurements` -> `getAmpUpstreamSpectrumMeasurements`
  everywhere (schema files renamed via `git mv`, mock server method, LangGraph `StructuredTool`
  name + system prompt, orchestrator call + trace name, eval `FULL_CHAIN`). The old name stays
  as a private one-line alias in the mock server for backward compatibility.
- **Tool 3 output trimmed.** `getAllAmpsInSegment` returns only `{status, ampListRef, ampCount}`
  (dropped `segmentId`/`topologyTimestamp`); topology remains internal server configuration.
- **Localization evidence behind handles.** The public `localize` output now uses per-candidate
  `upstreamBoundaryDevice` + plural `downstreamBoundaryDevices[]` with
  `supportingDevicesRef`/`cleanBoundaryDevicesRef`/`uncertainDevicesRef` handles. Removed the
  top-level `likelySourceLocation`, inline `supportingDevices`/`cleanBoundaryDevices`/
  `uncertainDevices` arrays, and `recommendedNextAction`. Input `impairmentType` is now optional
  with `default: CPD` and dropped from `required` (matching their input schema).
- **Two-layer model split.** `domain/localization.py` keeps the internal `LocalizationResult`
  (inline evidence, single `likelySourceLocation`) so `GraphLocalizer` stays simple and
  unit-testable, and adds public `BoundaryCandidate` + `PublicLocalizationResult` plus
  `build_public_localization(rpd_id, port_id, result, put_handle)` that creates the evidence
  handles at the server boundary (store lives in the server, not the localizer).
- **Evidence resolver for scoring.** `MockMcpServer.resolve_localization()` expands the opaque
  `*Ref` handles back into inline device lists so the eval harness/grader can score device ids
  without changing the wire output.
- **Tests updated together:** added a localize-output schema-conformance test; updated grader
  (`source_match` now reads `candidateLocations`), topology/e2e/langgraph/cnn/plant-topology
  tests to the new candidate shape and canonical tool name. Full suite: **192 passed, 0 failed**.

**Not done here (deferred, unchanged):** FastMCP transport, `HttpToolAdapter`, Redis/VM storage,
adopting their 211-overlay corpus, and comparing their graph-modeling localizer.

### Deterministic device-id binding (2026-07-11) — DONE

**Why:** running the LangGraph agent with the live `google/gemma-4-E4B-it` SLM against a REAL
plant segment reproduced Jay's hallucination concern. On the 56-amp port-1 segment the SLM
passed `rpdId='RPD-0000000001'`, `portId='P1'` — it copied the *format* of the old prompt
examples (`RPD-1`/`P1`) and mangled the real numeric id. Tool 1 accepted it (no topology check),
but `getAllAmpsInSegment` correctly returned `INVALID_RPD_PORT` and the agent escalated. The
16-amp port-3 segment happened to succeed with the same prompt — i.e. non-deterministic id
handling, exactly the failure mode to remove.

**Fix (in `agent/langgraph_agent.py`):**

- `McpToolset` now owns `rpd_id`/`port_id` (defaulted from the scenario, overridden from the
  alarm in `run_from_alarm`). The RPD/port are bound deterministically; the SLM never supplies
  them. Tool schemas dropped the id fields: `getRPDSpectrumMeasurements` and
  `getAllAmpsInSegment` take **no arguments** (`_NoArgs`); `localizeUpstreamSpectrumImpairmentSource`
  takes only `impairmentType` + `classificationSetRefs`. The SLM still chooses *which* tool and
  passes short tool-produced handles (`measurementSetRef`, `ampListRef`) verbatim.
- System prompt neutralized: removed the `RPD-1`/`P1` examples and added "you do NOT pass rpdId
  or portId to any tool, and you never invent or reformat device identifiers."

**Evidence:** after the change the same 56-amp segment runs all 6 steps with the correct bound
ids (`rpdId='0000000001'`, `portId='1'`), `impairedCount: 43`, and localizes to the injected
fault subtree. Offline suite **193 passed, 1 skipped**; an opt-in live-SLM integration test
(`test_langgraph_agent_real_plant_live_slm`, skipped unless `SLM_BASE_URL` is set) exercises the
real 16-amp segment end to end. New helper `examples/demo_real_plant_slm.py` reproduces the run.

**Note / follow-up:** handle args (`ampListRef`, `measurementSetRef`) are still SLM-supplied;
they are short, backend-issued handles the SLM echoes, so lower risk. A stricter design could
also pin/normalize those from the run context if handle drift is ever observed.
