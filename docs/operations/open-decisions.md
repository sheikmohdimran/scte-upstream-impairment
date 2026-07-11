# Open Decisions And Owners

Track unresolved external decisions that affect model quality, deployment readiness, or contract
assumptions.

## Signal And Model (Irene, Bhaskar)

- CPD signature finalization: floor rise, comb spacing/amplitude, effective band.
- CNN input and normalization alignment for 256-bin upstream traces.
- 7-class legacy model to 9-label contract mapping.
- Confidence calibration thresholds for clean vs impaired boundaries.

## Topology And Platform (Randy)

- ~~Real `getAllAmpsInSegment` payload shape and sample data.~~ **RESOLVED** — CableLabs
  provided RF plant data-package samples (`network-topology-example-1/2.json`): a flat
  `components[]` + directed `edges[]` graph. Parser implemented in
  `src/uil/localizer/plant_topology.py`; confirmed by transcript that a topology tool exists
  (Randy: "we'll have the topology stuff"; Chai building it).
- ~~Trigger contract: SNMP trap and/or Kafka schema.~~ **PARTIALLY RESOLVED** — alarm eval
  set (`alarms.json`) defines the trigger semantics (call/noCall). Gate implemented in
  `src/uil/agent/trigger.py`. Transport (SNMP trap vs Kafka) still open.
- OCP/OpenShift timeline and access readiness.
- Expected concurrent localization scale.

### Topology parser assumptions (implemented; see `plant_topology.py` docstring)

- **A1** `RfSource` (fiber node) is the upstream root; impairments propagate toward it.
- **A2** Edges are downstream (`source -> target` toward subscribers); reversed for upstream
  pathing. Verified: `RfSource` has only outgoing edges in both samples.
- **A3** The whole graph under the root is one segment; per-`portId` leg slicing not yet wired.
- **A4** Roles by type: `RfAmp`=measured node; `RfSplitter/RfTap/RfCoupler/RfPowerInserter`=
  passive common-point candidates; `RfPort/RfCable`=collapsed connectors (cable length summed
  into distance); `device`=subscriber home leaf (dropped); unknown types treated as connectors.
- **A5** Device identity = component `id`; `name` carried for human handoff only.

### Pending items (topology)

- ~~Server-level e2e over a full real plant (74/47 amps) needs a per-amp impairment-label
  injection strategy; current e2e still uses the synthetic amp list.~~ **DONE (2026-07-11):**
  `tests/test_plant_topology.py::test_real_plant_e2e_localizes_per_port` runs the full 6-step
  orchestrator over **every** real RPD-port segment of both plants (plant-1 ports 1/2/3 = 56/2/16
  amps = 74; plant-2 ports 1/2/3/4 = 2/20/20/5 = 47), with deterministic per-amp label injection
  (fault subtree = CPD, rest Clean) and an independent oracle: the localized boundary must equal
  the planted subtree and exclude the clean upstream amps. The live-SLM demo
  (`examples/demo_real_plant_slm.py`) additionally drives segments through the real SLM. Parser +
  localizer remain unit-tested directly against both real files.
- Adopt the CableLabs scenario-overlay format (`basePlant`, `rpdId`, `portId`, `faults`,
  `overrides`, baked `devices`) for full-plant e2e tests. Their extracted corpus contains
  211 scenarios over 9 base plants.
- ~~Align our MCP contracts with the extracted server's current `shared/*.schema.json` before
  changing schemas. In particular, the plant data-package is server configuration, **not** a
  raw `getAllAmpsInSegment` response; do not change the raw schema to expose the data-package.~~
  **DONE (2026-07-11):** reference schemas, Pydantic models, mock server, agent tools/prompts,
  handoff builder and tests aligned to `shared/*.schema.json` (see "Addressed in standalone
  code" below). The data-package stays server config; the raw schema was not changed to expose it.
- ~~Reconcile passive-node output semantics.~~ **ADDRESSED LOCALLY:** topology traversal retains
  passive nodes structurally, while public refs remain RPD/AMP only. A passive internal common
  point maps to the nearest measurable upstream RPD/AMP boundary and remains in the description.
  Confirm this policy with Randy/Irene before claiming parity with their localizer.

### Topology bindings and remaining clarifications (Randy / Chai)

- **C1 — implementation-confirmed:** `rpdId` identifies the `RfSource`; `portId` is resolved
  by `find_rpd_port_node(graph, rpdId, portId)`. Both real graph tools use this binding.
- **C2 — implementation-confirmed:** the server loads/caches the whole configured plant graph
  (`SLM_TOPOLOGY_PATH`), then each tool scopes internally to
  `descendants(rpd_port_node)`. `getAllAmpsInSegment` stores only the resulting amp-id list
  behind `ampListRef`; topology is never sent to the SLM.
- **C3 — implementation-confirmed:** amp identity is component **`id`** (numeric string, e.g.
  `0000000001`), not `name`. Scenarios, stores, and graph tools consistently use ids. The SLM
  was observed hallucinating leading zeros; deterministic handle/id pass-through remains the
  mitigation.
- **C4** Alarm trigger transport (SNMP trap vs Kafka) and field stability of
  `entity.type`/`alarmType`. *(Still open.)*
- **Topology behavior confirmed (2026-07-10):** a single RPD per branch (no successive RPDs).
  Passive nodes remain in the NetworkX graph and affect paths/boundaries, but only RPDs and
  amps are classified and permitted in the current public `deviceRef` output schema.

### Addressed in standalone code (2026-07-11)

- Strict `RfSource` + direct `RfPort` resolution and descendant scoping, without NetworkX or
  `graph-modeling-utils`.
- Canonical `ampListRef -> {rpdId, portId, ampIds}` payload; Tool 3 now chains into Tool 4 for
  data-package topologies.
- Tool 6 no longer scans the handle store for an arbitrary first topology handle.
- Passive internal common points emit schema-valid RPD/AMP boundary refs.
- Focused parser/localizer/schema tests plus full suite: **171 passed, 0 skipped**.

### MCP contract aligned to CableLabs (2026-07-11)

**Decision:** treat the extracted `mcp-server/shared/*.schema.json` as the integration contract
and align our surface to it, because Jay runs our LangGraph agent against the real CableLabs MCP
server — tool names and I/O schemas must match by name or calls fail. Changes (schemas + Pydantic
models + mock server + agent tools/prompts + handoff builder + tests, atomically):

- Canonical tool name `getAmpUpstreamSpectrumMeasurements` (old `getAmpSpectrumMeasurements`
  kept only as a private alias).
- `getAllAmpsInSegment` returns only `{status, ampListRef, ampCount}` (topology stays internal).
- `localize` output moved to per-candidate `upstreamBoundaryDevice` + plural
  `downstreamBoundaryDevices[]` with `supportingDevicesRef`/`cleanBoundaryDevicesRef`/
  `uncertainDevicesRef` handles; removed top-level `likelySourceLocation`/inline evidence/
  `recommendedNextAction`; `impairmentType` input now optional (`default: CPD`).
- Internal `GraphLocalizer` result unchanged; server converts it to the public contract via
  `build_public_localization` and can re-expand evidence handles via `resolve_localization`.
- Full suite after the change: **192 passed, 0 failed**.

### Deterministic device-id binding (2026-07-11)

**Decision:** bind `rpdId`/`portId` into the LangGraph tools at agent construction and remove
them from the SLM-visible tool schemas, so the SLM chooses only *which* tool and never supplies
device identifiers. Rationale: a live run of `google/gemma-4-E4B-it` against a real 56-amp plant
segment sent `rpdId='RPD-0000000001'`/`portId='P1'` (anchored on the old `RPD-1`/`P1` prompt
examples), which `getAllAmpsInSegment` rejected with `INVALID_RPD_PORT`. This is the id-
hallucination failure mode Jay flagged; deterministic binding is the mitigation (versus relying
on LangGraph pass-through alone).

- `getRPDSpectrumMeasurements` / `getAllAmpsInSegment` take no arguments; `localize` takes only
  `impairmentType` + `classificationSetRefs`. `McpToolset` holds the bound ids (from the alarm
  in `run_from_alarm`, else scenario defaults).
- Prompt examples changed from `RPD-1`/`P1` to an explicit "do not pass or reformat ids" rule.
- Verified: the same 56-amp segment now completes all 6 steps and localizes to the injected
  fault. Offline suite **193 passed, 1 skipped**; opt-in live-SLM integration test added.
- Follow-up (open): handle args (`ampListRef`, `measurementSetRef`) remain SLM-supplied; pin
  them too only if handle drift is observed.

### 2026-07-10 sync (Intel/Jay) — items affecting this repo

- **Two localizer implementations exist.** CableLabs has a separate graph-modeling utility
  (their repo); ours is `graph_localizer.py`. Plan: semantic-diff the two and pick the better /
  combine. They will share their utility as a Docker image.
- **Test scenarios available.** The extracted codebase contains 211 overlays:
  `multi_branch` (137), `suspect` (24), `single_boundary` (22), `all_impaired` (15), and
  `common_element` (13), over 9 base plants. Seven additional deployment scenarios exist.
- **Safe replacement plants available.** `mcp-server/input/README.md` explicitly labels its
  bundled plants as example/test plants, not production data. Our two OneDrive fixtures are
  not exact hash matches; their sensitivity is therefore not proven either way. Prefer the
  declared example/test plants for a shareable test suite.
- **CableLabs runtime.** FastMCP server on Kubernetes with Redis + VictoriaMetrics, LangGraph
  over a Gemma model. The graph tools (steps 3 and 6) are real; spectrum acquisition and
  classification tools still use deterministic/schema-accurate mocks or the CNN path.
- **SLM context length.** Their vLLM was capped at 4096 tokens and some topologies hit 4097
  (system prompt + tool schemas + user prompt), failing a few cases; raised to 8192. Our SLM
  path passes handles only, but watch prompt+schema size.
- **RPD-clean short-circuit** already implemented: if step-2 RPD analysis has `impairedCount == 0`
  the orchestrator stops before amp enumeration (no full chain).

## Contract Confirmation (Randy, Irene)

- Error taxonomy completeness by tool.
- `partial_success` payload guarantees (`failedCount`, `failedDevicesRef`).
- Handoff content requirements for operator workflows.

## Current Mitigation

- Keep assumptions explicit in docs and tests.
- Preserve handles-only reference surface.
- Fail closed to escalation on ambiguity and conflicting labels.
- Record every run to a tamper-evident, append-only audit log (`src/uil/agent/trace_store.py`),
  on by default, for regulatory traceability.
