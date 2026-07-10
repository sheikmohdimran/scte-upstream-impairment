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

- Server-level e2e over a full real plant (74/47 amps) needs a per-amp impairment-label
  injection strategy; current e2e still uses the synthetic amp list. Parser + localizer are
  unit-tested directly against both real files.
- Wire per-`portId` segment slicing once binding (C1/C2) is confirmed.
- Update `schemas/tools/raw/getAllAmpsInSegment.schema.json` to the data-package/edge-list shape.

### Clarifications needed from CableLabs (Randy / Chai)

- **C1** How does an alarm's `rpdId`/`portId` bind to the topology graph? Assumed
  `RfSource` ≈ the RPD/node and `portId` selects one of its `RfPort` subtrees (one `RfSource`
  per sample file today).
- **C2** Does `getAllAmpsInSegment` (or the localize tool) receive the whole plant graph or a
  pre-sliced per-segment subgraph? Randy's "DB-record handle to the set of amps" suggests
  pre-slicing.
- **C3** Is the real amp identifier the component `id` or the `name` (e.g. `AMP1`)?
- **C4** Alarm trigger transport (SNMP trap vs Kafka) and field stability of
  `entity.type`/`alarmType`.

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
