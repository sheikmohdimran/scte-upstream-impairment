# Open Decisions And Owners

Track unresolved external decisions that affect model quality, deployment readiness, or contract
assumptions.

## Signal And Model (Irene, Bhaskar)

- CPD signature finalization: floor rise, comb spacing/amplitude, effective band.
- CNN input and normalization alignment for 256-bin upstream traces.
- 7-class legacy model to 9-label contract mapping.
- Confidence calibration thresholds for clean vs impaired boundaries.

## Topology And Platform (Randy)

- Real `getAllAmpsInSegment` payload shape and sample data.
- Trigger contract: SNMP trap and/or Kafka schema.
- OCP/OpenShift timeline and access readiness.
- Expected concurrent localization scale.

## Contract Confirmation (Randy, Irene)

- Error taxonomy completeness by tool.
- `partial_success` payload guarantees (`failedCount`, `failedDevicesRef`).
- Handoff content requirements for operator workflows.

## Current Mitigation

- Keep assumptions explicit in docs and tests.
- Preserve handles-only reference surface.
- Fail closed to escalation on ambiguity and conflicting labels.
