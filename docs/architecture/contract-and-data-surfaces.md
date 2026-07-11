# Contract And Data Surfaces

This document defines the stable seam between the SLM orchestration layer and backend RF data
systems.

## Reference Surface (SLM-facing)

The SLM receives and passes only handles, IDs, and counts:

- `measurementRef`
- `measurementSetRef`
- `ampListRef`
- `classificationSetRef`
- count fields such as `ampCount`, `failedCount`

Reference outputs may return `success`, `error`, or `partial_success` (tool-dependent), with
structured error codes used by orchestrator policy.

## Raw Surface (Backend-only)

Raw RF payloads and topology internals remain in server storage and are never embedded into
reference responses:

- `rawSpectrum` traces: `maxHold`, `minHold`, `average`
- capture metadata and analyzer internals
- topology details used by graph localization

## Stability Rule

As long as an implementation preserves the same reference schema semantics, simulator-based and
live-capture-based backends are interchangeable without changing orchestrator state contracts.

## Tool Sequence Contract

1. `getRPDSpectrumMeasurements`
2. `analyzeSpectrumMeasurements` (RPD)
3. `getAllAmpsInSegment`
4. `getAmpUpstreamSpectrumMeasurements`
5. `analyzeSpectrumMeasurements` (amps)
6. `localizeUpstreamSpectrumImpairmentSource`

This sequence is fixed; SLM responsibility is argument selection and off-happy-path handling.

> Contract note (2026-07-11): tool 4 was renamed from `getAmpSpectrumMeasurements` to the
> canonical `getAmpUpstreamSpectrumMeasurements` to match the CableLabs `shared/*.schema.json`;
> the old name is kept only as a private compatibility alias.
