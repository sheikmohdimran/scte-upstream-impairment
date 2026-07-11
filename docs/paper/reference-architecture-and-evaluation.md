# Bounded, Instruction-Guided Agentic CPD Localization — Paper Sections

> Draft manuscript sections. Efficiency figures are measured on the local platform via
> `examples/eval_slm.py`; items marked **[PLACEHOLDER]** require a cloud baseline run, power
> instrumentation, or fine-tuning experiments not yet performed.

---

# Reference Architecture: Bounded, Instruction-Guided Agentic Workflow

## Design thesis

We adopt a deliberately *bounded* agentic design: a small language model (SLM) is granted
autonomy only over **orchestration**—which tool to invoke next, how to interpret opaque
results, and how to recover from errors—while all diagnostically consequential computation
(signal classification and fault localization) remains in deterministic, independently
verifiable software. The SLM is further *instruction-guided*: a fixed system prompt encodes the
intended tool sequence and an explicit recovery policy, so the model operates as a constrained
controller rather than an open-ended planner. This separation is motivated by three
requirements common to physical-infrastructure diagnostics: (i) **reliability** under a
resource-constrained on-premises SLM, (ii) **safety**, in that a hallucinated numeric result
must not be able to mislocate a physical fault, and (iii) **auditability** for regulated
operational environments.

## Separation of concerns

The workflow partitions responsibility into an *agentic control plane* (the SLM) and a
*deterministic reasoning core*:

```
        ┌───────────────────────── Agentic control plane (SLM) ─────────────────────────┐
Alarm ─▶│  trigger gate → tool selection → handle plumbing → error/partial-result policy │
        └───────────────────────────────────┬───────────────────────────────────────────┘
                                             │ opaque handles + counts (never raw spectra)
        ┌────────────────────────────────────▼──────────────────────────────────────────┐
        │ Deterministic reasoning core:  spectrum classifier  |  graph localizer          │
        └─────────────────────────────────────────────────────────────────────────────── ┘
```

The SLM never observes raw spectra, topology internals, or classifier artifacts; it exchanges
only **reference-surface** objects—opaque handles (`measurementRef`, `ampListRef`,
`classificationSetRef`, `uncertainDevicesRef`) and scalar counts. The deterministic core owns
the raw surface and the numerically sensitive decisions. Consequently, the agent's freedom is
confined to *sequencing and control flow*, not to inference over signals.

## Bounding mechanisms

The workflow is bounded along four explicit axes:

1. **Closed tool set.** The agent may call only a fixed set of measurement/analysis tools; it
   cannot synthesize new capabilities or free-form actions.
2. **Fixed, one-shot sequence.** Tools follow a canonical order over the amplifier leg and are
   executed *once* per run rather than via recursive drill-down, which bounds run length and
   cost and eliminates unbounded exploration.
3. **Deterministic subcomputations.** Classification and localization are pure functions of
   their inputs, so identical evidence yields identical diagnoses regardless of the model's
   phrasing.
4. **Gated entry.** A deterministic pre-condition (the alarm-trigger gate) decides whether the
   workflow runs at all, preventing spurious activation.

The canonical chain is:

| Step | Tool | Produces |
|---|---|---|
| 1 | `getRPDSpectrumMeasurements` | RPD `measurementRef` |
| 2 | `analyzeSpectrumMeasurements` | RPD `classificationSetRef`, counts |
| 3 | `getAllAmpsInSegment` | `ampListRef`, amp count |
| 4 | `getAmpSpectrumMeasurements` | amp `measurementSetRef` (or `partial_success`) |
| 5 | `analyzeSpectrumMeasurements` | amp `classificationSetRef` |
| 6 | `localizeUpstreamSpectrumImpairmentSource` | localized source or `low_confidence` + next action |

## Instruction guidance

The SLM receives a system prompt that specifies (a) the happy-path sequence above, (b) the
handle to thread from each step into the next, and (c) an explicit recovery policy. Guidance is
*procedural but not scripted*: the model retains discretion to deviate when tools return errors
or partial results, which is precisely the behavior the architecture intends the agent to own.
This design keeps the prompt as the single, inspectable specification of agent behavior, and
lets a deterministic reference implementation of the same procedure serve as an oracle.

## Failure handling and escalation

The recovery policy the agent is instructed to enforce is:

- **Transient errors** (e.g., stale or momentarily unavailable measurements) → retry the same
  call once.
- **Non-transient errors** (e.g., device unreachable, topology unavailable) → stop and escalate
  with a concise reason; do not loop.
- **Partial success** (a subset of amplifier legs fail to measure) → proceed with the measured
  subset; unmeasured devices are surfaced as an uncertainty handle downstream.
- **Low-confidence localization** (conflicting or insufficient evidence) → do not continue
  calling tools; escalate to a human with a structured handoff.

Because the consequential decision (whether evidence supports a confident localization) is
computed deterministically, escalation is triggered by verifiable conditions rather than by
model self-assessment.

## Auditability

Every run is recorded as an append-only, tamper-evident trace (a hash-chained sequence of tool
calls, arguments, reference-surface results, and the terminal outcome), containing only handles
and counts—never raw signals. This yields a complete, replayable account of *why* the agent
acted as it did, satisfying regulatory traceability while preserving the handle/raw separation.

## Design rationale

Bounding the agent trades generality for properties that matter in operational deployment.
Fixing the tool set and sequence makes the search space small enough that a modest SLM can
orchestrate reliably; keeping classification and localization deterministic guarantees that
agent variability cannot alter a physical diagnosis; and the handle-only reference surface both
reduces the model's context burden and structurally prevents raw-signal leakage. The same
procedure admits a deterministic reference implementation, enabling the agent to be *graded*
against independently-derived ground truth—a property we exploit in our evaluation (§Results).

---

# The CPD Agentic Flow, End to End

We illustrate the architecture with a single concrete resolution of a common-path distortion
(CPD) impairment, tracing one run from alarm to auditable outcome.

**(1) Entry.** An upstream fault manifests as elevated forward-error-correction (FEC) counters
at a remote PHY device (RPD) port. The resulting alarm enters the deterministic trigger gate,
which fires the workflow only when the alarm is an upstream FEC-error type raised on an RPD
upstream port; all other alarms short-circuit with a `not_triggered` verdict and zero tool
calls. On a positive alarm, the workflow is seeded with the alarm's `rpdId`/`portId`.

**(2) Confirm at the RPD.** The agent calls `getRPDSpectrumMeasurements` to capture the upstream
spectrum at the port and `analyzeSpectrumMeasurements` to classify it. The agent observes only a
classification handle and summary counts (impaired count, classes present); the raw 5–85 MHz
trace never enters its context. A non-zero impaired count for a CPD-consistent signature
confirms an impairment worth localizing; a clean result terminates the run as a possible
intermittent condition.

**(3) Enumerate the leg.** `getAllAmpsInSegment` returns a handle to the amplifier cascade fed
by that port, together with an amplifier count—one-shot enumeration of the whole leg rather than
recursive descent.

**(4) Measure the cascade.** `getAmpSpectrumMeasurements` captures spectra at every amplifier via
the list handle. This step may return `partial_success` when individual legs are unreachable;
the agent is instructed to proceed with the measured subset, and the unmeasured devices are
later surfaced as an uncertainty handle.

**(5) Classify and (6) localize.** A second `analyzeSpectrumMeasurements` classifies the
amplifier set, and `localizeUpstreamSpectrumImpairmentSource` pools the RPD and amplifier
classifications and runs deterministic common-point analysis: since an upstream impairment is
seen by the faulted device and everything downstream of it, the localizer identifies the
boundary between the impaired and clean regions.

**Worked outcome.** In a representative run, the localizer reports `impairmentType = CPD`, a
branch common point beneath amplifier `A2` (the shared ancestor of the impaired amplifiers
`{A2, A4, A8}`), and—because one clean sibling leg failed to measure—a `low_confidence` status
with an uncertainty handle. The agent, following its recovery policy, does not continue calling
tools; it emits a structured human handoff naming the candidate branch and the unmeasured
device. The full run is written to an append-only, tamper-evident audit record containing only
handles and counts.

The salient point is that every diagnostically consequential value—the classification and the
localized boundary—was produced deterministically; the SLM's contribution was to sequence the
calls, thread the handles, absorb the partial failure, and decide to escalate.

---

# The Compute Thesis: Running on General-Purpose Compute

The central systems claim of this work is that a bounded diagnostic workflow of this kind can be
run in full by a **right-sized local model on a general-purpose CPU, with no dedicated AI
accelerator.** In our deployment the SLM (`gemma-4-E4B-it`, a ~4-billion-parameter
instruction-tuned model) is served through vLLM (bf16) behind an OpenAI-compatible endpoint and
executes the entire CPD workflow—trigger, six-step orchestration, error recovery, and human
handoff—on an Intel Xeon 6 (Granite Rapids-SP) platform.

**Why a CPU suffices here.** The workflow is deliberately low-reasoning: the model performs
short, structured tool-selection turns over a closed tool set, not open-ended chain-of-thought
over long contexts. Empirically, a full resolution takes on the order of **5–6 model turns and
generates only ~220 tokens** (§Results); the heavy numerical work (classification, localization)
is deterministic software, not model inference. A bounded control task of this shape does not
require a frontier model, and therefore does not require a dedicated accelerator to host
one—which is precisely what makes a small local model the rational choice rather than a
compromise.

**Intel AMX.** The platform's built-in Advanced Matrix Extensions accelerate the low-precision
matrix multiplications that dominate transformer inference. On our test system the CPU exposes
`amx_tile`, `amx_bf16`, and `amx_int8`, allowing the SLM to run efficiently on the CPU without
offload to a discrete device. AMX is the enabling mechanism that turns "possible on a CPU" into
"practical on the CPU operators already own."

**An honest comparison.** We do not claim CPU inference outperforms a GPU. A GPU will serve more
tokens per second, and our measured per-resolution latency (median ≈13 s, §Results) reflects CPU
inference. Our claim is narrower and, for this application, more relevant: for a bounded task
hosted on a modest model, a general-purpose CPU already deployed at the network edge resolves
the workflow at **lower cost and power and without provisioning specialized hardware.** This is
the operational sense in which the work minimizes reliance on dedicated AI accelerators, the
stated goal of the abstract.

**Economics anchor.** The economic argument rests on right-sizing. Because a resolution generates
only ~220 tokens across ~6 short turns, the marginal inference cost on an already-owned edge
server is effectively zero, whereas the same workload on a metered cloud frontier model incurs a
per-resolution token charge. **[PLACEHOLDER: cloud baseline $/resolution and tokens/resolution
vs. local near-zero marginal cost.]** Because the task is bounded, fine-tuning can shrink the
model further—on the order of 12B toward 3B or 2B parameters **[PLACEHOLDER: fine-tuned model +
retained accuracy]**—which frees concurrency and context budget on the same server and improves
the cost and power picture without changing the workflow.

---

# Benchmarking Methodology

We evaluate along two orthogonal axes: **functional correctness** (does the agent orchestrate
the workflow correctly?) and **efficiency** (at what latency, turn count, and token cost?).

**Functional correctness.** We use a seeded scenario generator that plants a known CPD fault in
a randomly-shaped amplifier tree and derives the correct outcome—trigger decision, whether
localization is warranted, the fault location, and the expected uncertainty—**independently of
the localizer under test** (an import guard enforces that the generator never imports the
localizer). Each generated case is graded by a rubric that scores the agent's orchestration, not
the deterministic math it drives:

- *Hard criteria* (gating): correct trigger decision; localization invoked only when warranted;
  no fabricated confident localization; the reported source references the planted fault.
- *Soft criteria* (reported, non-gating): normalized final status, impairment label,
  uncertainty-handle emission, tool-sequence order, error-recovery behavior, and tool-call count.

Two agents run the identical suite: a deterministic reference orchestrator, which serves as an
oracle and must match ground truth on every case, and the SLM under test. Cases span nine fault
dimensions (single-device, branch, span, partial measurement failure, clean/intermittent,
non-transient error, transient recovery, transient failure, and non-triggering alarms). The
generator and the mock measurement server are both seeded, so any reported figure is
reproducible from its seed (`make eval-slm PER=<n> SEED=<s>`).

**Efficiency.** For each resolved case we record end-to-end wall-clock latency, the number of
model turns, and token usage (input/output/total, read from the endpoint's `usage_metadata`).
We report medians and means over the *triggered* cases (those that actually invoke the SLM;
non-triggering alarms short-circuit before inference). Power measurement, sustained
multi-request throughput, and a cloud cost baseline are **[PLACEHOLDER]** — not yet instrumented.

**Hardware.** The system under test is served with vLLM (`gemma-4-E4B-it`, bf16) on an Intel
Xeon 6767P (Xeon 6 / Granite Rapids-SP), 2 sockets × 64 cores (256 threads), with AMX
(`amx_tile`, `amx_bf16`, `amx_int8`). The cloud baseline for the economic comparison is
**[PLACEHOLDER: model + endpoint]**.

**Scope.** All tool responses are served by a mock MCP server that reproduces the
reference-surface contract; the classifier uses the deterministic rule-based path, as the CNN
and full spectrum generator are out of scope here (§Conclusions). Consequently these benchmarks
isolate orchestration quality and inference efficiency, not classifier accuracy on real plant
signals. Impairments are restricted to CPD, the one class the simulator and rule classifier
round-trip faithfully.

---

# Results

**Functional correctness.** The deterministic reference orchestrator matches ground truth on
every generated case across all seeds, validating the harness. On the CPD suite (three cases per
dimension, 27 cases), the local SLM (`gemma-4-E4B-it`) passes **27/27 (100%)** on all hard
criteria; 24 of the 27 cases invoked the SLM (the three non-triggering alarms short-circuit at
the gate). The broader repository test suite reports 159 passing tests.

| Fault dimension | Cases | Passed (hard criteria) |
|---|---|---|
| single_device | 3 | 3/3 |
| branch | 3 | 3/3 |
| span | 3 | 3/3 |
| partial_failure | 3 | 3/3 |
| clean / intermittent | 3 | 3/3 |
| non_transient_error | 3 | 3/3 |
| transient_recovery | 3 | 3/3 |
| transient_fail | 3 | 3/3 |
| not_triggered | 3 | 3/3 |
| **Total** | **27** | **27/27 (100%)** |

The model correctly (i) gated non-triggering alarms with zero tool calls, (ii) drove the
six-step chain in order and threaded handles into localization, (iii) proceeded through partial
measurement failures to a `low_confidence` localization with an uncertainty handle, and (iv)
escalated on unrecoverable errors without looping.

**Efficiency (measured, Xeon 6767P, 24 triggered resolutions).**

| Metric | Value |
|---|---|
| Latency / resolution | median **12.94 s**, mean 11.21 s, max 15.28 s |
| Model turns / resolution | mean **5.6** |
| Tokens / resolution (total) | mean **≈7,900** |
| Tokens / resolution (generated / output) | mean **≈220** |
| Total tokens over 24 resolutions | 190,097 |

Two observations. First, generation is tiny (~220 output tokens per resolution): the model emits
short, structured tool calls, corroborating the "low-reasoning" premise behind the compute
thesis. Second, total tokens are dominated by input, because each ReAct turn re-sends the system
prompt, tool schemas, and history; this is the natural lever for optimization (prompt/schema
compaction, KV-cache reuse) and a direct beneficiary of fine-tuning to a smaller model.

**Behavioral observation.** The only recurring soft-criterion deviation was in error recovery:
on a persistently stale RPD measurement, the model retried more than the single retry prescribed
by the reference policy before terminating. Because it still reached the correct terminal state,
the hard criteria passed; we report this as a calibration signal for prompt or policy tuning
rather than a failure.

**Not yet measured. [PLACEHOLDER]** cloud-baseline tokens and cost per resolution; platform power
and derived cost-per-resolution; sustained throughput and tail latency under concurrency;
fine-tuned smaller-model accuracy/latency.

---

# Compliance & Operator Considerations

For an operator, the bounded design is a feature, not a limitation. The agent follows a **fixed,
reviewable sequence** of named tools with a documented recovery policy, and every run yields an
append-only, tamper-evident audit record of the calls made, the handles exchanged, and the
terminal decision. This is precisely the form of traceability that operators and their auditors
require to trust an automated diagnostic in production: behavior is specified by an inspectable
prompt, bounded by a closed tool set, and reconstructable after the fact from the audit
trail—without exposing raw subscriber-plant signals, since only handles and counts are recorded.

The **human-in-the-loop** boundary is drawn by verifiable conditions rather than model judgment.
Some failures deliberately return to a person: a device that cannot be reached, an unavailable
topology, or a low-confidence/conflicting localization all produce a structured handoff naming
the affected devices and the reason. Other failures the agent is authorized to work around: a
missing amplifier measurement (`partial_success`) does not abort the run, because the
deterministic localizer can still resolve the boundary from the measured subset while flagging
the unmeasured devices as uncertain. The result is an automation envelope that is aggressive
where it is safe and conservative where it is not, with the dividing line encoded in policy and
enforced by deterministic checks.

---

# Deployment Blueprint

The workflow is designed to run where the telemetry is freshest: at the **headend, hub, and
metro-edge tier, co-located with the vCMTS.** Placing inference next to the data source avoids
backhauling spectra to a centralized service and keeps raw plant signals on-premises, consistent
with the handle/raw separation.

Crucially, the compute target is **the same Intel Xeon-SP family operators already deploy** for
vCMTS and adjacent network functions; our benchmarks run on an Intel Xeon 6 (Granite Rapids-SP,
6767P) with AMX. Because the SLM is served on the CPU via vLLM with AMX acceleration, the
diagnostic can be **consolidated onto infrastructure the operator already owns** rather than
requiring a separate accelerator tier. At the measured ~13 s median per resolution and ~220
generated tokens, a single node hosts the workflow as an incremental workload; precise
per-socket resolution throughput and recommended concurrency await the throughput benchmark
**[PLACEHOLDER: resolutions/second per socket, memory footprint, concurrency per node]**.

---

# Conclusions & Future Work

**What is demonstrated.** We present a bounded, instruction-guided agentic architecture in which
a small local model orchestrates a fixed six-step CPD localization workflow while classification
and localization remain deterministic and auditable. On a reproducible, ground-truth-graded
benchmark the model resolves the workflow with full hard-criteria accuracy (27/27), and it does
so on general-purpose Intel Xeon 6 CPUs with AMX—no dedicated accelerator—supporting the
abstract's goal of minimizing reliance on specialized AI hardware.

**What is exploratory.** The efficiency figures are reported on mocked tools and the rule-based
classifier; they characterize orchestration and inference, not classifier accuracy on real
signals. The fine-tuning results toward smaller models are a direction we outline rather than
fully establish here.

**Consciously out of scope.** This paper does not address the upstream alarm-triage orchestrator
that would select among impairment types, the configuration tool, or the downstream
action/ticketing step; we treat these as adjacent components in the broader system.

**Toward real plant data.** The measurement tools here are mocked to the reference-surface
contract. Validation against real plant spectra—as the CableLabs environment comes online—is the
primary next step, including replacing the rule classifier with the CNN over realistic captures.

**Economic conclusion.** The closing argument is an economic reframe: scaling AI across a network
does not depend on deploying frontier models everywhere, but on **right-sizing models to bounded
tasks.** For workloads like CPD localization, a small local model on existing CPUs is not merely
sufficient—it is the cost- and power-rational choice, and the token-economics contrast with a
cloud frontier baseline quantifies the gap **[PLACEHOLDER]**.

**Future direction.** We aim to extend the bounded pattern from CPD to additional upstream
anomalies, to compose these bounded agents toward a broader multi-agent diagnostic vision, to
pursue fine-tuning toward still-smaller models (freeing concurrency and context on the same
hardware), and to validate the full pipeline on real plant data.
