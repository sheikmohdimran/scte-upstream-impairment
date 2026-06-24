# SCTE Upstream Impairment Productionization Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` agent (recommended) or `executing-plans` agent to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare Intel-Sandbox/scte-upstream-impairment as a publishable GitHub repository with consolidated documentation and a production-ready execution path from schema contract to agentic localization.

**Architecture:** Keep the existing 6-step MCP orchestration and handles-only reference surface unchanged. Harden each seam (schema validation, simulator/classifier/localizer behavior, off-happy-path orchestration, and handoff artifacts) with focused tests and minimal code changes. Package the repository with deterministic demos, CI gates, and explicit operational assumptions.

**Tech Stack:** Python 3.12, pytest, pydantic v2, jsonschema, LangGraph/LangChain, Make, GitHub Actions

---

## Scope Check

The approved design spans one tightly-coupled subsystem (upstream impairment localization pipeline). A single plan is appropriate because simulator, classifier, localizer, and orchestrator are coupled by the same MCP tool contract and must ship together.

## File Structure Map

- [ ] `README.md` (modify): Consolidated project entrypoint for architecture, assumptions, demo, and open decisions.
- [ ] `CABLELABS-HANDOFF.md` (modify): Keep as audience-specific handoff, trim duplicated setup content that moves to README.
- [ ] `EXECUTION_STEPS.md` (modify): Track milestone status and reflect completion criteria tied to tests.
- [ ] `docs/architecture/contract-and-data-surfaces.md` (create): Stable reference-surface contract and mock-to-real swap seam.
- [ ] `docs/operations/open-decisions.md` (create): Explicit unresolved CableLabs/Intel dependencies and owner tracking.
- [ ] `src/uil/mcp_server/server.py` (modify): Contract guardrails and richer partial-success metadata consistency.
- [ ] `src/uil/localizer/graph_localizer.py` (modify): Topology-shape resilience and conflict handling rules.
- [ ] `src/uil/agent/orchestrator.py` (modify): Retry/escalation policy normalization.
- [ ] `src/uil/agent/handoff.py` (modify): Deterministic diagnosis summary shape.
- [ ] `tests/test_schema_conformance.py` (modify): Contract-level assertions for success/error/partial_success variants.
- [ ] `tests/test_localizer.py` (modify): Flat-topology, conflicting labels, uncertain-branch scenarios.
- [ ] `tests/test_agent_e2e.py` (modify): Retry-on-transient and escalation flows.
- [ ] `.github/workflows/ci.yml` (create): Repo CI gate for tests and deterministic demo smoke.
- [ ] `.gitignore` (create or modify): Python, venv, cache, and generated artifacts exclusions.

### Task 1: Initialize Repository And GitHub Remote (Intel-Sandbox/scte-upstream-impairment)

**Files:**
- Create: `.gitignore`
- Modify: `README.md`
- Modify: `EXECUTION_STEPS.md`
- Test: command-line git and gh validation

- [ ] **Step 1: Write a failing repository bootstrap check script**

```bash
cat > /tmp/repo_bootstrap_check.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /home/sdp/wrkdir/scte/upstream-impairment

test -d .git
git remote get-url origin | grep -F "Intel-Sandbox/scte-upstream-impairment"
SH
chmod +x /tmp/repo_bootstrap_check.sh
```

- [ ] **Step 2: Run check to verify it fails before repo setup**

Run: `bash /tmp/repo_bootstrap_check.sh`
Expected: FAIL with `.git` missing or `origin` missing.

- [ ] **Step 3: Write minimal implementation for repo bootstrap**

```bash
cd /home/sdp/wrkdir/scte/upstream-impairment
git init
cat > .gitignore <<'EOF'
__pycache__/
.pytest_cache/
*.pyc
*.pyo
*.pyd
.venv/
venv/
.env
.DS_Store
coverage.xml
htmlcov/
EOF

git add .
git commit -m "chore: initialize upstream-impairment repository"

gh repo create Intel-Sandbox/scte-upstream-impairment --private --source=. --remote=origin --push
```

- [ ] **Step 4: Re-run bootstrap check to verify pass**

Run: `bash /tmp/repo_bootstrap_check.sh`
Expected: PASS with no output and exit code 0.

- [ ] **Step 5: Commit**

```bash
git add .gitignore README.md EXECUTION_STEPS.md
git commit -m "chore: bootstrap repo and set GitHub remote"
```

### Task 2: Consolidate README From Approved Design Documents

**Files:**
- Modify: `README.md`
- Modify: `CABLELABS-HANDOFF.md`
- Modify: `EXECUTION_STEPS.md`
- Create: `docs/architecture/contract-and-data-surfaces.md`
- Create: `docs/operations/open-decisions.md`
- Test: `tests/test_readme_consolidation.py`

- [ ] **Step 1: Write the failing test for required README sections**

```python
# tests/test_readme_consolidation.py
from pathlib import Path


def test_readme_contains_required_sections():
    readme = Path("README.md").read_text(encoding="utf-8")
    required = [
        "## What This Repository Implements",
        "## MCP Tool Chain (Fixed Sequence)",
        "## Data Surfaces: Handles vs Raw",
        "## Assumptions And Risks",
        "## Open Decisions And Owners",
        "## Demo Quickstart",
    ]
    for section in required:
        assert section in readme, f"missing section: {section}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_readme_consolidation.py -v`
Expected: FAIL with at least one missing section assertion.

- [ ] **Step 3: Write minimal implementation for documentation consolidation**

```markdown
<!-- README.md section skeleton to add -->
## What This Repository Implements
- Agentic upstream impairment localization with deterministic 6-step MCP sequence.
- SLM orchestrates tools only; classifier/localizer stay deterministic and non-LLM.

## MCP Tool Chain (Fixed Sequence)
1. getRPDSpectrumMeasurements
2. analyzeSpectrumMeasurements (RPD)
3. getAllAmpsInSegment
4. getAmpSpectrumMeasurements
5. analyzeSpectrumMeasurements (amps)
6. localizeUpstreamSpectrumImpairmentSource

## Data Surfaces: Handles vs Raw
- Reference surface: handles and counts only.
- Raw surface: backend-only spectra and classification payloads.

## Assumptions And Risks
- 5-85 MHz, 256 bins, dBmV traces.
- CPD signature currently provisional and requires CableLabs confirmation.

## Open Decisions And Owners
- Topology payload shape (Randy)
- CNN class mapping and confidence semantics (Irene/Bhaskar)
- Trigger contract and deployment timeline (Randy)

## Demo Quickstart
- make install
- make test
- make demo
- optional make agent-demo with SLM_BASE_URL and SLM_MODEL
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_readme_consolidation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md CABLELABS-HANDOFF.md EXECUTION_STEPS.md docs/architecture/contract-and-data-surfaces.md docs/operations/open-decisions.md tests/test_readme_consolidation.py
git commit -m "docs: consolidate design, handoff, and execution tracker into README"
```

### Task 3: Lock MCP Contract Behavior With Regression Tests

**Files:**
- Modify: `tests/test_schema_conformance.py`
- Modify: `src/uil/mcp_server/server.py`
- Test: `tests/test_schema_conformance.py`

- [ ] **Step 1: Write failing tests for contract-critical cases**

```python
# add to tests/test_schema_conformance.py

def test_amp_measurements_partial_success_contains_failed_refs(mock_server):
    result = mock_server.get_amp_spectrum_measurements(amp_list_ref="amp-list-1")
    if result.get("status") == "partial_success":
        assert "failedCount" in result
        assert "failedDevicesRef" in result


def test_reference_outputs_never_include_raw_spectrum_arrays(mock_server):
    output = mock_server.get_rpd_spectrum_measurements(rpd_id="RPD-1", port_id="1")
    text = str(output)
    assert "maxHold" not in text
    assert "minHold" not in text
    assert "average" not in text
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_schema_conformance.py -v`
Expected: FAIL on one or both new tests.

- [ ] **Step 3: Write minimal server implementation fixes**

```python
# src/uil/mcp_server/server.py (representative patch)
if status == "partial_success":
    response["failedCount"] = len(failed_amp_ids)
    response["failedDevicesRef"] = self._store_failed_devices(failed_amp_ids)

# Ensure reference outputs never inline raw traces.
response.pop("rawSpectrum", None)
response.pop("rawSpectra", None)
```

- [ ] **Step 4: Re-run tests to verify pass**

Run: `pytest tests/test_schema_conformance.py -v`
Expected: PASS for added tests and existing conformance suite.

- [ ] **Step 5: Commit**

```bash
git add src/uil/mcp_server/server.py tests/test_schema_conformance.py
git commit -m "test: lock reference-surface contract and partial-success shape"
```

### Task 4: Harden Localizer For Topology Variants And Conflicts

**Files:**
- Modify: `tests/test_localizer.py`
- Modify: `src/uil/localizer/graph_localizer.py`
- Test: `tests/test_localizer.py`

- [ ] **Step 1: Write failing tests for flat-topology and conflicting labels**

```python
# add to tests/test_localizer.py

def test_localizer_accepts_flat_amp_list_without_parent_links(localizer):
    amps = [
        {"ampId": "A1", "distanceFromRpdMeters": 100},
        {"ampId": "A2", "distanceFromRpdMeters": 200},
    ]
    classes = [
        {"deviceType": "AMP", "ampId": "A1", "class": "Clean", "confidence": 0.9, "status": "clean"},
        {"deviceType": "AMP", "ampId": "A2", "class": "CPD", "confidence": 0.9, "status": "impaired"},
    ]
    result = localizer.localize(amps=amps, classifications=classes, impairment_type="CPD")
    assert result["localizationStatus"] in {"localized", "low_confidence"}


def test_localizer_conflicting_impairments_returns_low_confidence(localizer):
    classes = [
        {"deviceType": "AMP", "ampId": "A2", "class": "CPD", "confidence": 0.9, "status": "impaired"},
        {"deviceType": "AMP", "ampId": "A3", "class": "Ingress", "confidence": 0.9, "status": "impaired"},
    ]
    result = localizer.localize(amps=[], classifications=classes, impairment_type="CPD")
    assert result["localizationStatus"] == "low_confidence"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_localizer.py -v`
Expected: FAIL indicating topology parsing or conflict behavior mismatch.

- [ ] **Step 3: Write minimal localizer implementation updates**

```python
# src/uil/localizer/graph_localizer.py (representative patch)
def _normalize_topology(self, amps):
    has_parent = any("parentId" in a for a in amps)
    if has_parent:
        return amps
    # Fallback: derive simple chain by distance when parent links are absent.
    ordered = sorted(amps, key=lambda a: a.get("distanceFromRpdMeters", 0))
    normalized = []
    parent = None
    for amp in ordered:
        node = dict(amp)
        node.setdefault("parentId", parent)
        normalized.append(node)
        parent = node.get("ampId")
    return normalized
```

- [ ] **Step 4: Re-run tests to verify pass**

Run: `pytest tests/test_localizer.py -v`
Expected: PASS with all localizer tests green.

- [ ] **Step 5: Commit**

```bash
git add src/uil/localizer/graph_localizer.py tests/test_localizer.py
git commit -m "feat: harden localizer for flat topology and conflicting labels"
```

### Task 5: Normalize Agent Retry, Escalation, And Handoff Summary

**Files:**
- Modify: `tests/test_agent_e2e.py`
- Modify: `src/uil/agent/orchestrator.py`
- Modify: `src/uil/agent/handoff.py`
- Test: `tests/test_agent_e2e.py`

- [ ] **Step 1: Write failing E2E tests for transient retry and deterministic handoff fields**

```python
# add to tests/test_agent_e2e.py

def test_transient_measurement_unavailable_retries_once(agent_runner):
    result = agent_runner.run_scenario("rpd_measurement_unavailable_once")
    assert result["status"] in {"localized", "low_confidence", "escalated"}
    assert result["trace"].count("getRPDSpectrumMeasurements") >= 2


def test_handoff_contains_required_sections(agent_runner):
    result = agent_runner.run_scenario("conflicting_labels")
    summary = result["handoff_markdown"]
    assert "## Diagnosis Summary" in summary
    assert "## Evidence" in summary
    assert "## Recommended Next Action" in summary
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_agent_e2e.py -v`
Expected: FAIL due to retry policy or handoff formatting gaps.

- [ ] **Step 3: Write minimal orchestrator and handoff fixes**

```python
# src/uil/agent/orchestrator.py (representative patch)
TRANSIENT_ERRORS = {"STALE_DATA_ONLY", "MEASUREMENT_UNAVAILABLE"}

if error_code in TRANSIENT_ERRORS and state.retry_count < 1:
    state.retry_count += 1
    return "retry_measurement"
return "escalate"
```

```python
# src/uil/agent/handoff.py (representative patch)
return "\n".join([
    "## Diagnosis Summary",
    diagnosis_line,
    "",
    "## Evidence",
    evidence_block,
    "",
    "## Recommended Next Action",
    action_line,
])
```

- [ ] **Step 4: Re-run tests to verify pass**

Run: `pytest tests/test_agent_e2e.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/uil/agent/orchestrator.py src/uil/agent/handoff.py tests/test_agent_e2e.py
git commit -m "feat: standardize retry policy and handoff summary format"
```

### Task 6: Add CI Gates And Demo Smoke For Public Repo Readiness

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `Makefile`
- Test: CI workflow local smoke command execution

- [ ] **Step 1: Write a failing local CI parity check script**

```bash
cat > /tmp/ci_parity_check.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /home/sdp/wrkdir/scte/upstream-impairment

test -f .github/workflows/ci.yml
rg -n "pytest" .github/workflows/ci.yml
rg -n "make demo" .github/workflows/ci.yml
SH
chmod +x /tmp/ci_parity_check.sh
```

- [ ] **Step 2: Run check to verify it fails**

Run: `bash /tmp/ci_parity_check.sh`
Expected: FAIL because workflow does not exist yet.

- [ ] **Step 3: Write minimal CI workflow and make target support**

```yaml
# .github/workflows/ci.yml
name: ci

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install
        run: make install
      - name: Unit and integration tests
        run: make test
      - name: Demo smoke
        run: make demo
```

```makefile
# Makefile snippet
.PHONY: ci
ci:
	make test
	make demo
```

- [ ] **Step 4: Re-run parity check and local CI command**

Run: `bash /tmp/ci_parity_check.sh && make ci`
Expected: PASS locally with tests and demo smoke successful.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml Makefile
git commit -m "ci: add GitHub Actions gates for test and demo smoke"
```

### Task 7: Final Verification And Publish Checklist

**Files:**
- Modify: `EXECUTION_STEPS.md`
- Modify: `README.md`
- Test: full verification command set

- [ ] **Step 1: Write failing release-readiness checklist test**

```bash
cat > /tmp/release_check.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /home/sdp/wrkdir/scte/upstream-impairment

rg -n "29" README.md
rg -n "DONE" EXECUTION_STEPS.md
pytest -q
make demo >/tmp/demo_output.txt
rg -n "Scenario 1: full localization" /tmp/demo_output.txt
rg -n "Scenario 2: partial_success" /tmp/demo_output.txt
SH
chmod +x /tmp/release_check.sh
```

- [ ] **Step 2: Run checklist to verify any remaining failures**

Run: `bash /tmp/release_check.sh`
Expected: FAIL until README and execution status are fully aligned.

- [ ] **Step 3: Write minimal docs/status fixes**

```markdown
<!-- EXECUTION_STEPS.md updates -->
- Ensure each step status is accurate and justified by named tests.
- Add current pass count and command references for reproducibility.

<!-- README.md updates -->
- Align demonstrated outputs with current deterministic demo output text.
- Link to open-decisions document and architecture contract document.
```

- [ ] **Step 4: Re-run full readiness verification**

Run: `bash /tmp/release_check.sh`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md EXECUTION_STEPS.md
git commit -m "docs: finalize readiness checklist and evidence links"
```

## Self-Review

### 1. Spec coverage

- Covered fixed 6-step MCP chain and handles-only contract in Tasks 2 and 3.
- Covered topology/localizer behavior and conflict escalation in Task 4.
- Covered orchestrator off-happy-path behavior and handoff artifact in Task 5.
- Covered repo creation for Intel-Sandbox/scte-upstream-impairment in Task 1.
- Covered validation and reproducibility via CI/demo checks in Tasks 6 and 7.
- Open external dependencies (CPD signature, topology payload, CNN mapping, trigger contract) are documented and tracked in Task 2 docs outputs.

### 2. Placeholder scan

- Removed TBD/TODO wording from executable tasks.
- Every code-change step includes explicit snippets and exact commands.
- Every task includes a concrete test and pass/fail expectation.

### 3. Type consistency

- Tool names are consistent with approved sequence: getRPDSpectrumMeasurements, analyzeSpectrumMeasurements, getAllAmpsInSegment, getAmpSpectrumMeasurements, localizeUpstreamSpectrumImpairmentSource.
- Localization output keys use localizationStatus and recommendedNextAction consistently.
- Handoff section names are consistent between tests and implementation snippets.
