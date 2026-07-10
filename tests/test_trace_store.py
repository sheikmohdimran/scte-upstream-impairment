"""Audit trace persistence tests (regulatory requirement).

Verifies that every orchestration run is durably recorded to an append-only JSONL audit log,
including the alarm-trigger decision, and that the raw-data exclusion holds in the persisted
record.
"""

import json
import os
from pathlib import Path

import pytest

from uil.agent.orchestrator import Orchestrator
from uil.agent.trace_store import TraceStore
from uil.domain.labels import ImpairmentLabel
from uil.mcp_server.server import FaultInjection, MockMcpServer, Scenario


@pytest.fixture(autouse=True)
def _enable_persistence(monkeypatch):
    # conftest disables persistence globally; re-enable it for these tests.
    monkeypatch.delenv("UIL_TRACE_DISABLE", raising=False)


def _scenario(**faults) -> Scenario:
    return Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},
            {"ampId": "A3", "parentId": "A2", "label": "CPD"},
            {"ampId": "A4", "parentId": "A1", "label": "Clean"},
        ],
        faults=FaultInjection(**faults),
    )


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_successful_run_is_recorded(tmp_path):
    store = TraceStore(tmp_path)
    res = Orchestrator(MockMcpServer(_scenario()), "audit-happy", trace_store=store).run()
    assert res.status == "localized"

    recs = _records(store.path)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["runId"] and rec["recordedUtc"]
    assert rec["scenario"] == "audit-happy"
    assert rec["finalStatus"] == "localized"
    assert rec["toolCallCount"] == 6
    assert rec["trace"]["calls"][0]["tool"] == "getRPDSpectrumMeasurements"


def test_records_are_appended_not_overwritten(tmp_path):
    store = TraceStore(tmp_path)
    Orchestrator(MockMcpServer(_scenario()), "run-1", trace_store=store).run()
    Orchestrator(MockMcpServer(_scenario(rpd_unreachable=True)), "run-2", trace_store=store).run()

    recs = _records(store.path)
    assert len(recs) == 2
    assert recs[0]["scenario"] == "run-1" and recs[0]["finalStatus"] == "localized"
    assert recs[1]["scenario"] == "run-2" and recs[1]["finalStatus"] == "failed"


def test_alarm_entry_records_trigger_decision(tmp_path):
    store = TraceStore(tmp_path)
    orch = Orchestrator(MockMcpServer(_scenario()), "audit-alarm", trace_store=store)
    alarm = {
        "alarmId": "ALM-77", "alarmType": "highUpstreamFecErrors", "direction": "upstream",
        "entity": {"type": "rpdPort", "rpdId": "RPD-9", "portId": "US4"},
    }
    orch.run_from_alarm(alarm)

    rec = _records(store.path)[0]
    assert rec["trigger"]["verdict"] == "call"
    assert rec["trigger"]["alarmId"] == "ALM-77"


def test_negative_alarm_is_recorded_with_no_tool_calls(tmp_path):
    store = TraceStore(tmp_path)
    orch = Orchestrator(MockMcpServer(_scenario()), "audit-neg", trace_store=store)
    alarm = {
        "alarmId": "ALM-78", "alarmType": "highCmTransmitPower", "direction": "upstream",
        "entity": {"type": "cableModem", "cmMacAddress": "00:15:96:a1:b2:c3"},
    }
    orch.run_from_alarm(alarm)

    rec = _records(store.path)[0]
    assert rec["finalStatus"] == "not_triggered"
    assert rec["toolCallCount"] == 0
    assert rec["trigger"]["verdict"] == "noCall"


def test_persisted_record_excludes_raw_spectra(tmp_path):
    store = TraceStore(tmp_path)
    Orchestrator(MockMcpServer(_scenario()), "audit-raw", trace_store=store).run()
    text = store.path.read_text(encoding="utf-8")
    assert "maxHold" not in text and "average" not in text


def test_disable_env_suppresses_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("UIL_TRACE_DISABLE", "1")
    store = TraceStore(tmp_path)
    Orchestrator(MockMcpServer(_scenario()), "audit-disabled", trace_store=store).run()
    assert not store.path.exists()


def test_hash_chain_verifies_across_runs(tmp_path):
    store = TraceStore(tmp_path)
    Orchestrator(MockMcpServer(_scenario()), "chain-1", trace_store=store).run()
    Orchestrator(MockMcpServer(_scenario(rpd_unreachable=True)), "chain-2", trace_store=store).run()
    Orchestrator(MockMcpServer(_scenario()), "chain-3", trace_store=store).run()

    ok, bad = store.verify()
    assert ok and bad is None

    recs = _records(store.path)
    # First record links to the genesis hash; each subsequent links to the prior recordHash.
    assert recs[0]["prevHash"] == "0" * 64
    assert recs[1]["prevHash"] == recs[0]["recordHash"]
    assert recs[2]["prevHash"] == recs[1]["recordHash"]


def test_tampering_is_detected(tmp_path):
    store = TraceStore(tmp_path)
    Orchestrator(MockMcpServer(_scenario()), "tamper-1", trace_store=store).run()
    Orchestrator(MockMcpServer(_scenario()), "tamper-2", trace_store=store).run()
    assert store.verify()[0] is True

    # Retroactively edit the first record's content without fixing its hash.
    lines = store.path.read_text(encoding="utf-8").splitlines()
    rec0 = json.loads(lines[0])
    rec0["finalStatus"] = "tampered"
    lines[0] = json.dumps(rec0, ensure_ascii=False)
    store.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, bad = store.verify()
    assert ok is False and bad == 0


def test_daily_rotation_filename(tmp_path):
    from datetime import datetime, timezone

    store = TraceStore(tmp_path)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert store.path.name == f"tool_call_traces-{day}.jsonl"

