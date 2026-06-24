"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uil.domain.labels import ImpairmentLabel  # noqa: E402
from uil.mcp_server.server import FaultInjection, MockMcpServer, Scenario  # noqa: E402


@pytest.fixture
def cpd_scenario() -> Scenario:
    return Scenario(
        rpdId="RPD-1", portId="P1", rpd_label=ImpairmentLabel.CPD,
        amps=[
            {"ampId": "A1", "parentId": None, "label": "Clean"},
            {"ampId": "A2", "parentId": "A1", "label": "CPD"},
            {"ampId": "A3", "parentId": "A2", "label": "CPD"},
            {"ampId": "A4", "parentId": "A1", "label": "Clean"},
        ],
    )


@pytest.fixture
def server(cpd_scenario: Scenario) -> MockMcpServer:
    return MockMcpServer(cpd_scenario)
