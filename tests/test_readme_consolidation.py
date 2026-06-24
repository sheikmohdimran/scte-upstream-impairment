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
