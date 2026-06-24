PY := /home/sdp/vllm-env/bin/python

.PHONY: test demo agent-demo install install-agent lint

install:
	$(PY) -m pip install -e .[dev]

install-agent:
	$(PY) -m pip install -e .[dev,agent]

test:
	$(PY) -m pytest -q

demo:
	$(PY) examples/demo.py

# Real LangGraph + SLM agent. Requires SLM_BASE_URL / SLM_MODEL to point at an
# OpenAI-compatible endpoint (local llama.cpp / vLLM / Ollama shim, or hosted).
agent-demo:
	PYTHONPATH=src $(PY) -m uil.agent.langgraph_agent
