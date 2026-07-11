PY ?= python3

.PHONY: test demo agent-demo install install-agent lint ci

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

# Grade the live SLM on generated scenarios with known ground truth.
# Requires the same SLM_BASE_URL / SLM_MODEL / OPENAI_API_KEY env as agent-demo.
eval-slm:
	PYTHONPATH=src $(PY) examples/eval_slm.py --per-dimension $(or $(PER),2) --seed $(or $(SEED),0)

ci:
	$(MAKE) test
	$(MAKE) demo
