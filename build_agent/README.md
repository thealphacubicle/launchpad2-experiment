# Launchpad Agent Workbench

This folder contains a self-contained mockup of an agentic system built for quick demos.
Two specialised agents share a consistent Streamlit front end:

- **Process Mapping Agent** – orchestrates multiple LLM-powered workers to turn a problem
  statement into clarifying questions, a staged playbook, and a Mermaid diagram that can be
  pasted into documentation or whiteboarding tools.
- **Research Launch Agent** – queries the NYC OpenData catalog live (with deterministic
  fallbacks) and returns a narrative brief plus dataset recommendations keyed to the
  initiative's constraints.

The process mapper now relies on an OpenAI-compatible LLM. Set `OPENAI_API_KEY` and install
the `openai` Python package (listed in `requirements.txt`). The research tab still performs
HTTPS calls to the OpenData portal and falls back to a small curated catalogue if the
network is unavailable.

## Getting started

Create a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Streamlit experience

Run the mock console locally:

```bash
streamlit run build_agent/app.py
```

- Use the *Process Mapping* tab to paste a problem statement, pick a detail level, and copy
  the generated Mermaid diagram into your planning docs.
- Switch to *Research Launch* to provide constraints (budget ceiling, scale) and gather a
  synthesised launch brief with the top datasets to explore first. Live searches may take a
  moment while the catalog request completes.

## Command line helpers

For a minimal CLI experience, invoke:

```bash
python -m build_agent.cli_demo process "Reduce onboarding time for support engineers"
python -m build_agent.cli_demo research "Revive downtown retail foot traffic" --budget 200000 --scale City
```

The process command prints the diagram and supporting assumptions, while the research
command surfaces the dataset picks and next-step checklist.

## Project layout

```
build_agent/
├── agent.py         # LLM-backed process mapper + research launch helpers
├── agents/          # Context getter, drawer, and reasoning LLM workers
├── app.py           # Streamlit UI with tabbed agents
├── cli_demo.py      # Command line entry points mirroring the UI flows
├── llm_client.py    # Lightweight OpenAI chat wrapper + JSON extraction helper
├── requirements.txt # Streamlit UI + requests + OpenAI client
└── README.md        # You are here
```

The process agent now orchestrates three LLM-guided workers (context getter, drawer, and
reasoning) while the research agent layers deterministic scoring atop live NYC OpenData
catalog results with a graceful offline fallback.
