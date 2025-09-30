# Launchpad Agent Workbench

This folder contains a self-contained mockup of an agentic system built for quick demos.
Two specialised agents share a consistent Streamlit front end:

- **Process Mapping Agent** – turns a problem statement into a staged playbook and a Mermaid
  diagram that can be pasted into documentation or whiteboarding tools.
- **Research Launch Agent** – queries the NYC OpenData catalog live (with deterministic
  fallbacks) and returns a narrative brief plus dataset recommendations keyed to the
  initiative's constraints.

No external LLM is required, but the research tab performs HTTPS calls to the OpenData
portal. The app falls back to a small curated catalogue if the network is unavailable.

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
├── agent.py         # Deterministic agent logic (process mapping + research launch)
├── app.py           # Streamlit UI with tabbed agents
├── cli_demo.py      # Command line entry points mirroring the UI flows
├── requirements.txt # Streamlit UI + requests for OpenData calls
└── README.md        # You are here
```

The process agent is entirely deterministic; the research agent layers deterministic
scoring atop live NYC OpenData catalog results with a graceful offline fallback.
