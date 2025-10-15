# Build Agent Demo (Revamped)

This demo now uses a Master Orchestrator agent that delegates to three supervisors:

- Research Supervisor – web search and Boston OpenData MCP tool
- Documentation Supervisor – Process map generator tool (Mermaid)
- Frameworks Supervisor – Lightweight suggestions for structures/templates

## Requirements

Install dependencies:

```bash
pip install -r build_agent/requirements.txt
```

Set your OpenAI key:

```bash
export OPENAI_API_KEY="sk-..."
```

Additional packages added for MCP support:

- `mcp>=1.2`
- `httpx>=0.27`

## Run

CLI:

```bash
python -m build_agent.cli_demo "How many 311 datasets exist on Boston Open Data?"
```

Streamlit UI:

```bash
streamlit run build_agent/app.py
```

## What changed

- Bundled the Boston OpenData MCP server under `build_agent/mcp_servers/boston_opendata`.
- Added `boston_opendata` LangChain tool that spawns the server and calls MCP actions.
- Added `generate_process_map` tool for Mermaid diagrams.
- Created supervisors under `build_agent/supervisors/` and a Master Orchestrator under
  `build_agent/agents/master.py`.
