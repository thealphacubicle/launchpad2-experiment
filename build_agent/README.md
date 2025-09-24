# Build Agent Demo

This folder contains a minimal LangChain agent that combines a web search tool and a
simple local utility (a clock) to answer user questions. The agent can be exercised
from the command line or through a lightweight Streamlit user interface.

## Features

- **One-step web search:** Uses the DuckDuckGo search API through LangChain's community
tools to retrieve fresh information from the web.
- **Local utility tool:** Provides the current local time so the agent can ground its
responses with simple contextual data.
- **Reusable agent factory:** A single helper builds the agent executor so both the CLI
and UI demos stay in sync.
- **Streamlit UI mockup:** A minimal front end that shows how the agent could be embedded
in an application.

## Requirements

Install the dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

The agent uses OpenAI's API for language reasoning. Set your API key before running any
of the demos:

```bash
export OPENAI_API_KEY="sk-..."
```

## Running the CLI demo

The command line interface runs a single prompt and prints the agent's final answer and
intermediate tool usage.

```bash
python -m build_agent.cli_demo "What happened in tech news today?"
```

## Running the Streamlit mock UI

Launch the UI locally:

```bash
streamlit run build_agent/app.py
```

Enter a question in the textbox and click **Run agent** to see the reasoning trace and the
final response.

## Project layout

```
build_agent/
├── agent.py           # Reusable agent factory
├── cli_demo.py        # Command line entry-point
├── streamlit_app.py   # Streamlit mock UI
├── requirements.txt   # Tooling and runtime dependencies
└── README.md
```

The code is intentionally small and heavily commented to illustrate how an agent, tools,
and interfaces fit together.
