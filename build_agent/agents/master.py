from __future__ import annotations

from typing import Any, Dict, Optional

from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from build_agent.supervisors.research import build_research_supervisor
from build_agent.supervisors.documentation import build_documentation_supervisor
from build_agent.supervisors.frameworks import build_frameworks_supervisor
from build_agent.agents.context import get_chat_history


# Build supervisors once per process; agents are lightweight but we reuse for speed.
_supervisors: Dict[str, AgentExecutor] = {}


def _get_research(model: str, temp: float, api_key: Optional[str]) -> AgentExecutor:
    key = f"research:{model}:{temp}"
    if key not in _supervisors:
        _supervisors[key] = build_research_supervisor(
            model=model, temperature=temp, api_key=api_key
        )
    return _supervisors[key]


def _get_docs(model: str, temp: float, api_key: Optional[str]) -> AgentExecutor:
    key = f"docs:{model}:{temp}"
    if key not in _supervisors:
        _supervisors[key] = build_documentation_supervisor(
            model=model, temperature=temp, api_key=api_key
        )
    return _supervisors[key]


def _get_frameworks(model: str, temp: float, api_key: Optional[str]) -> AgentExecutor:
    key = f"frameworks:{model}:{temp}"
    if key not in _supervisors:
        _supervisors[key] = build_frameworks_supervisor(
            model=model, temperature=temp, api_key=api_key
        )
    return _supervisors[key]


@tool("delegate_research")
def delegate_research(
    input: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> str:
    """Route a task to the Research Supervisor. Input is the user's query."""
    exec_ = _get_research(model, temperature, api_key)
    res: Dict[str, Any] = exec_.invoke(
        {"input": input, "chat_history": get_chat_history()}
    )
    return res.get("output", "")


@tool("delegate_documentation")
def delegate_documentation(
    input: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> str:
    """Route a task to the Documentation Supervisor. Input is the user's query."""
    exec_ = _get_docs(model, temperature, api_key)
    res: Dict[str, Any] = exec_.invoke(
        {"input": input, "chat_history": get_chat_history()}
    )
    return res.get("output", "")


@tool("delegate_frameworks")
def delegate_frameworks(
    input: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> str:
    """Route a task to the Frameworks Supervisor. Input is the user's query."""
    exec_ = _get_frameworks(model, temperature, api_key)
    res: Dict[str, Any] = exec_.invoke(
        {"input": input, "chat_history": get_chat_history()}
    )
    return res.get("output", "")


SYSTEM = (
    "You are the Master Orchestrator. Decide which supervisor(s) to engage: "
    "Research (Boston data or general web research), Documentation (process maps, docs), "
    "Frameworks (structured approaches). Call exactly the appropriate delegate tool with "
    "a well-scoped instruction. If multiple supervisors are helpful, call them in sequence "
    "and compose a final answer."
)


def build_master_agent(
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> AgentExecutor:
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    tools = [
        delegate_research,
        delegate_documentation,
        delegate_frameworks,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
    )


__all__ = [
    "build_master_agent",
    "delegate_research",
    "delegate_documentation",
    "delegate_frameworks",
]
