"""Utilities for building a minimal LangChain agent demo."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import os
from dotenv import load_dotenv


@tool
def get_current_time(timezone: Optional[str] = None) -> str:
    """Return the current time as an ISO-8601 string.

    Args:
        timezone: An optional human-readable label (e.g. "UTC" or "EST").

    Returns:
        A formatted timestamp string that includes the provided timezone label when available.
    """

    timestamp = datetime.now().isoformat(timespec="seconds")
    if timezone:
        return f"{timestamp} ({timezone})"
    return timestamp


def build_agent(model: str = "gpt-3.5-turbo", temperature: float = 0.2, api_key = None) -> AgentExecutor:
    """Build an agent executor configured with web search and a clock tool.

    Args:
        model: The chat model identifier understood by ``ChatOpenAI``.
        temperature: Softens or sharpens the model's creativity.

    Returns:
        A configured ``AgentExecutor`` ready to invoke with an ``input`` string.
    """
    load_dotenv()
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    tools = [get_current_time]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the available tools to answer "
                "questions. Use the clock tool to mention the current time when it adds "
                "helpful context.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )


def run_agent(question: str) -> str:
    """Convenience helper that executes the configured agent once."""

    executor = build_agent()
    result = executor.invoke({"input": question})
    return result["output"]


__all__ = ["build_agent", "run_agent", "get_current_time"]
