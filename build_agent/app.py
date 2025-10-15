"""Streamlit mock UI for the LangChain agent demo."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import streamlit as st

from build_agent.agents.master import build_master_agent
from build_agent.agents.context import set_chat_history
from langchain_core.messages import HumanMessage, AIMessage
import os, dotenv

# OpenAI Key
dotenv.load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Master Orchestrator Demo", page_icon="🧭", layout="centered"
)
st.title("🧭 Master Orchestrator Agent")
st.write("Routes queries to Research, Documentation, or Frameworks supervisors.")


@st.cache_resource(show_spinner=False)
def get_agent(model: str, temperature: float):
    """Return a cached agent executor so repeated runs are fast."""

    return build_master_agent(model=model, temperature=temperature, api_key=API_KEY)


def format_intermediate_steps(steps: List[Tuple[Any, Any]]) -> List[str]:
    """Render the agent's intermediate steps for display."""

    formatted = []
    for action, observation in steps:
        tool_name = getattr(action, "tool", "tool")
        tool_input = str(getattr(action, "tool_input", getattr(action, "input", "")))
        observation_text = str(observation)
        formatted.append(
            f"**{tool_name}** ← `{tool_input}`\n" f"Observation: {observation_text}"
        )
    return formatted


with st.sidebar:
    st.header("Configuration")
    selected_model = st.text_input("Chat model", value="gpt-3.5-turbo")
    selected_temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
    )
    st.caption("The agent requires an `OPENAI_API_KEY` environment variable to be set.")
    if st.button("Reset conversation"):
        st.session_state.chat_history = []

prompt = st.text_area("What would you like to know?", height=120)
run_clicked = st.button("Run agent", type="primary")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # type: ignore[var-annotated]

# Render previous turns
if st.session_state.chat_history:
    st.subheader("Conversation")
    for msg in st.session_state.chat_history:
        role = "You" if isinstance(msg, HumanMessage) else "Assistant"
        st.markdown(f"**{role}:** {msg.content}")

if run_clicked and prompt:
    with st.spinner("Thinking..."):
        # Append user turn and set shared context
        st.session_state.chat_history.append(HumanMessage(prompt))
        set_chat_history(st.session_state.chat_history)

        agent = get_agent(selected_model, selected_temperature)
        result: Dict[str, Any] = agent.invoke({
            "input": prompt,
            "chat_history": st.session_state.chat_history,
        })

    # Append assistant turn
    st.session_state.chat_history.append(AIMessage(result["output"]))

    st.subheader("Final answer")
    st.write(result["output"])

    intermediate_steps = result.get("intermediate_steps", [])
    if intermediate_steps:
        st.subheader("Tool trace")
        for idx, step in enumerate(
            format_intermediate_steps(intermediate_steps), start=1
        ):
            st.markdown(f"{idx}. {step}")
else:
    st.info("Enter a question above and click **Run agent** to see it in action.")
