"""Streamlit chatbot UI for the LangChain agent demo."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import streamlit as st
from datetime import datetime

from build_agent.agents.master import build_master_agent
from build_agent.agents.context import set_chat_history
from langchain_core.messages import HumanMessage, AIMessage
import os, dotenv

# OpenAI Key
dotenv.load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Master Orchestrator Chat",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for chat bubbles
st.markdown(
    """
<style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        margin-left: 20%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    
    .assistant-message {
        background: #f8f9fa;
        color: #333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        margin-right: 20%;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        word-wrap: break-word;
    }
    
    .message-time {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 4px;
    }
    
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 20px;
        border-top: 1px solid #e9ecef;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    .main-container {
        padding-bottom: 120px;
    }
    
    .welcome-message {
        text-align: center;
        padding: 40px 20px;
        color: #666;
    }
    
    .welcome-message h2 {
        color: #333;
        margin-bottom: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


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


def render_message_bubble(message, is_user: bool, timestamp: str = None):
    """Render a message bubble with proper styling."""
    if is_user:
        st.markdown(
            f"""
        <div class="user-message">
            {message}
            <div class="message-time">{timestamp}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="assistant-message">
            {message}
            <div class="message-time">{timestamp}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.text_input("Chat model", value="gpt-4o-mini")
    selected_temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
    )
    st.caption("The agent requires an `OPENAI_API_KEY` environment variable to be set.")

    st.divider()

    if st.button("🗑️ Clear conversation", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.markdown("### 🧭 Master Orchestrator")
    st.markdown("Routes queries to specialized supervisors:")
    st.markdown("• **Research** - Boston data & web research")
    st.markdown("• **Documentation** - Process maps & docs")
    st.markdown("• **Frameworks** - Structured approaches")


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main chat interface
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown(
    """
<div style="text-align: center; padding: 20px 0; border-bottom: 1px solid #e9ecef; margin-bottom: 20px;">
    <h1 style="margin: 0; color: #333;">🧭 Master Orchestrator</h1>
    <p style="margin: 5px 0 0 0; color: #666;">Your intelligent assistant for research, documentation, and frameworks</p>
</div>
""",
    unsafe_allow_html=True,
)

# Chat messages
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        is_user = isinstance(msg, HumanMessage)
        timestamp = datetime.now().strftime("%H:%M")
        render_message_bubble(msg.content, is_user, timestamp)
else:
    st.markdown(
        """
    <div class="welcome-message">
        <h2>👋 Welcome to Master Orchestrator</h2>
        <p>Ask me anything! I can help with research, documentation, and framework suggestions.</p>
        <p><em>Try asking: "What data is available about Boston housing?" or "Create a process map for user onboarding"</em></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# Chat input
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])

with col1:
    prompt = st.text_input(
        "Type your message here...",
        key="chat_input",
        placeholder="Ask me anything about research, documentation, or frameworks...",
        label_visibility="collapsed",
    )

with col2:
    send_clicked = st.button("Send", type="primary", use_container_width=True)

# Handle message sending
if send_clicked and prompt:
    with st.spinner("🤔 Thinking..."):
        # Append user message
        user_msg = HumanMessage(prompt)
        st.session_state.chat_history.append(user_msg)
        set_chat_history(st.session_state.chat_history)

        # Get agent response
        agent = get_agent(selected_model, selected_temperature)
        result: Dict[str, Any] = agent.invoke(
            {
                "input": prompt,
                "chat_history": st.session_state.chat_history,
            }
        )

    # Append assistant response
    st.session_state.chat_history.append(AIMessage(result["output"]))

    # Rerun to show new messages (input will be cleared automatically)
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
