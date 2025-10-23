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

# Custom CSS for professional styling
st.markdown(
    """
<style>
    /* Main layout improvements */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .header-container {
        background: transparent;
        color: white;
        padding: 1rem 0;
        margin: 0;
        border-radius: 0;
        box-shadow: none;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
        text-align: center;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #ffffff;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Welcome section improvements */
    .welcome-container {
        background: transparent;
        padding: 2rem 0;
        margin: 1rem 0;
        text-align: center;
    }
    
    .welcome-title {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .welcome-description {
        font-size: 1.1rem;
        color: #ffffff;
        margin: 0 0 1.5rem 0;
        line-height: 1.6;
    }
    
    .welcome-examples {
        background: transparent;
        border-radius: 0;
        padding: 0;
        margin-top: 1.5rem;
        border-left: none;
    }
    
    .welcome-examples p {
        margin: 0;
        font-style: italic;
        color: #ffffff;
        font-size: 0.95rem;
    }
    
    /* Chat styling improvements */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .user-message {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 20px 20px 6px 20px;
        margin: 12px 0;
        margin-left: 15%;
        box-shadow: 0 4px 12px rgba(0,123,255,0.2);
        word-wrap: break-word;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .assistant-message {
        background: #ffffff;
        color: #2c3e50;
        padding: 16px 20px;
        border-radius: 20px 20px 20px 6px;
        margin: 12px 0;
        margin-right: 15%;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        word-wrap: break-word;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .message-time {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 6px;
        font-weight: 500;
    }
    
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: transparent;
        padding: 20px;
        border-top: none;
        box-shadow: none;
        backdrop-filter: none;
    }
    
    .main-container {
        padding-bottom: 0;
    }
    
    /* Sidebar improvements */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        transition: border-color 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 3px rgba(0,123,255,0.1);
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
    # Master Orchestrator section at the top
    st.markdown("### 🧭 Master Orchestrator")
    st.markdown("Routes queries to specialized supervisors:")
    st.markdown("• **Research** - Boston data & web research")
    st.markdown("• **Documentation** - Process maps & docs")
    st.markdown("• **Frameworks** - Structured approaches")

    st.divider()

    # LLM Configuration in expander
    with st.expander("⚙️ LLM Configuration", expanded=False):
        st.markdown("**Chat Model**")
        selected_model = st.selectbox(
            "Select model",
            options=["gpt-4o-mini"],
            index=0,
            label_visibility="collapsed",
        )

        selected_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
        )
        st.caption(
            "The agent requires an `OPENAI_API_KEY` environment variable to be set."
        )

    st.divider()

    # Clear conversation button with better styling
    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("🗑️ Clear", type="secondary", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main chat interface
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown(
    """
<div class="header-container">
    <h1 class="header-title">🧭 Master Orchestrator</h1>
    <p class="header-subtitle">Your intelligent assistant for research, documentation, and frameworks</p>
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
    <div class="welcome-container">
        <h2 class="welcome-title">👋 Welcome to Master Orchestrator</h2>
        <p class="welcome-description">Ask me anything! I can help with research, documentation, and framework suggestions.</p>
        <div class="welcome-examples">
            <p><strong>Try asking:</strong> "What data is available about Boston housing?" or "Create a process map for user onboarding"</p>
        </div>
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
