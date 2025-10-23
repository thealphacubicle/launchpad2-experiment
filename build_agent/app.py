"""Streamlit chatbot UI for the LangChain agent demo."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import streamlit as st
from datetime import datetime
import re
import time
from streamlit_mermaid import st_mermaid

from build_agent.agents.master import build_master_agent
from build_agent.agents.context import set_chat_history
from langchain_core.messages import HumanMessage, AIMessage
import os, dotenv

# OpenAI Key
dotenv.load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Launchpad 2.0 Slim",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state early
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# Mermaid.js is now handled by streamlit-mermaid component

# Custom CSS for professional styling
st.markdown(
    """
<style>
    /* Main layout improvements */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1800px;
    }
    
    /* Mermaid diagram styling - clean and simple */
    .stMermaid {
        width: 100% !important;
        max-width: none !important;
    }
    
    /* Ensure diagram containers are wide */
    .stMermaid > div {
        width: 100% !important;
        min-width: 1000px !important;
    }
    
    /* Make the diagram itself larger */
    .stMermaid svg {
        width: 100% !important;
        height: auto !important;
        min-width: 1000px !important;
    }
    
    /* Ensure chat messages don't constrain diagram width */
    .assistant-message {
        max-width: none !important;
        width: 100% !important;
    }
    
    /* Aggressively remove ALL vertical spacing around diagrams */
    .stMermaid {
        margin: 0 !important;
        padding: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        height: auto !important;
    }
    
    /* Remove ALL spacing in containers */
    .stContainer {
        width: 100% !important;
        max-width: none !important;
        margin: 0 !important;
        padding: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove ALL spacing from Streamlit blocks */
    .stMermaid + div,
    .stMermaid + * {
        margin: 0 !important;
        padding: 0 !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Remove ALL bottom spacing from diagram containers */
    .stMermaid .stContainer,
    .stMermaid > div,
    .stMermaid > div > div {
        margin: 0 !important;
        padding: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Force diagram to be compact */
    .stMermaid svg {
        margin: 0 !important;
        padding: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove Streamlit block spacing */
    .stMermaid,
    .stMermaid + .element-container,
    .stMermaid + .stBlock {
        margin: 0 !important;
        padding: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Target any remaining Streamlit spacing */
    div[data-testid="stMermaid"] {
        margin: 0 !important;
        padding: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
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
    
    /* Streaming response styling */
    .streaming-response {
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
        position: relative;
        min-height: 50px;
    }
    
    /* Enhanced streaming cursor animation */
    .streaming-cursor {
        animation: blink 1.2s infinite;
        color: #007bff;
        font-weight: bold;
        font-size: 1.1em;
        margin-left: 2px;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    /* Typing indicator styling */
    .typing-indicator {
        background: #f8f9fa;
        color: #6c757d;
        padding: 12px 20px;
        border-radius: 20px 20px 20px 6px;
        margin: 12px 0;
        margin-right: 15%;
        border: 1px solid #e9ecef;
        font-size: 0.9rem;
        font-style: italic;
        display: flex;
        align-items: center;
        gap: 8px;
        animation: pulse 2s infinite;
    }
    
    /* Animated typing dots */
    .typing-dots span {
        animation: typingDots 1.4s infinite;
        animation-delay: calc(var(--i) * 0.2s);
    }
    
    .typing-dots span:nth-child(1) { --i: 0; }
    .typing-dots span:nth-child(2) { --i: 1; }
    .typing-dots span:nth-child(3) { --i: 2; }
    
    @keyframes typingDots {
        0%, 60%, 100% { opacity: 0.3; }
        30% { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    
    /* Enhanced streaming animation for the entire response */
    .streaming-response {
        animation: fadeInUp 0.3s ease-out;
    }
    
    /* Completed response styling */
    .streaming-response.completed {
        animation: fadeInUp 0.3s ease-out, completionPulse 0.5s ease-out 0.2s;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes completionPulse {
        0% { 
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        50% { 
            box-shadow: 0 4px 20px rgba(0,123,255,0.2);
        }
        100% { 
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
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


def parse_mermaid_content(content: str) -> Tuple[str, List[str]]:
    """Parse content to extract Mermaid diagrams and return text with diagrams replaced."""
    # Pattern to match mermaid code blocks - handle various formatting
    pattern = r"```mermaid\s*\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)

    # Replace mermaid blocks with placeholders
    text_content = re.sub(pattern, "MERMAID_PLACEHOLDER", content, flags=re.DOTALL)

    return text_content, matches


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
        # Parse content for Mermaid diagrams
        text_content, mermaid_diagrams = parse_mermaid_content(message)

        # If we have Mermaid diagrams, render them inline with better sizing
        if mermaid_diagrams:
            # Split content by placeholders and render accordingly
            parts = text_content.split("MERMAID_PLACEHOLDER")
            diagram_index = 0

            # Start the message container
            st.markdown('<div class="assistant-message">', unsafe_allow_html=True)

            # Render each part
            for i, part in enumerate(parts):
                if part.strip():  # Only render non-empty text parts
                    st.markdown(part, unsafe_allow_html=True)

                # If this isn't the last part, render a Mermaid diagram
                if i < len(parts) - 1 and diagram_index < len(mermaid_diagrams):
                    diagram_code = mermaid_diagrams[diagram_index].strip()

                    # Use streamlit-mermaid component to render the diagram with better sizing
                    st.markdown("**📊 Process Diagram:**")

                    # Create a container with better width for the diagram
                    with st.container():
                        st_mermaid(diagram_code, height=300)

                    diagram_index += 1

            # Close the message container and add timestamp
            st.markdown(
                f'<div class="message-time">{timestamp}</div>', unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # No Mermaid diagrams, render normally
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
    # Launchpad 2.0 Slim section at the top
    st.markdown("### 🚀 Launchpad 2.0 Slim")
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


# Session state already initialized at the top

# Main chat interface
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown(
    """
<div class="header-container">
    <h1 class="header-title">🚀 Launchpad 2.0 Slim</h1>
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
        <h2 class="welcome-title">👋 Welcome to Launchpad 2.0 Slim</h2>
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
    # Use a dynamic key that changes when we want to clear
    input_key = f"chat_input_{st.session_state.clear_input}"
    prompt = st.text_input(
        "Type your message here...",
        key=input_key,
        placeholder="Ask me anything about research, documentation, or frameworks...",
        label_visibility="collapsed",
    )

with col2:
    send_clicked = st.button("Send", type="primary", use_container_width=True)

# Handle message sending
if send_clicked and prompt:
    # Store the prompt before clearing
    user_prompt = prompt

    # Append user message immediately
    user_msg = HumanMessage(user_prompt)
    st.session_state.chat_history.append(user_msg)
    set_chat_history(st.session_state.chat_history)

    # Toggle the clear_input flag to clear the input field
    st.session_state.clear_input = not st.session_state.clear_input

    # Get agent response with streaming
    agent = get_agent(selected_model, selected_temperature)

    # Create a placeholder for the streaming response
    response_placeholder = st.empty()
    status_placeholder = st.empty()
    full_response = ""
    current_display = ""

    # Show initial typing indicator with progress dots
    status_placeholder.markdown(
        """
        <div class="typing-indicator">
            🤖 Assistant is thinking
            <span class="typing-dots">
                <span>.</span><span>.</span><span>.</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stream the response with enhanced visual feedback
    for chunk in agent.stream(
        {
            "input": user_prompt,
            "chat_history": st.session_state.chat_history,
        }
    ):
        if "output" in chunk:
            new_content = chunk["output"]
            full_response += new_content

            # Clear status indicator once we start getting content
            if new_content.strip():
                status_placeholder.empty()

            # Implement word-by-word streaming for more natural feel
            words = new_content.split(" ")
            for i, word in enumerate(words):
                if i > 0:  # Add space before each word except the first
                    current_display += " "
                current_display += word

                # Update the placeholder with the current response and streaming cursor
                response_placeholder.markdown(
                    f'<div class="streaming-response">{current_display}<span class="streaming-cursor">▋</span></div>',
                    unsafe_allow_html=True,
                )

                # Variable delay based on word characteristics for more realistic typing
                if word.endswith((".", "!", "?", ";", ":")):
                    time.sleep(0.15)  # Longer pause after sentences
                elif word.endswith(","):
                    time.sleep(0.08)  # Medium pause after commas
                elif "\n" in word:
                    time.sleep(0.2)  # Longer pause for line breaks
                else:
                    time.sleep(0.05)  # Normal word typing speed

    # Remove streaming cursor and show final response
    response_placeholder.markdown(
        f'<div class="streaming-response completed">{current_display}</div>',
        unsafe_allow_html=True,
    )

    # Brief pause to show the final response before adding to history
    time.sleep(0.5)

    # Append the complete assistant response to chat history
    st.session_state.chat_history.append(AIMessage(full_response))

    # Rerun to show all new messages
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
