"""Streamlit chat UI for Enterprise Knowledge Search Assistant with High-Contrast Pure Black Text."""

import re
import sys
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ---------------------------------------------------------------------------
# Cache: load embedding model and Chroma collection ONCE at startup.
# This triggers the module-level caches in search_vector and search_keyword
# without making any LLM call.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge base...")
def _warmup():
    """Pre-load the sentence-transformers model and Chroma collection into cache."""
    from src.search_vector import get_embedding_model, get_chroma_client
    from src.search_keyword import _load_index
    get_embedding_model()
    get_chroma_client()
    _load_index()
    return True


_warmup()

# Import answer_question only after warmup so config loads cleanly
from src.answer import answer_question  # noqa: E402


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Search Assistant",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# High-Contrast CSS: ALL TEXT #000000 (Pure Black), Unclipped Responsive Header
# ---------------------------------------------------------------------------

NEUMORPHIC_CSS = """
<style>
/* 1. Global & Inherited Text Rules: Everything Pure Black #000000 */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: #F9EAD7 !important;
    color: #000000 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Force all standard text elements to pure black */
p, span, div, h1, h2, h3, h4, h5, h6, label, input, textarea, button, a, li, summary, b, strong, i, em, code {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Hyperlinks and email addresses */
a {
    color: #000000 !important;
    text-decoration: underline !important;
    font-weight: 600 !important;
}

/* Remove awkward container constraints */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 6.5rem !important;
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Sidebar styling */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: #F9EAD7 !important;
    border-right: 1px solid rgba(214, 194, 170, 0.7) !important;
}

[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
    color: #000000 !important;
    font-weight: 700 !important;
}

/* 2. Unclipped, Clean, Responsive Header */
.header-wrapper {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 12px 18px;
    padding: 10px 0 16px 0;
    margin-bottom: 1.2rem;
    border-bottom: 2px solid rgba(209, 192, 172, 0.6);
    width: 100%;
}

.header-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: #000000 !important;
    margin: 0;
    padding: 0;
    letter-spacing: -0.5px;
    white-space: nowrap;
}

.header-subtitle {
    font-size: 0.95rem;
    font-weight: 600;
    color: #000000 !important;
    background-color: rgba(191, 226, 58, 0.45);
    padding: 3px 10px;
    border-radius: 6px;
    border: 1px solid rgba(160, 185, 45, 0.5);
    white-space: nowrap;
}

/* Responsive wrap on small screens */
@media (max-width: 768px) {
    .header-title {
        white-space: normal;
        font-size: 1.35rem;
    }
    .header-subtitle {
        white-space: normal;
    }
}

/* Welcome Area */
.welcome-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
}

.welcome-title {
    font-size: 2rem;
    font-weight: 800;
    color: #000000 !important;
    margin-bottom: 0.5rem;
}

.welcome-subtitle {
    font-size: 1.05rem;
    font-weight: 500;
    color: #000000 !important;
    margin-bottom: 2rem;
}

/* Suggestion Cards */
div[data-testid="stColumn"] .stButton > button {
    height: 100% !important;
    min-height: 75px !important;
    background-color: #F9EAD7 !important;
    color: #000000 !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    border-radius: 14px !important;
    padding: 12px 14px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    text-align: left !important;
    box-shadow: 4px 4px 10px #dfccb5, -4px -4px 10px #ffffff !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stColumn"] .stButton > button:hover {
    border-color: #BFE23A !important;
    box-shadow: 2px 2px 5px #dfccb5, -2px -2px 5px #ffffff !important;
    color: #000000 !important;
}

/* Sidebar sample question buttons */
[data-testid="stSidebar"] .stButton > button {
    background-color: #F9EAD7 !important;
    color: #000000 !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    border-radius: 12px !important;
    padding: 10px 12px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-align: left !important;
    margin-bottom: 6px !important;
    box-shadow: 3px 3px 8px #dfccb5, -3px -3px 8px #ffffff !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    color: #000000 !important;
    border-color: #BFE23A !important;
}

[data-testid="stSidebar"] .stButton > button * {
    color: #000000 !important;
}

/* Clear chat button: Subtle crimson border/accent but black text */
.clear-chat-btn button {
    border-color: #E10600 !important;
    font-weight: 700 !important;
    color: #000000 !important;
}

.clear-chat-btn button:hover {
    background-color: #FEEFED !important;
    border-color: #E10600 !important;
    color: #000000 !important;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    padding: 0.75rem 0 !important;
}

/* User Message: Inset Neumorphic, Aligned Right */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: #F3E0CA !important;
    border-radius: 18px 4px 18px 18px !important;
    padding: 14px 18px !important;
    box-shadow: inset 2px 2px 5px #d9c5ae, inset -2px -2px 5px #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    max-width: 80% !important;
    margin-left: auto !important;
}

/* Assistant Message: Raised Neumorphic, Aligned Left */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background: #F9EAD7 !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 16px 20px !important;
    box-shadow: 5px 5px 12px #dfccb5, -5px -5px 12px #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.9) !important;
    line-height: 1.6 !important;
    max-width: 90% !important;
}

[data-testid="stChatMessageContent"] * {
    color: #000000 !important;
}

/* Citations Pill Badge */
.citation-badge {
    display: inline-block;
    background-color: #BFE23A;
    color: #000000 !important;
    font-weight: 800;
    font-size: 0.75rem;
    padding: 1px 7px;
    margin: 0 3px;
    border-radius: 6px;
    box-shadow: 1px 1px 3px rgba(160, 185, 45, 0.4);
    vertical-align: baseline;
    border: 1px solid #A5C925;
}

/* Sources Expander */
[data-testid="stExpander"] {
    background: #F9EAD7 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.7) !important;
    box-shadow: 3px 3px 8px #dfccb5, -3px -3px 8px #ffffff !important;
    margin-top: 8px !important;
}

[data-testid="stExpander"] summary, [data-testid="stExpander"] summary span {
    font-weight: 700 !important;
    color: #000000 !important;
    font-size: 0.88rem !important;
}

div[data-testid="stText"], pre {
    background: #FAF1E4 !important;
    border-radius: 8px !important;
    padding: 10px !important;
    box-shadow: inset 2px 2px 4px #dfccb5, inset -2px -2px 4px #ffffff !important;
    font-size: 0.82rem !important;
    color: #000000 !important;
    font-family: monospace !important;
    white-space: pre-wrap !important;
    border: 1px solid rgba(209, 192, 172, 0.4) !important;
}

/* Alerts / Warnings */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 3px 3px 8px #dfccb5, -3px -3px 8px #ffffff !important;
}

.stAlert * {
    color: #000000 !important;
}

/* Chat Input: Zero Black Bar, Pure Black Text & Placeholder */
[data-testid="stBottom"], [data-testid="stBottom"] > div {
    background-color: #F9EAD7 !important;
    border-top: 1px solid rgba(214, 194, 170, 0.6) !important;
}

[data-testid="stChatInput"] {
    background-color: transparent !important;
}

[data-testid="stChatInput"] > div {
    background-color: #FDF6EE !important;
    border-radius: 16px !important;
    box-shadow: inset 3px 3px 6px #dfccb5, inset -3px -3px 6px #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    padding: 4px 8px !important;
}

[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #000000 !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #000000 !important;
    opacity: 0.85 !important;
}

/* Lime-green Send Button */
[data-testid="stChatInput"] button {
    background-color: #BFE23A !important;
    color: #000000 !important;
    border-radius: 12px !important;
    border: 1px solid #A5C925 !important;
    box-shadow: 2px 2px 5px #c8dc79, -2px -2px 5px #ffffff !important;
}

[data-testid="stChatInput"] button:hover {
    background-color: #CEEE49 !important;
    color: #000000 !important;
}

[data-testid="stChatInput"] button svg {
    fill: #000000 !important;
}
</style>
"""

st.markdown(NEUMORPHIC_CSS, unsafe_allow_html=True)


def format_citations(text: str) -> str:
    """Format inline citations like [1] into styled neumorphic pill badges with black text."""
    return re.sub(r"\[(\d+)\]", r'<span class="citation-badge">[\1]</span>', text)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str, "result": dict|None}

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ---------------------------------------------------------------------------
# Header (Unclipped, Single Row / Responsive, Exact Headings)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="header-wrapper">
        <h1 class="header-title">Enterprise Knowledge Search Assistant</h1>
        <span class="header-subtitle">Hybrid RAG • ChromaDB & BM25</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar (Compact & High Contrast Pure Black Text)
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS = [
    "What is the VPN address and how long must my password be?",
    "What is the per diem limit for meals during travel?",
    "How many PTO days do I get per year?",
    "Who do I contact about a security incident?",
]

with st.sidebar:
    st.subheader("Sample Questions")

    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sb_{q}", use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    st.markdown('<div class="clear-chat-btn">', unsafe_allow_html=True)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Chat Area: Empty State vs Conversation View
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    # Centered Welcome Section when no messages exist
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-title">How can I help you today?</div>
            <div class="welcome-subtitle">Ask questions about Acme Corp's enterprise knowledge base.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Compact Suggested Question Cards
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"🔑 {SAMPLE_QUESTIONS[0]}", key="card_0", use_container_width=True):
            st.session_state.pending_question = SAMPLE_QUESTIONS[0]
            st.rerun()
        if st.button(f"🏖️ {SAMPLE_QUESTIONS[2]}", key="card_2", use_container_width=True):
            st.session_state.pending_question = SAMPLE_QUESTIONS[2]
            st.rerun()
    with col2:
        if st.button(f"🍽️ {SAMPLE_QUESTIONS[1]}", key="card_1", use_container_width=True):
            st.session_state.pending_question = SAMPLE_QUESTIONS[1]
            st.rerun()
        if st.button(f"🚨 {SAMPLE_QUESTIONS[3]}", key="card_3", use_container_width=True):
            st.session_state.pending_question = SAMPLE_QUESTIONS[3]
            st.rerun()

else:
    # Render existing conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(format_citations(msg["content"]), unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

            # Sources expander for assistant messages
            if msg["role"] == "assistant" and msg.get("result") is not None:
                result = msg["result"]
                if result.get("error"):
                    st.warning(result["error"])
                elif result.get("grounded") and result.get("sources"):
                    with st.expander("📄 Sources"):
                        for src in result["sources"]:
                            st.markdown(f"**{src.get('source', '')}** — *{src.get('title', '')}*")
                            st.text(src.get("text", "")[:600])
                            st.divider()
                elif not result.get("grounded"):
                    st.info("No supporting documents found.")


# ---------------------------------------------------------------------------
# Input Handling
# ---------------------------------------------------------------------------

chat_input = st.chat_input("Ask a question about Acme Corp policies...")

# If triggered by a card or sidebar button, consume pending_question
question: str | None = chat_input or st.session_state.pending_question
if st.session_state.pending_question:
    st.session_state.pending_question = None

if question:
    # 1. User message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question, "result": None})

    # 2. Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                result = answer_question(question)
                answer = result.get("answer", "I couldn't find this in the knowledge base.")
            except Exception as exc:
                result = {"answer": str(exc), "sources": [], "grounded": False}
                st.error(f"Error generating answer: {exc}")
                answer = str(exc)

        st.markdown(format_citations(answer), unsafe_allow_html=True)

        if result.get("error"):
            st.warning(result["error"])
        elif result.get("grounded") and result.get("sources"):
            with st.expander("📄 Sources"):
                for src in result["sources"]:
                    st.markdown(f"**{src.get('source', '')}** — *{src.get('title', '')}*")
                    st.text(src.get("text", "")[:600])
                    st.divider()
        elif not result.get("grounded"):
            st.info("No supporting documents found.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "result": result,
    })
    st.rerun()
