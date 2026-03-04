import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# CONSTANTS
# -------------------------
PLATFORM_APPLE = "IOS18"
PLATFORM_GOOGLE = "PIXEL"
MAX_INPUT_LENGTH = 2000       # characters — hard cap on user input
HISTORY_WINDOW = 8            # number of messages to pass as context
RETRIEVAL_K = 8               # number of docs to retrieve


# -------------------------
# DB PATH RESOLVER
# -------------------------
def _get_db_path():
    for base in (_APP_DIR, os.getcwd(), "."):
        p = os.path.abspath(os.path.join(base, "care_vector_db"))
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "chroma.sqlite3")):
            return p
    return os.path.join(_APP_DIR, "care_vector_db")


# -------------------------
# SYSTEM PROMPT LOADER
# Fix: resolve path relative to script dir, not cwd
# -------------------------
@st.cache_data
def load_system_prompt():
    path = os.path.join(_APP_DIR, "system_prompt.txt")
    if not os.path.exists(path):
        st.error(f"system_prompt.txt not found at: {path}")
        st.stop()
    with open(path, encoding="utf-8") as f:
        return f.read()


# -------------------------
# VECTORSTORE LOADER
# -------------------------
@st.cache_resource
def load_vectorstore(db_path):
    if not os.path.exists(db_path):
        return None
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )


# -------------------------
# BUILD CHAIN
# Fix: system prompt loaded once via cache; chain rebuilt only when platform changes
# -------------------------
@st.cache_resource
def build_chain(_vectorstore, _system_prompt, platform_filter=None):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _system_prompt),
        ("human", """
Official Support Documentation:
{context}

Conversation History:
{history}

Current platform context: {current_platform}

User Message:
{input}
""")
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    search_kwargs = {"k": RETRIEVAL_K}
    if platform_filter is not None:
        search_kwargs["filter"] = {"platform": platform_filter}

    retriever = _vectorstore.as_retriever(search_kwargs=search_kwargs)
    return create_retrieval_chain(retriever, combine_docs_chain)


# -------------------------
# DEVICE DETECTION
# Scans current message only — session state carries confirmed platform forward
# -------------------------
def get_platform_filter(text: str):
    if not text:
        return None
    t = text.lower()
    if "pixel" in t or "google" in t:
        return PLATFORM_GOOGLE
    if "iphone" in t or "apple" in t or "ios" in t:
        return PLATFORM_APPLE
    return None


# -------------------------
# INPUT SANITIZER
# Strips control characters, enforces length cap
# -------------------------
def sanitize_input(text: str) -> str:
    if not text:
        return ""
    # Remove null bytes and other control characters (keep newlines)
    sanitized = "".join(
        ch for ch in text if ch == "\n" or (ord(ch) >= 32 and ord(ch) != 127)
    )
    return sanitized[:MAX_INPUT_LENGTH].strip()


# -------------------------
# RAG USAGE DETECTOR
# Returns True only when the response is genuinely grounded in retrieved docs.
# Heuristic: check if any meaningful chunk of the response overlaps with doc content.
# This prevents sources showing on greetings, scam acks, off-topic redirects, etc.
# -------------------------
def response_used_rag(response: str, docs: list) -> bool:
    if not docs or not response:
        return False

    response_lower = response.lower()

    for doc in docs:
        # Take first 200 chars of the chunk as a fingerprint
        chunk_words = doc.page_content.strip().lower().split()
        # Build trigrams from the chunk
        trigrams = [
            " ".join(chunk_words[i:i+3])
            for i in range(len(chunk_words) - 2)
        ]
        # If 2 or more trigrams from this doc appear in the response, it was used
        matches = sum(1 for tg in trigrams if tg in response_lower)
        if matches >= 2:
            return True

    return False


# -------------------------
# SOURCE FORMATTER
# Renders retrieved docs as clean structured citations in Streamlit
# -------------------------
def render_sources(docs: list):
    if not docs:
        return

    # Deduplicate by source path, preserve order
    seen = set()
    unique_docs = []
    for doc in docs:
        src = doc.metadata.get("source", "")
        if src not in seen:
            seen.add(src)
            unique_docs.append(doc)

    with st.expander(f"📄 {len(unique_docs)} source{'s' if len(unique_docs) > 1 else ''} referenced", expanded=False):

        for i, doc in enumerate(unique_docs, 1):
            meta = doc.metadata

            source_path = meta.get("source", "")
            platform    = meta.get("platform", "")
            section     = meta.get("section", meta.get("title", ""))
            page        = meta.get("page", None)
            source_name = os.path.basename(source_path) if source_path else "Unknown"

            # Platform badge colours
            if platform == PLATFORM_APPLE:
                badge_color  = "#e8f4f8"
                badge_border = "#0071e3"
                badge_text   = "🍎 iPhone (iOS 18)"
            elif platform == PLATFORM_GOOGLE:
                badge_color  = "#e8f5e9"
                badge_border = "#34a853"
                badge_text   = "📱 Google Pixel"
            else:
                badge_color  = "#f5f5f5"
                badge_border = "#aaa"
                badge_text   = "📄 General"

            # Build header line
            page_tag = f" &nbsp;·&nbsp; Page {page}" if page is not None else ""
            section_tag = f"<br><span style='font-size:0.78em;color:#888;'>Section: {section}</span>" if section else ""

            st.markdown(
                f"""
                <div style='
                    border: 1px solid {badge_border};
                    border-left: 4px solid {badge_border};
                    border-radius: 8px;
                    background: {badge_color};
                    padding: 12px 16px 6px 16px;
                    margin-bottom: 4px;
                '>
                    <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;'>
                        <span style='font-weight:600; font-size:0.9em; color:#333;'>
                            {i}. {source_name}{page_tag}
                        </span>
                        <span style='
                            font-size:0.75em;
                            background:white;
                            border:1px solid {badge_border};
                            border-radius:20px;
                            padding:2px 10px;
                            color:{badge_border};
                            font-weight:500;
                        '>{badge_text}</span>
                    </div>
                    {section_tag}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Full content in a clean scrollable block — no truncation
            content = doc.page_content.strip()
            st.markdown(
                f"""
                <div style='
                    background: #ffffff;
                    border: 1px solid #e0e0e0;
                    border-top: none;
                    border-radius: 0 0 8px 8px;
                    padding: 12px 16px;
                    font-size: 0.84em;
                    color: #444;
                    line-height: 1.65;
                    white-space: pre-wrap;
                    margin-bottom: 14px;
                '>{content}</div>
                """,
                unsafe_allow_html=True
            )


# -------------------------
# HISTORY BUILDER
# Fix: skips greeting (index 0), uses structured separator to prevent role spoofing
# -------------------------
def build_history_text(messages: list) -> str:
    # Skip the initial greeting message (index 0)
    relevant = messages[1:] if len(messages) > 1 else []
    # Take last HISTORY_WINDOW messages
    recent = relevant[-(HISTORY_WINDOW):]
    lines = []
    for m in recent:
        role_label = "User" if m["role"] == "user" else "Care"
        # Wrap content to prevent role-spoofing injection via message content
        lines.append(f"[{role_label}]: {m['content']}")
    return "\n".join(lines)


# -------------------------
# MAIN APP
# -------------------------
def main():
    st.set_page_config(page_title="Care Assistant", page_icon="💛")
    st.title("Care Assistant")

    # Session state init
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi there 👋 Welcome to Saksham Support. I'm here to help with your "
                    "technical issues — or to listen if something feels off or scam-related. "
                    "How can I help you today?"
                ),
            }
        ]
    if "last_platform" not in st.session_state:
        st.session_state.last_platform = None

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input_raw = st.chat_input("Type your message")

    if user_input_raw is None:
        return

    # Sanitize input
    user_input = sanitize_input(user_input_raw)
    if not user_input:
        return

    # Warn user if input was truncated
    if len(user_input_raw) > MAX_INPUT_LENGTH:
        st.warning(f"Your message was trimmed to {MAX_INPUT_LENGTH} characters.")

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Load resources
    db_path = _get_db_path()
    vectorstore = load_vectorstore(db_path)
    if vectorstore is None:
        st.error("Support database not available.")
        st.code(db_path, language=None)
        st.info(
            "**Local:** Run `python ingest.py` in the project folder, then restart.\n\n"
            "**Streamlit Cloud:** Include `care_vector_db` in the repo or set run command to: "
            "`python ingest.py && streamlit run app.py`"
        )
        return

    system_prompt = load_system_prompt()

    # Device detection: update session platform if user mentions device in this message
    detected = get_platform_filter(user_input)
    if detected is not None:
        st.session_state.last_platform = detected

    effective_filter = st.session_state.last_platform
    platform_for_prompt = effective_filter if effective_filter is not None else "NOT_CONFIRMED"

    chain = build_chain(vectorstore, system_prompt, platform_filter=effective_filter)
    history_text = build_history_text(st.session_state.messages)

    # Stream response with error handling
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        retrieved_docs = []

        try:
            for chunk in chain.stream({
                "input": user_input,
                "history": history_text,
                "current_platform": platform_for_prompt,
            }):
                # Capture retrieved docs from the first chunk that contains them
                if "context" in chunk and not retrieved_docs:
                    retrieved_docs = chunk["context"]

                if "answer" in chunk:
                    full_response += chunk["answer"]
                    response_placeholder.markdown(full_response)

        except Exception as e:
            st.error("Something went wrong while generating a response. Please try again.")
            st.exception(e)
            return

        # Only show sources when the response is genuinely doc-grounded
        if retrieved_docs and response_used_rag(full_response, retrieved_docs):
            render_sources(retrieved_docs)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })


if __name__ == "__main__":
    main()