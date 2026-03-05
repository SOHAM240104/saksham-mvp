import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from pydantic import ConfigDict
from typing import List

load_dotenv()

# Use Streamlit secrets when available (Cloud or local .streamlit/secrets.toml)
def _apply_streamlit_secrets():
    try:
        for key in (
            "OPENAI_API_KEY",
            "LANGSMITH_TRACING",
            "LANGSMITH_API_KEY",
            "LANGSMITH_PROJECT",
            "LANGSMITH_ENDPOINT",
        ):
            if key in st.secrets and str(st.secrets.get(key)).strip():
                os.environ[key] = str(st.secrets[key]).strip()
    except Exception:
        pass


_apply_streamlit_secrets()

_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# CONSTANTS
# -------------------------
PLATFORM_APPLE  = "IOS18"
PLATFORM_GOOGLE = "PIXEL"
MAX_INPUT_LENGTH = 2000
HISTORY_WINDOW   = 8
RETRIEVAL_K      = 5
REFUSAL_PHRASE   = "I couldn't find this information in the available sources."
ENABLE_GROUNDING_CHECK = False


# -------------------------
# DB PATH RESOLVER
# -------------------------
def _get_db_path():
    for base in (_APP_DIR, os.getcwd(), "."):
        for sub in ("combined", "ios18", "pixel"):
            p = os.path.abspath(os.path.join(base, "care_vector_db", sub))
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "index.faiss")):
                return p
    return os.path.join(_APP_DIR, "care_vector_db", "combined")


# -------------------------
# SYSTEM PROMPT LOADER
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
    return FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# -------------------------
# QUERY REWRITER
# -------------------------
@st.cache_data(show_spinner=False)
def rewrite_query(user_message: str, platform: str) -> str:
    if platform == PLATFORM_APPLE:
        platform_hint = "iPhone iOS 18"
    elif platform == PLATFORM_GOOGLE:
        platform_hint = "Google Pixel Android"
    else:
        platform_hint = "iPhone iOS 18 OR Google Pixel Android"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system = (
        "You are a search query expander for a phone support knowledge base. "
        "Your only job: rewrite the user's message into a clear, specific search query "
        "that retrieves the most relevant documentation from a vector database. "
        "Rules: expand abbreviations, infer intent from short or vague messages, "
        "add relevant synonyms and related feature names, include the platform. "
        "Return ONLY the rewritten query — no explanation, no extra text."
    )
    human = (
        f"Platform: {platform_hint}\n"
        f"User message: {user_message}\n\n"
        "Rewritten search query:"
    )
    try:
        result = llm.invoke([
            {"role": "system", "content": system},
            {"role": "user",   "content": human},
        ])
        rewritten = result.content.strip()
        return rewritten if rewritten else user_message
    except Exception:
        return user_message


# -------------------------
# PLATFORM FILTERED RETRIEVER
# Defined at module level — not inside build_chain — so it's clean and reusable.
# -------------------------
class PlatformFilteredRetriever(BaseRetriever):
    base: BaseRetriever
    platform: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = self.base.invoke(query)
        return [d for d in docs if d.metadata.get("platform") == self.platform]


# -------------------------
# BUILD CHAIN
# -------------------------
@st.cache_resource
def build_chain(_vectorstore, _system_prompt, platform_filter=None):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True,
    )

    _human_text = (
        "The following numbered sources are the ONLY allowed basis for your answer. "
        "Do not use general knowledge.\n\n"
        "Official support documentation (numbered sources):\n{context}\n\n"
        "Conversation history:\n{history}\n\n"
        "Current platform context: {current_platform}\n\n"
        "User message:\n{input}"
    )
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=_system_prompt),
        HumanMessagePromptTemplate.from_template(_human_text),
    ])

    document_prompt = PromptTemplate.from_template(
        "Source {source_index}:\n{page_content}"
    )
    combine_docs_chain = create_stuff_documents_chain(
        llm,
        prompt,
        document_prompt=document_prompt,
        document_separator="\n\n---\n\n",
    )

    base_retriever = _vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    if platform_filter is not None:
        retriever = PlatformFilteredRetriever(base=base_retriever, platform=platform_filter)
    else:
        retriever = base_retriever

    return (retriever, combine_docs_chain)


# -------------------------
# DEVICE DETECTION
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
# -------------------------
def sanitize_input(text: str) -> str:
    if not text:
        return ""
    sanitized = "".join(
        ch for ch in text if ch == "\n" or (ord(ch) >= 32 and ord(ch) != 127)
    )
    return sanitized[:MAX_INPUT_LENGTH].strip()


# -------------------------
# RAG USAGE DETECTOR
# Only show sources when the response is genuinely grounded in retrieved docs.
# Prevents sources panel appearing on greetings, scam acks, off-topic redirects.
# -------------------------
def response_used_rag(response: str, docs: list) -> bool:
    if not docs or not response:
        return False
    response_lower = response.lower()
    for doc in docs:
        chunk_words = doc.page_content.strip().lower().split()
        trigrams = [
            " ".join(chunk_words[i:i+3])
            for i in range(len(chunk_words) - 2)
        ]
        matches = sum(1 for tg in trigrams if tg in response_lower)
        if matches >= 2:
            return True
    return False


# -------------------------
# SOURCE FORMATTER
# Uses article title from metadata instead of URL filename.
# -------------------------
MAX_SOURCES_SHOWN = 5


def render_sources(docs: list):
    if not docs:
        return

    seen = set()
    unique_docs = []
    for doc in docs:
        src = doc.metadata.get("source", "")
        if src not in seen:
            seen.add(src)
            unique_docs.append(doc)
    unique_docs = unique_docs[:MAX_SOURCES_SHOWN]

    with st.expander(
        f"📄 {len(unique_docs)} source{'s' if len(unique_docs) > 1 else ''} referenced",
        expanded=False
    ):
        for i, doc in enumerate(unique_docs, 1):
            meta        = doc.metadata
            source_url  = meta.get("source", "")
            platform    = meta.get("platform", "")

            # Use article title from metadata — much more readable than URL basename
            # Falls back to the URL itself if title is missing
            title = meta.get("title", "")
            if not title or len(title) < 5:
                title = source_url  # last resort

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

            # Clickable title links to the source URL
            link_html = (
                f"<a href='{source_url}' target='_blank' style='color:{badge_border};"
                f"text-decoration:none;font-weight:600;font-size:0.9em;'>{title}</a>"
                if source_url else
                f"<span style='font-weight:600;font-size:0.9em;color:#333;'>{title}</span>"
            )

            st.markdown(
                f"""
                <div style='
                    border: 1px solid {badge_border};
                    border-left: 4px solid {badge_border};
                    border-radius: 8px;
                    background: {badge_color};
                    padding: 12px 16px 8px 16px;
                    margin-bottom: 4px;
                '>
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; flex-wrap:wrap; gap:6px;'>
                        <span>{i}. {link_html}</span>
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
                </div>
                """,
                unsafe_allow_html=True,
            )

            content = doc.page_content.strip()
            st.markdown(
                f"""
                <div style='
                    background:#ffffff;
                    border:1px solid #e0e0e0;
                    border-top:none;
                    border-radius:0 0 8px 8px;
                    padding:12px 16px;
                    font-size:0.84em;
                    color:#444;
                    line-height:1.65;
                    white-space:pre-wrap;
                    margin-bottom:14px;
                '>{content}</div>
                """,
                unsafe_allow_html=True,
            )


# -------------------------
# GROUNDING CHECK (optional)
# -------------------------
def _is_answer_grounded(answer: str, docs: list) -> bool:
    if not answer or not docs:
        return not answer
    context_blob = "\n\n".join(
        f"Source {i}:\n{d.page_content}" for i, d in enumerate(docs, 1)
    )
    checker = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = (
        "Does the following assistant answer contain ANY factual claim or step that is NOT "
        "stated or clearly implied in the numbered sources below? "
        "Answer only YES or NO.\n\n"
        "Sources:\n" + context_blob[:12000] + "\n\nAssistant answer:\n" + answer[:2000]
    )
    try:
        out  = checker.invoke(prompt)
        text = out.content.strip().upper() if hasattr(out, "content") else str(out).strip().upper()
        return text.startswith("NO")
    except Exception:
        return True


# -------------------------
# HISTORY BUILDER
# -------------------------
def build_history_text(messages: list) -> str:
    relevant = messages[1:] if len(messages) > 1 else []
    recent   = relevant[-(HISTORY_WINDOW):]
    lines    = []
    for m in recent:
        role_label = "User" if m["role"] == "user" else "Care"
        lines.append(f"[{role_label}]: {m['content']}")
    return "\n".join(lines)


# -------------------------
# MAIN APP
# -------------------------
def main():
    st.set_page_config(page_title="Care Assistant", page_icon="💛")
    st.title("Care Assistant")

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

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input_raw = st.chat_input("Type your message")
    if user_input_raw is None:
        return

    user_input = sanitize_input(user_input_raw)
    if not user_input:
        return

    if len(user_input_raw) > MAX_INPUT_LENGTH:
        st.warning(f"Your message was trimmed to {MAX_INPUT_LENGTH} characters.")

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    db_path     = _get_db_path()
    vectorstore = load_vectorstore(db_path)
    if vectorstore is None:
        st.error("Support database not available.")
        st.code(db_path, language=None)
        st.info(
            "**Local:** Run `python ingest_faiss.py` in the project folder, then restart.\n\n"
            "**Streamlit Cloud:** Include `care_vector_db/` in the repo or set run command to: "
            "`python ingest_faiss.py && streamlit run app.py`"
        )
        return

    system_prompt = load_system_prompt()

    detected = get_platform_filter(user_input)
    if detected is not None:
        st.session_state.last_platform = detected

    effective_filter    = st.session_state.last_platform
    platform_for_prompt = effective_filter if effective_filter is not None else "NOT_CONFIRMED"

    search_query = rewrite_query(user_input, platform_for_prompt)

    retriever, combine_chain = build_chain(
        vectorstore, system_prompt, platform_filter=effective_filter
    )
    history_text = build_history_text(st.session_state.messages)

    retrieved_docs = retriever.invoke(search_query)
    for i, d in enumerate(retrieved_docs, 1):
        d.metadata["source_index"] = i

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in combine_chain.stream({
                "input":            user_input,
                "history":          history_text,
                "current_platform": platform_for_prompt,
                "context":          retrieved_docs,
            }):
                if isinstance(chunk, str):
                    full_response += chunk
                    response_placeholder.markdown(full_response)
                elif isinstance(chunk, dict) and "answer" in chunk:
                    full_response += chunk["answer"]
                    response_placeholder.markdown(full_response)

        except Exception as e:
            st.error("Something went wrong while generating a response. Please try again.")
            st.exception(e)
            return

        if ENABLE_GROUNDING_CHECK and retrieved_docs and full_response:
            if not _is_answer_grounded(full_response, retrieved_docs):
                full_response = REFUSAL_PHRASE
                response_placeholder.markdown(full_response)

        render_sources(retrieved_docs)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()