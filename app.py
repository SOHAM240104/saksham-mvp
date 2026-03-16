import base64
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from agent_core import (
    _get_db_path,
    load_system_prompt as _load_system_prompt,
    load_vectorstore,
    build_retriever,
    make_search_tool,
    infer_platform_from_messages_lc,
    sanitize_input,
    PLATFORM_APPLE,
    PLATFORM_GOOGLE,
    MAX_INPUT_LENGTH,
)
from pdf_generator import create_sources_pdf

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


def _enable_langsmith_tracing():
    """Enable LangSmith tracing from env/secrets so it works locally and on Streamlit Cloud."""
    api_key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    try:
        from langsmith import Client, configure

        client = Client(
            api_key=api_key,
            api_url=os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        )
        configure(
            client=client,
            project_name=os.environ.get("LANGSMITH_PROJECT") or "default",
            enabled=True,
        )
    except Exception:
        pass


_apply_streamlit_secrets()
_enable_langsmith_tracing()

# -------------------------
# APP-ONLY CONSTANTS
# -------------------------
REFUSAL_PHRASE = "I couldn't find this information in the available sources."
MAX_SOURCES_SHOWN = 10


def load_system_prompt():
    """Load system prompt; show Streamlit error and stop if missing."""
    content = _load_system_prompt()
    if not content:
        st.error("system_prompt.txt not found.")
        st.stop()
    return content


# -------------------------
# SOURCE FORMATTER
# Uses article title from metadata instead of URL filename.
# Show more sources and rank by chunk count so ringtone / relevant pages appear.
# -------------------------
def render_sources(docs: list):
    if not docs:
        return

    # Count chunks per source (URL); keep one doc per source for display
    source_to_doc = {}
    source_to_count = {}
    for doc in docs:
        src = doc.metadata.get("source", "")
        if not src:
            continue
        if src not in source_to_doc:
            source_to_doc[src] = doc
        source_to_count[src] = source_to_count.get(src, 0) + 1

    # Sort by chunk count descending so most-relevant sources (e.g. ringtone) show first
    sorted_sources = sorted(
        source_to_count.keys(),
        key=lambda s: source_to_count[s],
        reverse=True,
    )
    unique_docs = [source_to_doc[s] for s in sorted_sources[:MAX_SOURCES_SHOWN]]

    with st.expander(
        f"📄 {len(unique_docs)} source{'s' if len(unique_docs) > 1 else ''} referenced",
        expanded=False
    ):
        for i, doc in enumerate(unique_docs, 1):
            meta        = doc.metadata or {}
            source_url  = meta.get("source", "")
            platform    = meta.get("platform", "")
            # Optional: header_path, title — always use .get() to avoid KeyError
            title = meta.get("title", "") or ""
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

    # Vectorstore: load once per session
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        db_path = _get_db_path()
        vectorstore = load_vectorstore(db_path)
        if vectorstore is None:
            st.error("Support database not available.")
            st.code(db_path, language=None)
            st.info(
                "**Local:** Run `python ingest.py` in the project folder, then restart.\n\n"
                "**Streamlit Cloud:** Include `care_vector_db/` in the repo or set run command to: "
                "`python ingest.py && streamlit run app.py`"
            )
            return
        st.session_state["vectorstore"] = vectorstore
        st.session_state["db_path"] = db_path

    # System prompt: load once per session
    if st.session_state.get("system_prompt") is None:
        content = load_system_prompt()  # may st.stop() if missing
        st.session_state["system_prompt"] = content
    system_prompt = st.session_state["system_prompt"]

    # Pre-build retrievers once per session
    if st.session_state.get("retrievers") is None:
        st.session_state["retrievers"] = {
            None: build_retriever(vectorstore, None),
            "IOS18": build_retriever(vectorstore, "IOS18"),
            "PIXEL": build_retriever(vectorstore, "PIXEL"),
        }
    retriever_map = st.session_state["retrievers"]
    # Build message list once: history + current user message
    history_lc = []
    for m in st.session_state.messages[1:]:
        if m["role"] == "user":
            history_lc.append(HumanMessage(content=m["content"]))
        else:
            history_lc.append(AIMessage(content=m["content"]))
    history_lc.append(HumanMessage(content=user_input))

    # Infer platform from this turn; once user says iPhone or Pixel, persist it for the rest of the session
    inferred_platform = infer_platform_from_messages_lc(history_lc)
    if inferred_platform in ("IOS18", "PIXEL"):
        st.session_state["confirmed_device"] = inferred_platform
    current_platform = (
        st.session_state.get("confirmed_device")
        or inferred_platform
    )

    tool_instruction = (
        "\n\n## TOOL USE — MANDATORY FOR STEPS/HOW-TO\n"
        "When the user asks how to do something on their phone (e.g. change ringtone, wallpaper, settings, use an app), "
        "you MUST call the search_support_docs tool in this turn. Do NOT skip the tool call even if you answered a similar question earlier in the conversation; the user only receives the sources PDF when you call the tool now. Use the query as the user's question and set platform to "
        "'IOS18' or 'PIXEL' when device is confirmed. Answer ONLY from the tool result. "
        "Never give steps from memory — if you do not call the tool, do not give steps."
    )
    system_with_platform = (
        system_prompt
        + tool_instruction
        + "\n\nCurrent device context (use for steps and filtering): "
        + current_platform
    )

    messages_lc = [SystemMessage(content=system_with_platform)] + history_lc

    # Persist docs_container across turns so that when support docs are fetched
    # in any tool call, the latest docs are available for PDF generation.
    docs_container: dict = st.session_state.get("docs_container", {})
    if docs_container is None:
        docs_container = {}

    search_tool = make_search_tool(vectorstore, docs_container, retriever_map=retriever_map)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([search_tool])

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            while True:
                response = llm_with_tools.invoke(messages_lc)
                messages_lc.append(response)

                if not getattr(response, "tool_calls", None) or not response.tool_calls:
                    full_response = response.content or ""
                    break

                for tc in response.tool_calls:
                    name = tc.get("name", "search_support_docs")
                    args = tc.get("args") or {}
                    tool_call_id = tc.get("id", "")
                    result = search_tool.invoke(args)
                    messages_lc.append(
                        ToolMessage(tool_call_id=tool_call_id, content=result)
                    )

            response_placeholder.markdown(full_response)

        except Exception as e:
            st.error("Something went wrong while generating a response. Please try again.")
            st.exception(e)
            return

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    # Save latest docs_container for future turns
    st.session_state["docs_container"] = docs_container

    # If search_support_docs was called and returned docs, show PDF (trust the tool stream).
    retrieved_docs = docs_container.get("docs", [])
    if retrieved_docs:
        try:
            pdf_b64 = create_sources_pdf(retrieved_docs[:MAX_SOURCES_SHOWN])
        except Exception:
            pdf_b64 = ""

        if pdf_b64:
            data_url = f"data:application/pdf;base64,{pdf_b64}"
            file_name = "Care_Support_Sources.pdf"
            with st.chat_message("assistant"):
                # File bubble
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 12px;
                        border: 1px solid #d0d7de;
                        background: #f7f7f8;
                        padding: 10px 12px;
                        display: flex;
                        align-items: center;
                        gap: 10px;
                        max-width: 420px;
                        margin-bottom: 8px;
                    ">
                        <div style="
                            width: 32px;
                            height: 40px;
                            border-radius: 6px;
                            background: linear-gradient(135deg,#f97316,#ea580c);
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            color:#fff;
                            font-size:0.7rem;
                            font-weight:600;
                        ">
                            PDF
                        </div>
                        <div style="flex:1; min-width:0;">
                            <div style="
                                font-size:0.85rem;
                                font-weight:600;
                                color:#111827;
                                overflow:hidden;
                                text-overflow:ellipsis;
                                white-space:nowrap;
                            ">{file_name}</div>
                            <div style="font-size:0.75rem;color:#6b7280;">
                                Tap to open the PDF preview below
                            </div>
                        </div>
                        <a href="{data_url}" target="_blank" style="
                            font-size:0.8rem;
                            font-weight:600;
                            color:#2563eb;
                            text-decoration:none;
                            white-space:nowrap;
                        ">
                            Open
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Inline PDF viewer
                st.markdown(
                    f"""
                    <iframe
                        src="{data_url}"
                        style="width:100%;max-width:540px;height:420px;border:1px solid #e5e7eb;border-radius:8px;"
                    ></iframe>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()