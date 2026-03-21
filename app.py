import base64
import hashlib
import os
import io
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore

    _OCR_AVAILABLE = True
except Exception:
    _OCR_AVAILABLE = False

try:
    # Used to convert uploaded images into publicly accessible URLs
    # for Google Fact Check `claims:imageSearch`.
    from image2url import Image2URLClient  # type: ignore

    _IMAGE2URL_AVAILABLE = True
except Exception:
    _IMAGE2URL_AVAILABLE = False

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
from tools.verify_claim_tool import verify_claim

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


def detect_mode_switch(user_text: str):
    """Detect tech/scam mode switch intent from user text; returns 'tech', 'scam', or None."""
    if not user_text or not str(user_text).strip():
        return None
    t = str(user_text).lower()
    tech_markers = (
        "tech mode",
        "switch to tech",
        "go to tech",
        "tech support",
        "phone help",
    )
    scam_markers = (
        "scam mode",
        "switch to scam",
        "check scam",
        "scam check",
        "fraud check",
    )
    for phrase in tech_markers:
        if phrase in t:
            return "tech"
    for phrase in scam_markers:
        if phrase in t:
            return "scam"
    return None


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

    if st.session_state.get("_processing"):
        with st.spinner("Processing..."):
            st.stop()

    # -------------------------
    # Mode selection (UI-only)
    # -------------------------
    last_mode_for_msg = st.session_state.get("_last_mode_for_msg", None)
    mode = st.session_state.get("mode", None)

    st.markdown(
        """
        <style>
        /* Subtle global background accents by mode */
        .stAppBackground {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_tech, col_scam = st.columns(2)
    with col_tech:
        if st.button("Tech Support", use_container_width=True, key="btn_mode_tech"):
            st.session_state.mode = "tech"
    with col_scam:
        if st.button(
            "Scam / Misinformation Check",
            use_container_width=True,
            key="btn_mode_scam",
        ):
            st.session_state.mode = "scam"

    mode = st.session_state.get("mode", None)

    # Apply CSS accents based on mode
    if mode == "tech":
        st.markdown(
            """
            <style>
            body { background: #f0f8ff; }
            div[data-testid="stChatMessage"] {
                background: rgba(0, 112, 243, 0.06);
                border-radius: 14px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif mode == "scam":
        st.markdown(
            """
            <style>
            body { background: #fff5f5; }
            div[data-testid="stChatMessage"] {
                background: rgba(255, 0, 0, 0.06);
                border-radius: 14px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

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

    # Add a visible mode-change message once per switch (do not clear history)
    if mode is not None and mode != st.session_state.get("_last_mode_for_msg", None):
        if mode == "scam":
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "You are now in Scam Detection mode. You can upload images or check suspicious messages.",
                }
            )
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "You are now in Tech Support mode. Ask about devices, platforms, or issues.",
                }
            )
        st.session_state["_last_mode_for_msg"] = mode
        st.session_state["_mode_just_changed"] = True
        # Limit context in the prompt to what happened after the last mode switch.
        st.session_state["_mode_history_start_idx"] = len(st.session_state.messages)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("image_b64"):
                try:
                    img_bytes = base64.b64decode(msg["image_b64"])
                    st.image(img_bytes)
                except Exception:
                    pass
            st.write(msg.get("display_content") or msg.get("content", ""))

    user_input_raw = st.chat_input("Type your message")
    # Text-based mode switching (phrases like "tech mode" / "scam mode"); no button required.
    mode_switch = detect_mode_switch(user_input_raw)
    if (
        mode_switch is not None
        and mode_switch != st.session_state.get("mode")
    ):
        st.session_state.mode = mode_switch
        st.session_state.messages.append(
            {"role": "user", "content": user_input_raw}
        )
        if mode_switch == "tech":
            _confirm = (
                "You're now in Tech Support mode. Ask about your phone or device."
            )
        else:
            _confirm = (
                "You're now in Scam Detection mode. Share the message or upload a screenshot."
            )
        st.session_state.messages.append(
            {"role": "assistant", "content": _confirm}
        )
        st.session_state["_last_mode_for_msg"] = mode_switch
        st.session_state["_mode_just_changed"] = True
        st.session_state["_mode_history_start_idx"] = len(st.session_state.messages)
        with st.chat_message("user"):
            st.write(user_input_raw)
        with st.chat_message("assistant"):
            st.write(_confirm)
        return

    if mode is None:
        if user_input_raw is not None:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "I can help with technical support or scam detection.\n\n"
                        "Please select a mode above and re-enter your query."
                    ),
                }
            )
        st.info(
            "Please select a mode to continue:\n"
            "* Tech Support (for troubleshooting)\n"
            "* Scam Check (for suspicious messages)"
        )
        return

    uploaded_file = None
    if mode == "scam":
        uploaded_file = st.file_uploader(
            "Upload screenshots of messages, emails, or suspicious content",
            type=["png", "jpg", "jpeg"],
        )

    image_fingerprint = None
    image_b64 = None
    uploaded_image_bytes = None
    ocr_text = ""
    ocr_extraction_failed = False

    if uploaded_file is not None:
        try:
            img_bytes = uploaded_file.getvalue()
            uploaded_image_bytes = img_bytes
            image_fingerprint = hashlib.sha256(img_bytes).hexdigest()
            image_b64 = base64.b64encode(img_bytes).decode("ascii")
        except Exception:
            image_fingerprint = None
            image_b64 = None

    # Decide whether we should process a turn:
    # - text submit always triggers
    # - image upload auto-triggers if it's a new upload (and no text was submitted yet)
    should_process_from_image = False
    if mode == "scam":
        # Auto-trigger scam checks when a new image is uploaded.
        should_process_from_image = (
            uploaded_file is not None
            and user_input_raw is None
            and image_fingerprint is not None
            and image_fingerprint
            != st.session_state.get("last_processed_image_fingerprint")
        )
    should_process = (user_input_raw is not None) or should_process_from_image
    if not should_process:
        return

    # OCR (if we have an uploaded image)
    if (
        mode == "scam"
        and uploaded_file is not None
        and _OCR_AVAILABLE
        and image_fingerprint is not None
    ):
        # Reuse OCR text if this image was already processed earlier in the session.
        cached_fp = st.session_state.get("last_processed_image_fingerprint")
        if cached_fp == image_fingerprint:
            ocr_text = st.session_state.get("last_ocr_text", "") or ""
            image_b64 = st.session_state.get("last_image_b64", image_b64)
            if not ocr_text.strip():
                ocr_extraction_failed = True
        else:
            try:
                img_bytes = uploaded_file.getvalue()
                img = Image.open(io.BytesIO(img_bytes))
                # Basic size guard; helps keep OCR fast.
                if max(img.size) > 2000:
                    img.thumbnail((2000, 2000))
                img = img.convert("RGB")
                ocr_text = pytesseract.image_to_string(img) or ""
                st.session_state["last_processed_image_fingerprint"] = image_fingerprint
                st.session_state["last_ocr_text"] = ocr_text
                st.session_state["last_image_b64"] = image_b64
                if not ocr_text.strip():
                    ocr_extraction_failed = True
            except Exception:
                ocr_text = ""
                st.session_state["last_processed_image_fingerprint"] = image_fingerprint
                st.session_state["last_ocr_text"] = ""
                st.session_state["last_image_b64"] = image_b64
                ocr_extraction_failed = True

    prompt_for_missing_text = (
        "Analyze this image for scams or misinformation."
        if mode == "scam"
        else "Ask for tech troubleshooting."
    )
    raw_combined_input = (
        user_input_raw if user_input_raw is not None else prompt_for_missing_text
    )
    display_user_content = raw_combined_input

    # Scam mode greeting/ack guardrail: don't run verify_claim for "hi/hello".
    if mode == "scam" and user_input_raw is not None:
        lowered = (user_input_raw or "").strip().lower()
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
        if lowered in greetings:
            greet_msg = (
                "Hi! Please paste the suspicious message/claim or upload a screenshot, and I’ll help verify it."
            )
            # Still show the user's greeting in chat history.
            user_display = user_input_raw or ""
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": sanitize_input(user_display),
                    "display_content": user_display,
                }
            )
            st.session_state.messages.append({"role": "assistant", "content": greet_msg})
            with st.chat_message("assistant"):
                st.write(greet_msg)
            st.session_state["_processing"] = False
            st.session_state["_mode_just_changed"] = False
            return
    if ocr_text.strip():
        raw_combined_input = (
            raw_combined_input
            + "\n\n[OCR text may contain errors]\n"
            + "[OCR Extracted Text]:\n"
            + ocr_text.strip()
        )

    # Mode-aware prompt injection for the LLM
    mode_prefix = ""
    if mode == "tech":
        mode_prefix = (
            "[MODE: TECH SUPPORT]\n"
            "Focus on troubleshooting, documentation, and platform-specific help.\n"
            "Ignore scam verification context unless the user explicitly asks for it."
        )
    else:
        mode_prefix = (
            "[MODE: SCAM DETECTION]\n"
            "Focus on identifying fraud, scams, phishing, and misinformation.\n"
            "Ignore unrelated technical context.\n"
            "Prefer using verify_claim tool."
        )

    # Platform memory safety: only inject platform context when this turn mentions it.
    if mode == "tech":
        inferred_platform_for_ctx = "NOT_CONFIRMED"
        try:
            inferred_platform_for_ctx = infer_platform_from_messages_lc(
                [HumanMessage(content=raw_combined_input)]
            )
        except Exception:
            inferred_platform_for_ctx = "NOT_CONFIRMED"

        if inferred_platform_for_ctx in ("IOS18", "PIXEL"):
            platform_text = (
                "iPhone" if inferred_platform_for_ctx == "IOS18" else "Pixel"
            )
            st.session_state["platform"] = platform_text
            mode_prefix += f"\n[PLATFORM CONTEXT: {platform_text}]"

    # Provide the original user text in a clearly delimited block so the model
    # uses it as tool input (instead of including mode instructions).
    raw_combined_input = (
        mode_prefix + "\n\n[USER INPUT]\n" + raw_combined_input
    )
    user_input = sanitize_input(raw_combined_input)
    if not user_input:
        return

    if len(raw_combined_input) > MAX_INPUT_LENGTH:
        st.warning(f"Your message was trimmed to {MAX_INPUT_LENGTH} characters.")

    user_msg = {"role": "user", "content": user_input}
    if image_b64:
        user_msg["image_b64"] = image_b64
    user_msg["display_content"] = display_user_content
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        if image_b64:
            st.image(base64.b64decode(image_b64))
        st.write(display_user_content)

    # Scam mode + uploaded image: convert to a public URL and call verify_claim
    # directly via imageSearch (backend handled).
    if mode == "scam" and uploaded_file is not None and uploaded_image_bytes and _IMAGE2URL_AVAILABLE:
        try:
            client = Image2URLClient(
                endpoint=os.environ.get("IMAGE2URL_ENDPOINT", "https://www.image2url.com/api/upload"),
                timeout=15,
                max_size_mb=2,
            )
            orig_name = getattr(uploaded_file, "name", "") or "upload.jpg"
            suffix = os.path.splitext(orig_name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(uploaded_image_bytes)
                tmp.flush()
                result = client.upload_image(tmp.name, filename=orig_name)
                image_url = getattr(result, "url", None) or (result.get("url") if isinstance(result, dict) else None)

            if image_url:
                scam_result = verify_claim.invoke({"query": f"IMAGE_URI:{image_url}"})
                st.session_state.messages.append({"role": "assistant", "content": scam_result})
                with st.chat_message("assistant"):
                    st.write(scam_result)
                st.session_state["_processing"] = False
                st.session_state["_mode_just_changed"] = False
                return
        except Exception:
            # Fall back to OCR -> verify_claim via regular claims:search flow.
            pass

    # OCR edge-case: if we failed to extract any text, do not call the LLM.
    if mode == "scam" and ocr_extraction_failed:
        ocr_empty_msg = (
            "I couldn't extract text from the image.\n"
            "Please describe what you'd like me to check."
        )
        st.session_state.messages.append({"role": "assistant", "content": ocr_empty_msg})
        with st.chat_message("assistant"):
            st.write(ocr_empty_msg)
        st.session_state["_processing"] = False
        st.session_state["_mode_just_changed"] = False
        return

    # Intent routing (wrong-mode redirects, scam vs tech) is defined only in system_prompt.txt — LLM handles it.

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
    history_start_idx = st.session_state.get("_mode_history_start_idx", 1)
    history_messages = st.session_state.messages[history_start_idx:]
    # `user_input` was already appended to `st.session_state.messages` above; avoid duplicating it.
    if history_messages and history_messages[-1].get("role") == "user":
        history_messages = history_messages[:-1]

    for m in history_messages:
        if m["role"] == "user":
            history_lc.append(HumanMessage(content=m["content"]))
        else:
            history_lc.append(AIMessage(content=m["content"]))
    history_lc.append(HumanMessage(content=user_input))

    # Infer platform from this turn; once user says iPhone or Pixel, persist it for the rest of the session
    inferred_platform = infer_platform_from_messages_lc(history_lc)
    if inferred_platform in ("IOS18", "PIXEL"):
        st.session_state["confirmed_device"] = inferred_platform
        if mode == "tech":
            st.session_state["platform"] = (
                "iPhone" if inferred_platform == "IOS18" else "Pixel"
            )
    current_platform = (
        st.session_state.get("confirmed_device")
        or inferred_platform
    )

    tool_instruction = ""
    if mode == "tech":
        tool_instruction = (
            "\n\n## TOOL USE — MANDATORY FOR STEPS/HOW-TO\n"
            "When the user asks how to do something or asks for troubleshooting on their phone (e.g. change ringtone, wallpaper, settings, use an app, why is my internet slow), "
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

    messages_lc = [SystemMessage(content=system_with_platform)]
    if st.session_state.get("_mode_just_changed"):
        mode_label = "SCAM" if mode == "scam" else "TECH"
        messages_lc.append(
            SystemMessage(
                content=f"You are now in {mode_label} mode. Ignore unrelated previous context."
            )
        )
    messages_lc += history_lc

    # Persist docs_container across turns so that when support docs are fetched
    # in any tool call, the latest docs are available for PDF generation.
    docs_container: dict = st.session_state.get("docs_container", {})
    if docs_container is None:
        docs_container = {}
    # Prevent stale Tech-mode retrievals from showing a PDF in Scam mode.
    if mode == "scam":
        docs_container["docs"] = []

    search_tool = make_search_tool(vectorstore, docs_container, retriever_map=retriever_map)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([search_tool, verify_claim])

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            st.session_state["_processing"] = True
            with st.spinner("Checking..."):
                external_tool_used = False
                tool_retry_no_tool_calls = 0
                while True:
                    response = llm_with_tools.invoke(messages_lc)
                    messages_lc.append(response)

                    if not getattr(response, "tool_calls", None) or not response.tool_calls:
                        # Guardrail: if tool usage was expected, re-prompt once.
                        if mode == "scam" and tool_retry_no_tool_calls < 1:
                            tool_retry_no_tool_calls += 1
                            messages_lc.append(
                                SystemMessage(
                                    content="Tool call was not used. You MUST call verify_claim for scam/misinformation verification."
                                )
                            )
                            continue

                        if (
                            mode == "tech"
                            and tool_retry_no_tool_calls < 1
                            and user_input_raw is not None
                        ):
                            tech_step_keywords_required = [
                                "how",
                                "change",
                                "setup",
                                "internet",
                                "wifi",
                                "wi-fi",
                                "bluetooth",
                                "ringtone",
                                "settings",
                                "why",
                                "troubleshoot",
                                "slow",
                            ]
                            user_lower = (user_input_raw or "").lower()
                            needs_tech_tool = any(
                                k in user_lower for k in tech_step_keywords_required
                            )
                            if needs_tech_tool:
                                tool_retry_no_tool_calls += 1
                                messages_lc.append(
                                    SystemMessage(
                                        content="Tool call was not used. You MUST call search_support_docs for technical troubleshooting/how-to."
                                    )
                                )
                                continue

                        full_response = response.content or ""
                        break

                    scam_tool_result: str | None = None
                    preferred_tool_name = (
                        "verify_claim" if mode == "scam" else "search_support_docs"
                    )
                    executed_preferred_tool = False

                    for tc in response.tool_calls:
                        name = tc.get("name", "search_support_docs")
                        args = tc.get("args") or {}
                        tool_call_id = tc.get("id", "")

                        if name == preferred_tool_name and not executed_preferred_tool:
                            executed_preferred_tool = True
                            external_tool_used = True
                            if name == "search_support_docs":
                                result = search_tool.invoke(args)
                            elif name == "verify_claim":
                                result = verify_claim.invoke(args)
                                scam_tool_result = result
                            else:
                                result = f"Tool not implemented: {name}"
                        else:
                            # Enforce "only one tool call per user query" by not
                            # executing additional tools, even if the model requested them.
                            result = (
                                "Ignoring additional tool calls; only one tool call is allowed per user query."
                            )

                        messages_lc.append(
                            ToolMessage(tool_call_id=tool_call_id, content=result)
                        )

                        if scam_tool_result is not None:
                            # Return scam verification output verbatim (skip second LLM step).
                            full_response = scam_tool_result
                            break

                    if scam_tool_result is not None:
                        break

                    if mode == "scam" and not (full_response or "").strip():
                        full_response = "No verified fact-check results found."

            response_placeholder.markdown(full_response)

        except Exception as e:
            st.session_state["_processing"] = False
            st.error("Something went wrong while generating a response. Please try again.")
            st.exception(e)
            return

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state["_processing"] = False
    st.session_state["_mode_just_changed"] = False
    # Save latest docs_container for future turns
    st.session_state["docs_container"] = docs_container

    # If search_support_docs was called and returned docs, show PDF (trust the tool stream).
    # Scam mode should never display stale/irrelevant PDFs.
    retrieved_docs = docs_container.get("docs", [])
    if mode == "tech" and retrieved_docs:
        try:
            pdf_b64 = create_sources_pdf(retrieved_docs[:MAX_SOURCES_SHOWN])
        except Exception as e:
            pdf_b64 = ""
            with st.chat_message("assistant"):
                st.warning(
                    "Sources PDF could not be generated (WeasyPrint). "
                    "On Streamlit Cloud, ensure `packages.txt` includes Cairo/Pango deps — see DEPLOY.md."
                )
                with st.expander("Technical details"):
                    st.exception(e)

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