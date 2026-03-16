"""
agent_core.py
=============
Decoupled AI logic and state management for Saksham MVP RAG.
Used by Streamlit (app.py).
# Evolution API / server.py integration commented out; Streamlit-only.
"""
import os

# Avoid OpenMP duplicate library crash on macOS (FAISS/numpy/sentence-transformers)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import List, Optional, Any, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.tools import tool
from pydantic import ConfigDict

load_dotenv()

# -------------------------
# CONSTANTS
# -------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
PLATFORM_APPLE = "IOS18"
PLATFORM_GOOGLE = "PIXEL"
MAX_INPUT_LENGTH = 2000
HISTORY_WINDOW = 8
ENSEMBLE_K = 20
RERANKER_TOP_N = 5
CRAG_RELEVANCE_THRESHOLD = 0.3
RRF_K = 60

GREETING = (
    "Hi there 👋 Welcome to Saksham Support. I'm here to help with your "
    "technical issues — or to listen if something feels off or scam-related. "
    "How can I help you today?"
)

# Session store for Evolution API / server.py (commented out — Streamlit-only)
# chat_histories: Dict[str, Dict[str, Any]] = {}


# -------------------------
# DB PATH & PROMPT
# -------------------------
def _get_db_path():
    for base in (_APP_DIR, os.getcwd(), "."):
        for sub in ("combined", "ios18", "pixel"):
            p = os.path.abspath(os.path.join(base, "care_vector_db", sub))
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "index.faiss")):
                return p
    return os.path.join(_APP_DIR, "care_vector_db", "combined")


def load_system_prompt() -> str:
    path = os.path.join(_APP_DIR, "system_prompt.txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


# -------------------------
# VECTORSTORE
# -------------------------
def load_vectorstore(db_path: str):
    if not os.path.exists(db_path):
        return None
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# -------------------------
# PLATFORM FILTERED RETRIEVER
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
# MMR RETRIEVER
# -------------------------
MMR_FETCH_K = 20
MMR_LAMBDA_MULT = 0.5


def build_retriever(_vectorstore, platform_filter=None):
    base = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RERANKER_TOP_N,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA_MULT,
        },
    )
    if platform_filter is not None:
        return PlatformFilteredRetriever(base=base, platform=platform_filter)
    return base


# -------------------------
# CRAG
# -------------------------
def _are_docs_relevant(query: str, docs: List[Document]) -> bool:
    if not docs or not query.strip():
        return False
    context = "\n\n".join(
        f"[Passage {i+1}]:\n{d.page_content[:800]}" for i, d in enumerate(docs[:5])
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = (
        "Does any of the following support-doc passages directly answer or clearly address "
        "this user question? Answer only YES or NO.\n\n"
        f"User question: {query[:500]}\n\nPassages:\n{context[:4000]}"
    )
    try:
        out = llm.invoke(prompt)
        text = (out.content or "").strip().upper()
        return text.startswith("YES")
    except Exception:
        return len(docs) > 0


def _rewrite_query(query: str) -> str:
    if not query.strip():
        return query
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = (
        "Rewrite this user question into a single clear search query for phone support documentation. "
        "Fix typos, expand obvious abbreviations (e.g. wifi → Wi-Fi), keep it concise. "
        "Output only the rewritten query, nothing else.\n\nUser question: " + query[:500]
    )
    try:
        out = llm.invoke(prompt)
        rewritten = (out.content or "").strip()
        return rewritten if rewritten else query
    except Exception:
        return query


# -------------------------
# SEARCH TOOL
# -------------------------
def make_search_tool(vectorstore, docs_container: dict, retriever_map: Optional[Dict[str, Any]] = None):
    """retriever_map: optional dict with keys None, 'IOS18', 'PIXEL' for pre-built retrievers."""

    @tool
    def search_support_docs(query: str, platform: str = "") -> str:
        """Search iPhone and Pixel support documentation. You MUST call this before giving ANY steps or how-to instructions (e.g. change ringtone, wallpaper, settings). The tool returns the only allowed source for your answer. Do NOT call for greetings, thanks, or scam mentions. query: the user's question or topic. platform: '' for both, or 'IOS18' for iPhone, 'PIXEL' for Pixel when the user has said which device."""
        platform_key = platform if platform in ("IOS18", "PIXEL") else None
        if retriever_map is not None:
            retriever = retriever_map.get(platform_key) or retriever_map.get(None)
        else:
            retriever = None
        if retriever is None:
            retriever = build_retriever(vectorstore, platform_key)
        docs = retriever.invoke(query, config={"run_name": "search_support_docs"})
        docs = (docs or [])[:RERANKER_TOP_N]
        docs_container["docs"] = docs
        if not docs:
            return "No relevant documentation found."
        for i, d in enumerate(docs, 1):
            d.metadata["source_index"] = i
        return "\n\n---\n\n".join(
            [f"Source {i}:\n{d.page_content}" for i, d in enumerate(docs, 1)]
        )

    return search_support_docs


# -------------------------
# PLATFORM INFERENCE (from messages_lc)
# -------------------------
def infer_platform_from_messages_lc(messages_lc: list) -> str:
    for m in reversed(messages_lc):
        if not isinstance(m, HumanMessage):
            continue
        content = (getattr(m, "content", None) or "").lower()
        if "pixel" in content or "google" in content:
            return PLATFORM_GOOGLE
        if "iphone" in content or "apple" in content or "ios" in content:
            return PLATFORM_APPLE
    return "NOT_CONFIRMED"


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
# Evolution API / server.py integration (commented out — Streamlit-only)
# -------------------------
# def get_last_docs(user_id: str) -> List[Document]:
#     """Return the last retrieved docs for this user (for PDF generation / test UI)."""
#     if user_id not in chat_histories:
#         return []
#     return (chat_histories[user_id].get("docs_container") or {}).get("docs") or []
#
#
# def _get_vectorstore():
#     db_path = _get_db_path()
#     return load_vectorstore(db_path)
#
#
# def get_agent_response(user_id: str, user_message: str) -> str:
#     """
#     Run the RAG agent for one user message; update in-memory history.
#     Returns the assistant's text response.
#     Used by Evolution API webhook (server.py).
#     """
#     user_message = sanitize_input(user_message)
#     if not user_message:
#         return ""
#
#     vectorstore = _get_vectorstore()
#     if vectorstore is None:
#         return "Support database is not available. Please try again later."
#
#     system_prompt = load_system_prompt()
#     if not system_prompt:
#         return "System configuration error. Please try again later."
#
#     if user_id not in chat_histories:
#         tool_instruction = (
#             "\n\n## TOOL USE — MANDATORY FOR STEPS/HOW-TO\n"
#             "When the user asks how to do something on their phone (e.g. change ringtone, wallpaper, settings), "
#             "you MUST call the search_support_docs tool first. Use the query as the user's question and set platform to "
#             "'IOS18' or 'PIXEL' if the user has already said which device. Answer ONLY from the tool result. "
#             "Never give steps from memory — if you do not call the tool, do not give steps."
#         )
#         system_with_platform = (
#             system_prompt + tool_instruction + "\n\nCurrent device context: NOT_CONFIRMED"
#         )
#         chat_histories[user_id] = {
#             "messages_lc": [
#                 SystemMessage(content=system_with_platform),
#                 AIMessage(content=GREETING),
#             ],
#             "docs_container": {},
#         }
#
#     messages_lc = chat_histories[user_id]["messages_lc"]
#     docs_container = chat_histories[user_id]["docs_container"]
#
#     messages_lc.append(HumanMessage(content=user_message))
#
#     inferred = infer_platform_from_messages_lc(messages_lc)
#     tool_instruction = (
#         "\n\n## TOOL USE — MANDATORY FOR STEPS/HOW-TO\n"
#         "When the user asks how to do something on their phone (e.g. change ringtone, wallpaper, settings), "
#         "you MUST call the search_support_docs tool first. Use the query as the user's question and set platform to "
#         "'IOS18' or 'PIXEL' if the user has already said which device. Answer ONLY from the tool result. "
#         "Never give steps from memory — if you do not call the tool, do not give steps."
#     )
#     system_with_platform = (
#         system_prompt
#         + tool_instruction
#         + "\n\nCurrent device context (use for steps and filtering): "
#         + inferred
#     )
#     messages_lc[0] = SystemMessage(content=system_with_platform)
#
#     search_tool = make_search_tool(vectorstore, docs_container)
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#     llm_with_tools = llm.bind_tools([search_tool])
#
#     full_response = ""
#     try:
#         while True:
#             response = llm_with_tools.invoke(messages_lc)
#             messages_lc.append(response)
#
#             if not getattr(response, "tool_calls", None):
#                 full_response = response.content or ""
#                 break
#
#             for tc in response.tool_calls:
#                 name = tc.get("name", "search_support_docs")
#                 args = tc.get("args") or {}
#                 tool_call_id = tc.get("id", "")
#                 result = search_tool.invoke(args)
#                 messages_lc.append(
#                     ToolMessage(tool_call_id=tool_call_id, content=result)
#                 )
#
#     except Exception:
#         full_response = "Something went wrong while generating a response. Please try again."
#
#     chat_histories[user_id]["messages_lc"] = messages_lc
#     return full_response
