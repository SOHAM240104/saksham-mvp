import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# Single DB (matches ingest.py); try app dir first, then cwd (for Streamlit Cloud / different launches)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_db_path():
    """Return first path where care_vector_db exists, else app dir (so error message shows where we looked)."""
    for base in (_APP_DIR, os.getcwd(), "."):
        p = os.path.join(base, "care_vector_db") if base != "." else "care_vector_db"
        p = os.path.abspath(p)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "chroma.sqlite3")):
            return p
    return os.path.join(_APP_DIR, "care_vector_db")

# Metadata values must match ingest.py platform names
PLATFORM_APPLE = "IOS18"
PLATFORM_GOOGLE = "PIXEL"


# -------------------------
# VECTORSTORE LOADER (single store, filter by metadata)
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
# BUILD STREAMING CHAIN (with optional metadata filter)
# -------------------------
def build_chain(vectorstore, platform_filter=None):
    with open("system_prompt.txt") as f:
        system_prompt = f.read()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
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

    search_kwargs = {"k": 8}
    if platform_filter is not None:
        search_kwargs["filter"] = {"platform": platform_filter}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return create_retrieval_chain(retriever, combine_docs_chain)


# -------------------------
# DEVICE DETECTION → platform for metadata filter
# -------------------------
def get_platform_filter(text):
    """Return PLATFORM_GOOGLE, PLATFORM_APPLE, or None (default to Apple docs)."""
    if not text:
        return None
    t = text.lower()
    if "pixel" in t or "google" in t:
        return PLATFORM_GOOGLE
    if "iphone" in t or "apple" in t:
        return PLATFORM_APPLE
    return None


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
                "content": "Hello 👋 Welcome to Saksham Support. I'm Care. I can help with iPhone (iOS 18) or Google Pixel, or guide you if something seems suspicious or scam-related. How can I help you today?",
            }
        ]
    if "last_platform" not in st.session_state:
        st.session_state.last_platform = PLATFORM_APPLE

    # Display previous messages (greeting is always the first message, then user can type)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message")

    if user_input is None:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Single vectorstore; route by metadata filter (platform)
    db_path = _get_db_path()
    vectorstore = load_vectorstore(db_path)
    if vectorstore is None:
        st.error("Support database not available.")
        st.code(db_path, language=None)
        st.info(
            "**Local:** In the project folder run `python ingest.py` once, then restart this app.\n\n"
            "**Streamlit Cloud:** Ensure the repo includes `care_vector_db` and redeploy, or set run command to: `python ingest.py && streamlit run app.py` (with Secrets)."
        )
        return

    # Retrieval filter: use explicit device from this message, else last known (no cross-platform bleed)
    platform_filter = get_platform_filter(user_input)
    if platform_filter is not None:
        st.session_state.last_platform = platform_filter
    else:
        platform_filter = st.session_state.last_platform

    chain = build_chain(vectorstore, platform_filter=platform_filter)

    # Limit history to last 8 messages
    recent_messages = st.session_state.messages[-8:]
    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in recent_messages]
    )
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        full_response = ""

        for chunk in chain.stream({
            "input": user_input,
            "history": history_text,
            "current_platform": st.session_state.last_platform,
        }):
            if "answer" in chunk:
                full_response += chunk["answer"]
                response_placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })


if __name__ == "__main__":
    main()