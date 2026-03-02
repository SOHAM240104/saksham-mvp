import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
# Use Streamlit Cloud secrets when available (deployment)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

APPLE_DB = "./chroma_db_apple"
GOOGLE_DB = "./chroma_db_google"

STRICT_FALLBACK = "I apologize, but I don't have enough specific information in the current support manuals to answer that."

UNSUPPORTED_BRANDS = ["samsung", "huawei", "oneplus", "xiaomi", "oppo", "vivo"]


# -------------------------
# RAG BUILDER
# -------------------------
@st.cache_resource
def build_rag_chain(db_path):

    if not os.path.exists(db_path):
        return None

    with open("system_prompt.txt") as f:
        system_prompt = f.read()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(system_prompt)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 8}),
        combine_docs_chain
    )


# -------------------------
# STATE INIT
# -------------------------
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.state = "INTENT"
        st.session_state.device = None
        st.session_state.model = None

        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Hello. I'm here to help you.\n\n"
                "Is this about your phone, or something that may be a scam?"
            )
        })


# -------------------------
# HELPERS
# -------------------------
def detect_scam(user_input):
    return "scam" in user_input.lower()


def detect_device(user_input):
    text = user_input.lower()
    if "iphone" in text:
        return "APPLE"
    if "pixel" in text:
        return "GOOGLE"
    return None


def detect_unsupported(user_input):
    text = user_input.lower()
    return any(brand in text for brand in UNSUPPORTED_BRANDS)


def unsure_model(user_input):
    text = user_input.lower()
    unsure_words = ["don't know", "not sure", "no idea", "how to check"]
    return any(word in text for word in unsure_words)


def is_unclear(text):
    text = text.strip().lower()
    if len(text) < 4:
        return True
    vague = ["help", "problem", "issue", "not working", "it", "this"]
    return text in vague


# -------------------------
# MAIN
# -------------------------
def main():

    st.set_page_config(page_title="Care Assistant", page_icon="💛")
    st.title("Care Assistant")

    init_state()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("You can type your message here")

    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    response = ""

    # -------------------------
    # SCAM HANDLING
    # -------------------------
    if detect_scam(user_input):
        response = (
            "Thank you for telling me.\n\n"
            "We are currently working on improving our scam support feature.\n"
            "It will be available very soon.\n\n"
            "If you would like help with your phone, I would be happy to assist."
        )

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
        return

    # -------------------------
    # INTENT STAGE
    # -------------------------
    if st.session_state.state == "INTENT":

        if "phone" in user_input.lower():
            st.session_state.state = "DEVICE"
            response = "Thank you. Are you using an iPhone or a Google Pixel?"
        else:
            response = "I just want to make sure I understand you. Is this about your phone?"

    # -------------------------
    # DEVICE STAGE
    # -------------------------
    elif st.session_state.state == "DEVICE":

        if detect_unsupported(user_input):
            response = (
                "Thank you for letting me know.\n\n"
                "At the moment, I am designed to support iPhone and Google Pixel devices.\n\n"
                "If you are using one of those, please tell me."
            )
        else:
            device = detect_device(user_input)

            if device:
                st.session_state.device = device
                st.session_state.state = "MODEL"

                response = (
                    "Thank you.\n\n"
                    "Do you know the model of the phone?\n\n"
                    "For example: iPhone 14, iPhone 13 Pro, Pixel 7, Pixel 8."
                )
            else:
                response = "Are you using an iPhone or a Google Pixel?"

    # -------------------------
    # MODEL STAGE
    # -------------------------
    elif st.session_state.state == "MODEL":

        if unsure_model(user_input):

            if st.session_state.device == "APPLE":
                response = (
                    "That's okay. Let me guide you.\n\n"
                    "1. Open the Settings app.\n"
                    "2. Tap 'General'.\n"
                    "3. Tap 'About'.\n"
                    "4. Look for 'Model Name'.\n\n"
                    "Please tell me what it says there."
                )
            else:
                response = (
                    "That's okay. Let me guide you.\n\n"
                    "1. Open the Settings app.\n"
                    "2. Scroll down and tap 'About phone'.\n"
                    "3. Look for 'Device name' or 'Model'.\n\n"
                    "Please tell me what it says there."
                )
        else:
            st.session_state.model = user_input
            st.session_state.state = "ISSUE"
            response = "Thank you. Now please tell me what issue the phone is having."

    # -------------------------
    # ISSUE STAGE
    # -------------------------
    elif st.session_state.state == "ISSUE":

        if is_unclear(user_input):
            response = (
                "That's perfectly okay.\n\n"
                "Could you describe what is happening in a little more detail?\n\n"
                "For example:\n"
                "- Is something not turning on?\n"
                "- Is the battery draining quickly?\n"
                "- Is Wi-Fi not connecting?"
            )
        else:
            db = APPLE_DB if st.session_state.device == "APPLE" else GOOGLE_DB
            rag_chain = build_rag_chain(db)

            if rag_chain is None:
                response = (
                    "Support data for this device is not available right now. "
                    "Please try again later or contact support."
                )
            else:
                with st.spinner("Let me check the official instructions for you..."):
                    result = rag_chain.invoke({"input": user_input})

                answer = result.get("answer", "").strip()

                if answer == STRICT_FALLBACK or not answer:
                    response = (
                        "I couldn't find clear instructions for that in the official guide.\n\n"
                        "Could you describe what you see on the screen?"
                    )
                else:
                    response = (
                        "Here are the steps that may help:\n\n"
                        f"{answer}\n\n"
                        "If something is still not working, please tell me what is happening now."
                    )

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.write(response)


if __name__ == "__main__":
    main()