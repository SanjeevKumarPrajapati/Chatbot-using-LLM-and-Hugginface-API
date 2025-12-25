from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import streamlit as st
import tempfile
import time
import os

HF_TOKEN = os.environ["HF_TOKEN"]

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    huggingfacehub_api_token=HF_TOKEN, 
    task="text-generation",
    streaming=True,
)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. "
     "Answer ONLY using the provided document. "
     "If the answer is not in the document, say you don't know.\n\n"
     "Document:\n{context}"
    ),
    MessagesPlaceholder("messages")
])
model=ChatHuggingFace(llm=llm)
parser=StrOutputParser()
chain = prompt | model | parser


# ---------- Page config ----------
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.chat-container {
    max-width: 720px;
    margin: auto;
}

.user-bubble {
    background: linear-gradient(135deg, #4f46e5, #3b82f6);
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 80%;
    float: right;
    clear: both;
}

.bot-bubble {
    background: #f1f5f9;
    color: #111827;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 80%;
    float: left;
    clear: both;
}

.header {
    text-align: center;
    margin-bottom: 20px;
}

.header h1 {
    background: linear-gradient(to right, #6366f1, #22d3ee);
    -webkit-background-clip: text;
    color: transparent;
    font-weight: 800;
}

.typing {
    font-style: italic;
    color: #6b7280;
}
.file-upload-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 20px;  
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="header">
    <h1>🤖 AI Chatbot</h1>
    <p>Your intelligent assistant</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Customize your chatbot experience")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Your Chats")


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []   # UI memory
    
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""


if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
    
if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = [ # LLM memory
        SystemMessage(content="You are a helpful assistant.")
    ]


# ---------- Chat history ----------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="bot-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

def reset_doc():
    st.session_state.doc_text = ""
    st.session_state.doc_name = None
    st.session_state.lc_messages = [
        SystemMessage(content="You are a helpful assistant.")
    ]
    st.session_state.messages = []


uploaded_file = st.file_uploader(" ", type=["pdf", "docx", "txt"],key="doc_uploader" )

if uploaded_file:

    # 🔁 New document uploaded
    if st.session_state.doc_name != uploaded_file.name:
        reset_doc()
        st.session_state.doc_name = uploaded_file.name

        uploaded_file.seek(0)

        try:
            # ---------- PDF ----------
            if uploaded_file.name.lower().endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    path = tmp.name

                loader = PyPDFLoader(path)
                docs = loader.load()

                st.session_state.doc_text = "\n".join(
                    d.page_content for d in docs if d.page_content.strip()
                )

            # ---------- DOCX ----------
            elif uploaded_file.name.lower().endswith(".docx"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(uploaded_file.read())
                    path = tmp.name

                loader = Docx2txtLoader(path)
                docs = loader.load()

                st.session_state.doc_text = "\n".join(
                    d.page_content for d in docs if d.page_content.strip()
                )

            # ---------- TXT ----------
            elif uploaded_file.name.lower().endswith(".txt"):
                content = uploaded_file.read()
                st.session_state.doc_text = content.decode(
                    "utf-8", errors="ignore"
                )

            else:
                st.error("Unsupported file type")

            if not st.session_state.doc_text.strip():
                st.error("Document is empty or unreadable")
            else:
                st.success(f"📄 Now using: {uploaded_file.name}")

        except Exception as e:
            st.error(f"Failed to load document: {e}")

    else:
        st.info(f"📄 Using current document: {st.session_state.doc_name}")


        
# ---------- Chat input ----------
user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message in UI memory
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    # Save user message (LLM)
    st.session_state.lc_messages.append(
        HumanMessage(content=user_input)
    )
    # Rerender immediately
    st.rerun()

#previous chats
if "messages" in st.session_state and st.session_state.messages:
    user_prompts = [
        msg["content"]
        for msg in st.session_state.messages
        if msg["role"] == "user"
    ]

    if user_prompts:
        for i, prompt in enumerate(reversed(user_prompts[-10:]), 1):
            st.sidebar.markdown(
                f"**{i}.** {prompt[:60]}{'...' if len(prompt) > 60 else ''}"
            )
    else:
        st.sidebar.caption("No prompts yet")
else:
    st.sidebar.caption("No prompts yet")



# ---------- Generate bot response ----------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Thinking..."):
        #time.sleep(0.6)  # simulate thinking
        context = st.session_state.get("doc_text", "")
        bot_response = chain.invoke(
            {
                "messages": st.session_state.lc_messages,
                "context": context[:4000]
            }
        )

        # Typing effect
        placeholder = st.empty()
        typed_text = ""

        for char in bot_response:
            typed_text += char
            placeholder.markdown(
                f'<div class="bot-bubble">{typed_text}</div>',
                unsafe_allow_html=True
            )
            time.sleep(0.015)

        st.session_state.messages.append(
            {"role": "assistant", "content": bot_response}
        )
        
        # Save assistant response (LLM)
        st.session_state.lc_messages.append(
            AIMessage(content=bot_response)
        )
        time.sleep(0.1)
        st.rerun()

