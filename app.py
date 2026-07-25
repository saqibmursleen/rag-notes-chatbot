"""
Second Brain — Streamlit app
------------------------------
Run with:  streamlit run app.py

Requires: step2_vectorstore.py run at least once already (so ./chroma_db exists),
OR just upload files through the sidebar here to build it fresh.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema import SystemMessage, HumanMessage

load_dotenv()

PERSIST_DIR = "chroma_db"
MODEL_NAME = "gpt-4o-mini"

# --------------------------------------------------------------------
# 1. SETUP — embeddings, LLM, vector store (loads existing DB if present)
# --------------------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")


@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


@st.cache_resource
def get_llm():
    return ChatOpenAI(model=MODEL_NAME, api_key=api_key)


@st.cache_resource
def get_vectorstore():
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())


def load_file_as_documents(uploaded_file) -> list:
    """Save an uploaded file to a temp path and load it with the right loader."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8")

    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = uploaded_file.name  # keep the real filename, not the temp path
    return docs


def add_documents_to_store(docs: list):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)


def list_indexed_sources() -> dict:
    """Return {filename: chunk_count} for everything currently in the store."""
    vectorstore = get_vectorstore()
    data = vectorstore.get()
    counts: dict = {}
    for meta in data.get("metadatas", []) or []:
        name = meta.get("source", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts


def reset_knowledge_base():
    vectorstore = get_vectorstore()
    ids = vectorstore.get().get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)


def answer_from_notes(question: str, k: int = 3) -> tuple[str, list]:
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(question, k=k)

    if not results:
        return "I don't have any notes to search yet — upload some files first.", []

    context = "\n\n---\n\n".join(doc.page_content for doc in results)
    system_prompt = (
        "You are a study assistant. Answer the user's question using ONLY the "
        "context below, taken from their own notes. If the answer isn't in the "
        "context, say so clearly instead of guessing.\n\n"
        f"CONTEXT:\n{context}"
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    output = get_llm().invoke(messages)
    return output.content, results


# --------------------------------------------------------------------
# 2. PAGE CONFIG + STYLING
# --------------------------------------------------------------------
st.set_page_config(page_title="Second Brain", page_icon="🧠", layout="centered")

st.markdown(
    """
    <style>
        .stApp { background-color: #0f172a; }
        .block-container { padding-top: 2rem; max-width: 720px; }
        section[data-testid="stChatMessage"] { border-radius: 14px; }
        [data-testid="stSidebar"] { background-color: #111827; }
        .stat-card {
            background: #1e293b;
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 8px;
            border: 1px solid #334155;
        }
        .stat-card .label { color: #94a3b8; font-size: 0.75rem; }
        .stat-card .value { color: white; font-size: 1.1rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h2 style='color:white; margin-bottom:0;'>🧠 Second Brain</h2>"
    "<p style='color:#94a3b8; margin-top:4px;'>Ask questions about your own notes</p>",
    unsafe_allow_html=True,
)
st.divider()

if not api_key:
    st.error(
        "No OPENAI_API_KEY found. Add it to a `.env` file locally, or to your "
        "deployment's secrets if this is running remotely.",
        icon="🔑",
    )
    st.stop()

# --------------------------------------------------------------------
# 3. SIDEBAR — knowledge base management
# --------------------------------------------------------------------
with st.sidebar:
    st.subheader("Knowledge base")

    indexed = list_indexed_sources()
    total_chunks = sum(indexed.values())

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="label">Files indexed</div>
            <div class="value">{len(indexed)}</div>
        </div>
        <div class="stat-card">
            <div class="label">Chunks stored</div>
            <div class="value">{total_chunks}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if indexed:
        with st.expander(f"📚 View indexed files ({len(indexed)})"):
            for name, count in sorted(indexed.items()):
                st.caption(f"📄 {name} — {count} chunk(s)")

    st.divider()
    st.subheader("Add to your knowledge base")
    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf notes",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Add these files", use_container_width=True):
        total_new_chunks = 0
        with st.spinner("Indexing..."):
            for f in uploaded_files:
                docs = load_file_as_documents(f)
                total_new_chunks += add_documents_to_store(docs)
        st.success(f"Added {len(uploaded_files)} file(s), {total_new_chunks} chunks indexed.")
        st.rerun()

    st.caption(
        "Notes persist in ./chroma_db between runs — you don't need to "
        "re-upload every time you restart the app."
    )

    st.divider()
    k = st.slider("Chunks retrieved per question (k)", min_value=1, max_value=10, value=3)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Ask me anything about your notes."}
            ]
            st.rerun()
    with col2:
        if st.button("⚠️ Reset KB", use_container_width=True):
            reset_knowledge_base()
            st.rerun()

# --------------------------------------------------------------------
# 4. SESSION STATE
# --------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your notes."}
    ]

# --------------------------------------------------------------------
# 5. RENDER EXISTING MESSAGES
# --------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "🧠" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# --------------------------------------------------------------------
# 6. HANDLE NEW INPUT
# --------------------------------------------------------------------
user_input = st.chat_input("Ask a question about your notes...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Searching your notes..."):
            try:
                reply, sources = answer_from_notes(user_input, k=k)
            except Exception as e:
                reply, sources = f"Something went wrong: {e}", []
        st.write(reply)

        if sources:
            with st.expander(f"📎 Sources used ({len(sources)})"):
                for doc in sources:
                    st.caption(f"📄 {doc.metadata.get('source', 'unknown')}")
                    st.text(doc.page_content[:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": reply})
