"""
Second Brain — Streamlit app
------------------------------
Run with:  streamlit run app.py

Requires: step2_vectorstore.py run at least once already (so ./chroma_db exists),
OR just upload files through the sidebar here to build it fresh.

Each visitor supplies their own OpenAI API key in the sidebar — the key lives
only in their browser session (st.session_state), is never written to disk,
and is not the key used to build/host this app.
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
# 1. SETUP — embeddings, LLM, vector store, all keyed by the caller's
#    own API key so nobody's requests are billed to anyone else's account.
# --------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings(api_key: str):
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    return ChatOpenAI(model=MODEL_NAME, api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_vectorstore(api_key: str):
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings(api_key))


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


def add_documents_to_store(docs: list, api_key: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = get_vectorstore(api_key)
    vectorstore.add_documents(chunks)
    return len(chunks)


def list_indexed_sources(api_key: str) -> dict:
    """Return {filename: chunk_count} for everything currently in the store."""
    vectorstore = get_vectorstore(api_key)
    data = vectorstore.get()
    counts: dict = {}
    for meta in data.get("metadatas", []) or []:
        name = meta.get("source", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts


def reset_knowledge_base(api_key: str):
    vectorstore = get_vectorstore(api_key)
    ids = vectorstore.get().get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)


def answer_from_notes(question: str, api_key: str, k: int = 3) -> tuple[str, list]:
    vectorstore = get_vectorstore(api_key)
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
    output = get_llm(api_key).invoke(messages)
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

# --------------------------------------------------------------------
# 3. SIDEBAR — API key + knowledge base management
# --------------------------------------------------------------------
with st.sidebar:
    st.subheader("Your OpenAI API key")
    st.session_state.setdefault("user_api_key", os.getenv("OPENAI_API_KEY", ""))
    st.text_input(
        "API key",
        type="password",
        key="user_api_key",
        placeholder="sk-...",
        help="Get one at platform.openai.com/api-keys. Used only for your own "
        "session — never stored or sent anywhere else.",
        label_visibility="collapsed",
    )
    st.caption("Your key stays in this browser session only. It's never saved to disk.")
    st.divider()

    api_key = st.session_state.user_api_key.strip()

    if api_key:
        st.subheader("Knowledge base")

        indexed = list_indexed_sources(api_key)
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
                    total_new_chunks += add_documents_to_store(docs, api_key)
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
                reset_knowledge_base(api_key)
                st.rerun()

if not api_key:
    st.info(
        "👈 Paste your OpenAI API key in the sidebar to get started. "
        "It's used only for your own questions and file uploads — this app "
        "doesn't have (or use) a shared key.",
        icon="🔑",
    )
    st.stop()

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
                reply, sources = answer_from_notes(user_input, api_key, k=k)
            except Exception as e:
                reply, sources = f"Something went wrong: {e}", []
        st.write(reply)

        if sources:
            with st.expander(f"📎 Sources used ({len(sources)})"):
                for doc in sources:
                    st.caption(f"📄 {doc.metadata.get('source', 'unknown')}")
                    st.text(doc.page_content[:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": reply})
