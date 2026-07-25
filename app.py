"""
Second Brain — Streamlit app
------------------------------
Run with:  streamlit run app.py

Each visitor supplies their own OpenAI API key in the sidebar — the key lives
only in their browser session (st.session_state), is never written to disk,
and is not the key used to build/host this app.

The knowledge base is also session-only: it lives in an in-memory Chroma
collection scoped to that browser tab, seeded with a couple of demo notes so
there's something to ask about immediately. Closing the tab clears it —
nobody's uploads or questions leak into anyone else's session.
"""

import glob
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

SAMPLE_NOTES_DIR = "sample_notes"
MODEL_NAME = "gpt-4o-mini"
SUGGESTED_QUESTIONS = [
    "What's active recall?",
    "How does the Pomodoro technique work?",
    "What's the two-minute rule?",
]

# --------------------------------------------------------------------
# 1. SETUP — embeddings and LLM are cached per API key (so nobody's
#    requests are billed to anyone else's account). The vector store is
#    NOT cached globally — it lives in st.session_state, one fresh
#    in-memory collection per browser session.
# --------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings(api_key: str):
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    return ChatOpenAI(model=MODEL_NAME, api_key=api_key)


def new_vectorstore(api_key: str) -> Chroma:
    """A fresh, in-memory (non-persisted) collection — isolated to this session."""
    return Chroma(embedding_function=get_embeddings(api_key))


def get_vectorstore(api_key: str) -> Chroma:
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = new_vectorstore(api_key)
        seed_sample_notes(api_key)
    return st.session_state.vectorstore


def seed_sample_notes(api_key: str):
    """Auto-load the bundled .txt demo notes so a fresh session isn't empty."""
    txt_paths = sorted(glob.glob(os.path.join(SAMPLE_NOTES_DIR, "*.txt")))
    for path in txt_paths:
        docs = TextLoader(path, encoding="utf-8").load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
        add_documents_to_store(docs, api_key)


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


def reset_knowledge_base(api_key: str, reseed: bool = False):
    st.session_state.vectorstore = new_vectorstore(api_key)
    if reseed:
        seed_sample_notes(api_key)


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
        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(139, 92, 246, 0.16), transparent 40%),
                radial-gradient(circle at 85% 10%, rgba(56, 189, 248, 0.12), transparent 40%),
                #0b1120;
        }
        .block-container { padding-top: 2.2rem; max-width: 740px; }
        section[data-testid="stChatMessage"] {
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.12);
        }
        [data-testid="stSidebar"] { background-color: #0f172a; }
        [data-testid="stChatInput"] textarea { border-radius: 12px; }

        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0;
            background: linear-gradient(90deg, #f8fafc 0%, #c7d2fe 60%, #7dd3fc 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .hero-sub { color: #94a3b8; margin-top: 6px; font-size: 0.95rem; }
        .badge-row { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }
        .badge {
            display: inline-block;
            font-size: 0.72rem;
            color: #c7d2fe;
            background: rgba(139, 92, 246, 0.14);
            border: 1px solid rgba(139, 92, 246, 0.35);
            border-radius: 999px;
            padding: 3px 10px;
        }

        .stat-card {
            background: linear-gradient(145deg, #1e293b, #17202f);
            border-radius: 12px;
            padding: 10px 14px;
            margin-bottom: 8px;
            border: 1px solid #334155;
        }
        .stat-card .label { color: #94a3b8; font-size: 0.75rem; }
        .stat-card .value { color: white; font-size: 1.15rem; font-weight: 700; }

        div[data-testid="stButton"] > button {
            border-radius: 999px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-title">🧠 Second Brain</div>
    <div class="hero-sub">Ask questions about your notes — answered strictly from what you gave it, nothing invented.</div>
    <div class="badge-row">
        <span class="badge">🔒 Session-only knowledge base</span>
        <span class="badge">🔑 Bring your own API key</span>
        <span class="badge">⚡ gpt-4o-mini</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# --------------------------------------------------------------------
# 3. SIDEBAR — API key + knowledge base management
# --------------------------------------------------------------------
with st.sidebar:
    st.subheader("🔑 Your OpenAI API key")
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
    st.caption("Stays in this browser session only — never saved to disk.")
    st.divider()

    api_key = st.session_state.user_api_key.strip()

    if api_key:
        st.subheader("📚 Knowledge base")
        st.caption("Session-only — resets when you close this tab.")

        indexed = list_indexed_sources(api_key)
        total_chunks = sum(indexed.values())

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""<div class="stat-card"><div class="label">Files</div>
                <div class="value">{len(indexed)}</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="stat-card"><div class="label">Chunks</div>
                <div class="value">{total_chunks}</div></div>""",
                unsafe_allow_html=True,
            )

        if indexed:
            with st.expander(f"View indexed files ({len(indexed)})"):
                for name, count in sorted(indexed.items()):
                    st.caption(f"📄 {name} — {count} chunk(s)")

        st.divider()
        st.subheader("➕ Add notes")
        uploaded_files = st.file_uploader(
            "Upload .txt or .pdf notes",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files and st.button("Add these files", use_container_width=True):
            total_new_chunks = 0
            with st.spinner("Indexing..."):
                for f in uploaded_files:
                    docs = load_file_as_documents(f)
                    total_new_chunks += add_documents_to_store(docs, api_key)
            st.success(f"Added {len(uploaded_files)} file(s), {total_new_chunks} chunks indexed.")
            st.rerun()

        st.divider()
        k = st.slider("Chunks retrieved per question (k)", min_value=1, max_value=10, value=3)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Chat", use_container_width=True, help="Clear the conversation"):
                st.session_state.messages = [
                    {"role": "assistant", "content": "Ask me anything about your notes."}
                ]
                st.rerun()
        with col2:
            if st.button("♻️ Reload demo", use_container_width=True, help="Reset to the bundled demo notes"):
                reset_knowledge_base(api_key, reseed=True)
                st.rerun()
        with col3:
            if st.button("⚠️ Wipe", use_container_width=True, help="Remove everything from the knowledge base"):
                reset_knowledge_base(api_key, reseed=False)
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
get_vectorstore(api_key)  # ensures the session's KB exists (and is seeded) before first render

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your notes — try one of the prompts below, or type your own."}
    ]

# --------------------------------------------------------------------
# 5. RENDER EXISTING MESSAGES
# --------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "🧠" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Quick-start suggestion chips — only show before the conversation gets going
pending_question = None
if len(st.session_state.messages) == 1:
    st.caption("Try asking:")
    chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, question in zip(chip_cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(question, use_container_width=True, key=f"chip_{question}"):
                pending_question = question

# --------------------------------------------------------------------
# 6. HANDLE NEW INPUT
# --------------------------------------------------------------------
user_input = st.chat_input("Ask a question about your notes...") or pending_question

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
