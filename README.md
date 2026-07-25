# Quill AI — RAG notes chatbot

A chatbot that answers questions from your own notes instead of general
training data. Ask it something, it retrieves the relevant chunks from a
vector store, and answers using only that context.

Live demo: https://ragnoteschatbot.streamlit.app

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

The three scripts below read `OPENAI_API_KEY` from a `.env` file in this
folder. The Streamlit app (`app.py`) doesn't use `.env` at all — it asks each
visitor for their own key in the sidebar, so nobody's usage gets billed to
anyone else's account.

## The build, in order

**`manual_embedding.py`** — embeds one chunk of text by hand, no framework,
just to see what an embedding actually is (a list of ~1536 numbers).

**`build_vectorstore.py`** — loads the sample notes, chunks them, embeds
everything, and stores it in a local Chroma database (`./chroma_db`). Runs a
test query worded differently than the notes to confirm it's matching on
meaning, not keywords.

**`cli_chat.py`** — the full retrieve-then-generate loop, in the terminal.
Same logic the app uses, minus the UI.

**`app.py`** — the actual Streamlit app. Paste an API key, upload `.txt`/`.pdf`
notes (or use the bundled demo notes), ask questions, see which source chunks
the answer came from.

```bash
python manual_embedding.py
python build_vectorstore.py
python cli_chat.py
streamlit run app.py
```

## How the app differs from the CLI scripts

- **Session-only knowledge base** — each browser session gets its own
  in-memory Chroma collection, seeded with the demo notes. Nothing persists
  to disk, and one visitor's uploads never touch another's session.
- **Bring your own API key** — entered in the sidebar, kept in
  `st.session_state`, never written anywhere.
- **Model choice** — gpt-4o-mini, gpt-4o, or gpt-4.1-mini, picked per session.

## Files

| File | Purpose |
|---|---|
| `manual_embedding.py` | One manual embedding call, no framework |
| `build_vectorstore.py` | Build & query a Chroma store from `sample_notes/` |
| `cli_chat.py` | Full RAG loop, terminal only |
| `app.py` | The Streamlit app |
| `sample_notes/` | Demo notes (study techniques, productivity tips) |
| `chroma_db/` | Local vector DB used by the CLI scripts (not the app) |

## Tuning

- `chunk_size` / `chunk_overlap` (currently 500/50) — the biggest lever for
  answer quality. Too small loses context, too big dilutes relevance.
- `k`, the number of chunks retrieved (default 3 in the app, adjustable via
  the sidebar slider) — more context, but also more room for irrelevant text
  to confuse the answer.
- Next step worth exploring: LangGraph, to make the retrieve-then-generate
  flow an explicit graph instead of a straight-line function.
