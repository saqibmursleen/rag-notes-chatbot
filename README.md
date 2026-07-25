# Second Brain — RAG project

A step-by-step build of a "second brain" chatbot that answers questions using
your own notes, instead of just general knowledge.

## Setup (do this once)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Put your `OPENAI_API_KEY` in a `.env` file in this folder (same one from Week02).

## Run order — do these IN ORDER, don't skip to the app

### Step 1 — see what an embedding is
```
python step1_manual_embedding.py
```
No database, no search. Just proves to you what "embedding a chunk of text"
actually produces (a list of ~1536 numbers).

### Step 2 — build a real vector store and test search
```
python step2_vectorstore.py
```
Loads both sample note files, chunks them, embeds them, stores them in a local
Chroma database (`./chroma_db`), then runs a test query using different
wording than the notes to prove it searches by meaning, not keywords.

### Step 3 — full RAG loop in the terminal
```
python step3_rag_chat_cli.py
```
Ask questions about the sample notes (Red-Black Trees, Dijkstra's, DBMS
normalization) right in the terminal. This is the same retrieval + generation
logic the final app uses, just without a UI, so you can see it clearly.

### Step 4 — the actual app
```
streamlit run app.py
```
Full UI: upload your own `.txt`/`.pdf` notes from the sidebar, ask questions
in the chat, see which source chunks the answer was pulled from.

## Files

| File | Purpose |
|---|---|
| `step1_manual_embedding.py` | Manual chunking + one embedding call, no framework |
| `step2_vectorstore.py` | Build & query a Chroma vector store from `sample_notes/` |
| `step3_rag_chat_cli.py` | Full RAG chat loop, terminal only |
| `app.py` | Streamlit UI — upload notes, chat, see sources |
| `sample_notes/` | Two test files (DAA, DBMS) so you have something to query immediately |
| `chroma_db/` | Created automatically after Step 2 — your vector database lives here |

## Things worth tuning once it's working

- **`chunk_size` / `chunk_overlap`** in the text splitter (currently 500/50) —
  the single biggest lever for answer quality. Too small loses context, too
  big dilutes relevance. Test with your own notes and adjust.
- **`k`** (how many chunks get retrieved, currently 3) — more chunks means
  more context but also more chance of irrelevant text confusing the answer.
- Once this feels natural, look into **LangGraph** to formalize the
  retrieve-then-generate steps as an explicit graph instead of the straight-line
  function it is now — that's the natural next step toward multi-agent systems.
