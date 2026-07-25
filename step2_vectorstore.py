"""
STEP 2 — Store embeddings in a real vector database (Chroma) and search them.

Run:  python step2_vectorstore.py

What this does:
  1. Loads every .txt file in sample_notes/
  2. Splits each into chunks using LangChain's text splitter (not manual this time)
  3. Embeds every chunk and stores the vectors in a local Chroma database
  4. Runs a test query using DIFFERENT WORDING than the notes use, to prove
     it's searching by meaning, not exact keyword matching

The Chroma database is saved to disk in ./chroma_db so Step 3 and the
final app can reuse it without re-embedding everything each time.
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# --------------------------------------------------------------------
# 1. Load all .txt files from sample_notes/
# --------------------------------------------------------------------
loader = DirectoryLoader(
    "sample_notes",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
documents = loader.load()
print(f"Loaded {len(documents)} document(s).")

# --------------------------------------------------------------------
# 2. Split into chunks
#    chunk_size and chunk_overlap are the two knobs you'll tune later.
#    Too small = loses context. Too big = dilutes relevance.
#    These numbers are a reasonable starting point, not a rule.
# --------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# --------------------------------------------------------------------
# 3. Embed and store in Chroma (saved to disk in ./chroma_db)
# --------------------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
)
print("Chroma database built and saved to ./chroma_db")

# --------------------------------------------------------------------
# 4. Test retrieval with DIFFERENT wording than the notes use.
#    The notes say "active recall" and "spaced repetition" — we ask about
#    "the best way to remember what I studied" instead, to prove semantic
#    search works.
# --------------------------------------------------------------------
test_query = "What's the best way to remember what I studied?"

results = vectorstore.similarity_search(test_query, k=2)

print(f"\nQuery: {test_query}")
print(f"Top {len(results)} matching chunks:\n")
for i, doc in enumerate(results):
    print(f"--- Match {i+1} (from {doc.metadata.get('source')}) ---")
    print(doc.page_content[:200].replace("\n", " "))
    print()

# --------------------------------------------------------------------
# What to check:
#  - Did it return the Red-Black Tree chunk even though we never said
#    "red-black" or "self-balancing" in the query? If yes, semantic
#    search is working. If it returned something unrelated, your
#    chunk_size or chunk_overlap probably needs adjusting.
# --------------------------------------------------------------------
