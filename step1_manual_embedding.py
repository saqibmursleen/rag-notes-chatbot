"""
STEP 1 — See what an embedding actually IS, with no framework hiding it.

Run:  python step1_manual_embedding.py

What this does:
  1. Reads one of your note files
  2. Splits it into paragraph-sized chunks (manually, no library)
  3. Sends ONE chunk to OpenAI's embedding model
  4. Prints the resulting vector so you can see its shape

Nothing here is "smart" yet — no search, no storage. The goal is purely
to demystify what "embedding a chunk of text" means before Step 2 wraps
this in a database.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# --------------------------------------------------------------------
# 1. Read the file
# --------------------------------------------------------------------
with open("sample_notes/daa_notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --------------------------------------------------------------------
# 2. Chunk it manually — split on blank lines (paragraph breaks)
# --------------------------------------------------------------------
chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

print(f"Split the file into {len(chunks)} chunks.\n")
for i, chunk in enumerate(chunks):
    preview = chunk.replace("\n", " ")[:80]
    print(f"  Chunk {i}: {preview}...")

# --------------------------------------------------------------------
# 3. Embed just the FIRST chunk, so you can inspect one vector closely
# --------------------------------------------------------------------
first_chunk = chunks[0]

print(f"\nEmbedding chunk 0:\n---\n{first_chunk}\n---\n")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=first_chunk,
)

vector = response.data[0].embedding

print(f"Embedding created. Vector length: {len(vector)}")
print(f"First 10 numbers of the vector: {vector[:10]}")

# --------------------------------------------------------------------
# What to notice:
#  - The vector has ~1536 numbers, regardless of how long the text was.
#  - These numbers encode MEANING. Two chunks about similar topics will
#    have vectors that are mathematically close together (small distance).
#  - That's the entire trick behind "search by meaning" — Step 2 stores
#    many of these vectors and finds the closest one to your question.
# --------------------------------------------------------------------
