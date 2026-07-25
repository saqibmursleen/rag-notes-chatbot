from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

with open("sample_notes/study_techniques.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

print(f"Split the file into {len(chunks)} chunks.\n")
for i, chunk in enumerate(chunks):
    preview = chunk.replace("\n", " ")[:80]
    print(f"  Chunk {i}: {preview}...")

first_chunk = chunks[0]
print(f"\nEmbedding chunk 0:\n---\n{first_chunk}\n---\n")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=first_chunk,
)

vector = response.data[0].embedding
print(f"Embedding created. Vector length: {len(vector)}")
print(f"First 10 numbers of the vector: {vector[:10]}")
