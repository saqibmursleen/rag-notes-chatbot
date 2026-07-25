from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

loader = DirectoryLoader(
    "sample_notes",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
documents = loader.load()
print(f"Loaded {len(documents)} document(s).")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
)
print("Chroma database built and saved to ./chroma_db")

test_query = "What's the best way to remember what I studied?"
results = vectorstore.similarity_search(test_query, k=2)

print(f"\nQuery: {test_query}")
print(f"Top {len(results)} matching chunks:\n")
for i, doc in enumerate(results):
    print(f"--- Match {i+1} (from {doc.metadata.get('source')}) ---")
    print(doc.page_content[:200].replace("\n", " "))
    print()
