"""
STEP 3 — Wire retrieval into your chatbot pattern. Terminal only, no UI yet.

Run:  python step3_rag_chat_cli.py

Requires: you've already run step2_vectorstore.py at least once, so
./chroma_db exists on disk.

What this does, per question you type:
  1. Embeds your question
  2. Retrieves the most relevant chunks from Chroma
  3. Stuffs those chunks into the SYSTEM prompt as context
  4. Sends that + your question to the LLM using the same
     SystemMessage/HumanMessage/AIMessage pattern from your langchain.ipynb
  5. Prints the answer — grounded in YOUR notes, not general training data
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.schema import SystemMessage, HumanMessage

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini")


def answer_from_notes(question: str, k: int = 3) -> str:
    # 1. Retrieve relevant chunks
    results = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc in results)

    # 2. Build a system prompt that FORCES the model to use only this context
    system_prompt = (
        "You are a study assistant. Answer the user's question using ONLY the "
        "context below, taken from their own notes. If the answer isn't in the "
        "context, say so clearly instead of guessing.\n\n"
        f"CONTEXT:\n{context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    output = llm.invoke(messages)
    return output.content


if __name__ == "__main__":
    print("Second Brain — ask questions about your notes. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer = answer_from_notes(question)
        print(f"\nBot: {answer}\n")
