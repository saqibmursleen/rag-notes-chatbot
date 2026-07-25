from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.schema import SystemMessage, HumanMessage

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini")


def answer_from_notes(question: str, k: int = 3) -> str:
    results = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc in results)

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
    print("Quill AI — ask questions about your notes. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer = answer_from_notes(question)
        print(f"\nBot: {answer}\n")
