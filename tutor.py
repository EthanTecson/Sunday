import os, sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

SYSTEM_PROMPT = """You are Sunday, a helpful CS tutor.
Your job is to explain lecture content clearly and answer questions about the material.
Keep explanations clear and helpful. Return Markdown."""

def format_context(docs):
    """Format retrieved documents for context."""
    lines = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        label = " | ".join([
            md.get("lecture", "Lecture ?"),
            md.get("section", "Section ?")
        ])
        src = md.get("source", "?")
        lines.append(f"[{i}] {label}  ·  `{src}`\n{d.page_content}")
    return "\n\n---\n\n".join(lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python tutor.py <COURSE_CODE>")
        sys.exit(1)

    course = sys.argv[1]
    course_dir = os.path.join("courses", course)
    persist_dir = os.path.join(course_dir, "persist")

    # Load env
    load_dotenv(".env", override=False)
    course_env = os.path.join(course_dir, ".env")
    if os.path.exists(course_env):
        load_dotenv(course_env, override=True)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("❌ Missing OPENAI_API_KEY in environment")

    # Check if index exists
    if not os.path.exists(persist_dir):
        print(f"❌ No index found for {course}")
        print(f"Run: python index.py {course}")
        sys.exit(1)

    # Initialize components
    embed_model = os.getenv("SUNDAY_EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("SUNDAY_CHAT_MODEL", "gpt-4o-mini")
    
    embeddings = OpenAIEmbeddings(model=embed_model)
    vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
    
    # Get document count
    try:
        doc_count = len(vs.get()["documents"])
    except:
        doc_count = "unknown"
    
    llm = ChatOpenAI(model=chat_model, temperature=0.2)
    
    print(f"🎓 Sunday ready for {course} ({doc_count} documents indexed)")
    print("Type your question (q to quit)")

    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not q: 
            continue
        if q.lower() in {"q", "quit", "exit"}:
            print("Bye!")
            break

        # Retrieve relevant documents
        try:
            docs = retriever.invoke(q)
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            continue

        context = format_context(docs)

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question:\n{q}\n\nUse the context below to answer:\n\nContext:\n{context}"}
        ]

        # Call model
        try:
            resp = llm.invoke(messages)
            answer = resp.content
        except Exception as e:
            print(f"❌ Model error: {e}")
            continue

        # Print response
        print("\n---")
        print(answer)

if __name__ == "__main__":
    main()
