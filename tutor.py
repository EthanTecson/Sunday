import os, sys, textwrap
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
EMBED_MODEL = os.getenv("SUNDAY_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("SUNDAY_CHAT_MODEL", "gpt-4.1-mini")
PERSIST_DIR = "persist"

SYSTEM_PROMPT = """You are Sunday, a Socratic CS tutor.
Teaching style: ask targeted questions first, then explain only as needed.
Never dump solutions. Prefer step-by-step guidance, checks for understanding,
and brief hints. When you cite sources, include lecture, section, and timestamps
from metadata if available. If the user says 'just tell me', still keep it concise.
Return Markdown."""

def format_context(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        label = " | ".join([
            md.get("lecture", "Lecture ?"),
            md.get("section", "Section ?"),
            f"{md.get('start','?')}→{md.get('end','?')}"
        ])
        src = md.get("source", "?")
        lines.append(f"[{i}] {label}  ·  `{src}`\n{d.page_content}")
    return "\n\n---\n\n".join(lines)

def citation_line(docs):
    bits = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        part = f"[{i}] {md.get('lecture','?')} · {md.get('section','?')} · {md.get('start','?')}→{md.get('end','?')} (`{md.get('source','?')}`)"
        bits.append(part)
    return "Sources: " + " | ".join(bits)

def main():
    # Load vector store & retriever
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)

    print("Sunday is ready. Type your question (q to quit).")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q: 
            continue
        if q.lower() in {"q","quit","exit"}:
            print("Bye!")
            break

        # Retrieve
        docs = retriever.get_relevant_documents(q)
        context = format_context(docs)

        # Prompt
        messages = [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": f"Question:\n{q}\n\nUse only the context below to ground your answer. If missing, say what else I should add.\n\nContext:\n{context}"}
        ]

        # Call model
        resp = llm.invoke(messages)
        answer = resp.content

        # Print with citations
        print("\n---")
        print(answer)
        print("\n" + citation_line(docs))

if __name__ == "__main__":
    main()

