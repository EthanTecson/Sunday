import os, sys, re, json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def parse_doc(path: str, root: str) -> Document:
    """Parse a document with optional YAML front-matter."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    
    meta = {}
    body = raw
    
    # Parse YAML front-matter if present
    FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
    m = FM_RE.match(raw)
    if m:
        header, body = m.group(1), m.group(2)
        for line in header.splitlines():
            if not line.strip(): 
                continue
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if key == "tags":
                # Convert tags list to comma-separated string for Chroma compatibility
                try:
                    tags_list = json.loads(val) if val else []
                except Exception:
                    tags_list = [t.strip() for t in val.strip("[]").split(",") if t.strip()]
                meta[key] = ", ".join(tags_list) if tags_list else ""
            else:
                meta[key] = val.strip('"')
    
    meta["source"] = os.path.relpath(path, start=root)
    return Document(page_content=body.strip(), metadata=meta)

def main():
    if len(sys.argv) < 2:
        print("Usage: python index.py <COURSE_CODE>")
        sys.exit(1)

    course = sys.argv[1]
    course_dir = os.path.join("courses", course)
    chunks_dir = os.path.join(course_dir, "data", "chunks")
    persist_dir = os.path.join(course_dir, "persist")

    # Load env (root first; course overrides optional)
    load_dotenv(".env", override=False)
    course_env = os.path.join(course_dir, ".env")
    if os.path.exists(course_env):
        load_dotenv(course_env, override=True)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("❌ Missing OPENAI_API_KEY in environment")

    embed_model = os.getenv("SUNDAY_EMBEDDING_MODEL", "text-embedding-3-small")
    
    if not os.path.isdir(chunks_dir):
        raise SystemExit(f"❌ Missing chunks folder: {chunks_dir}")

    # Parse documents
    docs = []
    for dirpath, _, filenames in os.walk(chunks_dir):
        for fn in sorted(filenames):
            if fn.lower().endswith(".txt"):
                docs.append(parse_doc(os.path.join(dirpath, fn), chunks_dir))
    
    if not docs:
        raise SystemExit(f"❌ No .txt chunks found in {chunks_dir}")

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model=embed_model)
    vs = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    
    print(f"✅ Indexed {len(docs)} chunks → {persist_dir}/")

if __name__ == "__main__":
    main()
