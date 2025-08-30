import os, re, json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chrome import Chroma
from langchain_core.documents import Document

load_dotenv()
EMBED_MODEL = os.getenv("SUNDAY_EMBEDDING_MODEL", "text-embedding-3-small")
PERSIST_DIR = "persist"
CHUNKS_DIR = "data/chunks"

FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)

def parse_doc(path: str) -> Document:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    meta = {}
    body = raw
    m = FM_RE.match(raw)
    if m:
        # naive YAML parser (keys: simple scalars or JSON-like lists)
        header, body = m.group(1), m.group(2)
        for line in header.splitlines():
            if not line.strip(): continue
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip().strip('"')
            if key == "tags":
                # allow JSON-like lists in tags
                try:
                    meta[key] = json.loads(val) if val else []
                except Exception:
                    meta[key] = [t.strip() for t in val.strip("[]").split(",") if t.strip()]
            else:
                meta[key] = val.strip('"')
    meta["source"] = os.path.relpath(path, start=CHUNKS_DIR)
    return Document(page_content=body.strip(), metadata=meta)

def iter_docs(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.lower().endswith(".txt"):
                yield parse_doc(os.path.join(dirpath, fn))

def main():
    docs = list(iter_docs(CHUNKS_DIR))
    if not docs:
        raise SystemExit(f"No .txt files found under {CHUNKS_DIR}. Add chunks first.")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma.from_documents(docs, embedding=embeddings, persist_directory=PERSIST_DIR)
    vs.persist()
    print(f"✅ Indexed {len(docs)} chunks → {PERSIST_DIR}/")

if __name__ == "__main__":
    main()
