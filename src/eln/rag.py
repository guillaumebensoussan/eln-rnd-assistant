"""Local RAG — ChromaDB + local embeddings.

Indexes experiments + documents (papers/patents) in a persistent local vector store.
Embeddings are computed on the machine: raw data never leaves the workstation
(IP-safe pillar).

Embeddings: Ollama `nomic-embed-text` if available, otherwise Chroma's built-in local
embedder (all-MiniLM, onnx) — no network call in either case.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = ROOT / "artifacts" / "chroma"
COLLECTION = "eln"


def _embedding_fn():
    """Try local Ollama, otherwise fall back to Chroma's local embedder."""
    try:
        import ollama  # noqa: F401

        client = chromadb.utils.embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings", model_name="nomic-embed-text"
        )
        # ping
        client(["ping"])
        return client
    except Exception:
        return embedding_functions.DefaultEmbeddingFunction()


class RAG:
    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = self.client.get_or_create_collection(
            name=COLLECTION, embedding_function=_embedding_fn()
        )

    def index(self) -> int:
        docs, ids, metas = [], [], []
        # experiments
        df = pd.read_csv(DATA_DIR / "clean_data.csv")
        for _, r in df.iterrows():
            text = "; ".join(f"{k}={v}" for k, v in r.items() if pd.notna(v))
            docs.append(text)
            ids.append(r["experiment_id"])
            metas.append({"kind": "experiment", "type": r["experiment_type"]})
        # documents
        for p in sorted(DOCS_DIR.glob("*.txt")):
            docs.append(p.read_text(encoding="utf-8"))
            ids.append(p.stem)
            metas.append({"kind": "patent" if "patent" in p.stem else "paper", "type": "document"})
        # idempotent upsert
        self.col.upsert(documents=docs, ids=ids, metadatas=metas)
        return len(docs)

    def search(self, query: str, k: int = 5) -> list[dict]:
        res = self.col.query(query_texts=[query], n_results=k)
        out = []
        for doc, _id, meta, dist in zip(
            res["documents"][0], res["ids"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append({"id": _id, "kind": meta.get("kind"), "distance": round(dist, 4), "text": doc})
        return out

    def context(self, query: str, k: int = 5) -> str:
        hits = self.search(query, k)
        return "\n".join(f"[{h['id']}] {h['text']}" for h in hits)


if __name__ == "__main__":
    rag = RAG()
    n = rag.index()
    print(f"indexed {n} documents")
    for q in ["which solvent gives the highest yield?", "what is the QC purity threshold?"]:
        print(f"\nQ: {q}")
        for h in rag.search(q, k=3):
            print(f"  - {h['id']} ({h['kind']}, dist={h['distance']})")
