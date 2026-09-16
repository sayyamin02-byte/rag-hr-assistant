"""
STEP 2 (upgraded) — the SAME retrieval, now on a REAL vector database: Chroma.

In store.py we hand-rolled the vector index with numpy so you could see the
mechanic (embed -> normalise -> dot product == cosine). That's perfect for
learning, but it has two real-world problems:

  1) It lives only in memory. Every time the program starts, we re-embed every
     chunk from scratch. With thousands of documents that's slow and wasteful.
  2) It does a brute-force compare against EVERY vector. Fine for 7 chunks,
     hopeless at a million.

A production vector database fixes both: it PERSISTS the vectors to disk (embed
once, reuse forever) and uses an ANN index (approximate nearest-neighbour, here
HNSW) so search stays fast as the collection grows. Chroma is the smallest such
database to run — no server, just a folder on disk.

The important lesson: we did NOT change the concept. Same embedding model, same
cosine similarity, same "return the best k chunks" — so this class exposes the
*exact same* .index() / .search() interface as VectorStore. That's why every
other file (app.py, agent.py, multi_agent.py, eval.py) can use it unchanged:

    from rag.store_chroma import ChromaStore as VectorStore   # drop-in swap
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import chromadb
from sentence_transformers import SentenceTransformer

from .ingest import Chunk


class ChromaStore:
    """Drop-in replacement for VectorStore, backed by a persistent Chroma DB.

    We keep control of the embeddings ourselves (same all-MiniLM-L6-v2 model as
    before, runs locally, no API key) and hand the vectors to Chroma. Chroma then
    stores them on disk and does the nearest-neighbour search for us.
    """

    def __init__(self, path: str = ".chroma", collection: str = "hr_policy",
                 model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # PersistentClient writes to a folder on disk instead of only memory.
        self.client = chromadb.PersistentClient(path=path)
        self.name = collection
        # "hnsw:space": "cosine" tells Chroma to rank by cosine similarity,
        # matching exactly what we did by hand in store.py.
        self.col = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"})

    def _embed(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False).tolist()

    def count(self) -> int:
        return self.col.count()

    def index(self, chunks: list[Chunk], rebuild: bool = False):
        """Embed every chunk ONCE and store it in Chroma (on disk).

        On later runs the collection is already populated, so we skip the work —
        that's the persistence win. Pass rebuild=True to force a fresh rebuild
        (e.g. after you edit the handbook).
        """
        if rebuild and self.count() > 0:
            self.client.delete_collection(self.name)
            self.col = self.client.get_or_create_collection(
                name=self.name, metadata={"hnsw:space": "cosine"})

        if self.count() == 0:
            self.col.add(
                ids=[str(c.chunk_id) for c in chunks],
                embeddings=self._embed([c.text for c in chunks]),
                documents=[c.text for c in chunks],
                # metadata rides alongside each vector — used to cite the source
                # and to filter (e.g. only search within "Sick Leave").
                metadatas=[{"section": c.section, "chunk_id": c.chunk_id} for c in chunks],
            )
            print(f"  (embedded + stored {len(chunks)} chunks in Chroma)")
        else:
            print(f"  (loaded {self.count()} chunks from disk — no re-embedding needed)")
        return self

    def search(self, query: str, top_k: int = 3):
        """Same signature and return shape as VectorStore.search:
        a list of (Chunk, similarity_score) sorted best-first."""
        res = self.col.query(
            query_embeddings=self._embed([query]),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            chunk = Chunk(text=text, section=meta.get("section", ""),
                          chunk_id=int(meta.get("chunk_id", 0)), meta=meta)
            # Chroma returns a cosine DISTANCE (0 = identical). Convert back to a
            # similarity (1 = identical) so the scores read like store.py's.
            out.append((chunk, 1.0 - float(dist)))
        return out


if __name__ == "__main__":
    from .ingest import build_chunks

    print("Building Chroma store (first run embeds; later runs load from disk):")
    store = ChromaStore().index(build_chunks("data/hr_policy.md"))
    print(f"Collection '{store.name}' holds {store.count()} chunks, persisted at ./.chroma/\n")

    for q in ["How many sick leaves do I get?",
              "internet bill reimbursement",
              "how long is maternity leave"]:
        print(f"Q: {q}")
        for chunk, score in store.search(q, top_k=2):
            print(f"  {score:.3f}  ({chunk.section})  {chunk.text[:70]}...")
        print()
