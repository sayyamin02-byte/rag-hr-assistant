"""
STEP 2 — Embeddings & the Vector Store (+ retrieval).

An EMBEDDING model maps text -> a dense vector (here, 384 numbers) such that
texts with similar MEANING land close together in that 384-dimensional space.
We use `all-MiniLM-L6-v2` (small, fast, runs locally, no API key).

We then compare vectors with COSINE SIMILARITY — the cosine of the angle between
them. Cosine ignores magnitude (document length), which is why it's the standard
for text. Trick: if we L2-NORMALISE every vector to length 1, then
cosine similarity == dot product == a single matrix multiply. That's exactly
what a vector database does under the hood — we do it by hand here so you can see
the mechanic. (Swapping this numpy index for FAISS/Chroma later changes nothing
conceptually — just scale.)
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import Chunk


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None  # shape (n_chunks, dim), L2-normalised

    def _embed(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True gives unit-length vectors, so dot == cosine.
        return self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False)

    def index(self, chunks: list[Chunk]):
        """Embed every chunk once and keep the matrix in memory."""
        self.chunks = chunks
        self.vectors = self._embed([c.text for c in chunks])
        return self

    def search(self, query: str, top_k: int = 3):
        """Embed the query, cosine-compare against all chunks, return the best k."""
        q = self._embed([query])[0]            # (dim,)
        scores = self.vectors @ q              # (n_chunks,) cosine similarities
        order = np.argsort(scores)[::-1][:top_k]  # indices of the highest scores
        return [(self.chunks[i], float(scores[i])) for i in order]


if __name__ == "__main__":
    from .ingest import build_chunks
    store = VectorStore().index(build_chunks("data/hr_policy.md"))
    for q in ["How many sick leaves do I get?",
              "internet bill reimbursement",
              "how long is maternity leave"]:
        print(f"\nQ: {q}")
        for chunk, score in store.search(q, top_k=2):
            print(f"  {score:.3f}  ({chunk.section})  {chunk.text[:70]}...")
