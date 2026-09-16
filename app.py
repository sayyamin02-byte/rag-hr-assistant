"""
The full RAG pipeline wired together — a small HR assistant you can chat with.

    ingest -> embed/index -> (loop) retrieve -> ground -> generate

Run:
    python app.py
Ask things like:
    "How many casual leaves per year and can I carry them forward?"
    "What is the internet reimbursement and by when must I claim it?"
    "How long is maternity leave and who is eligible?"
Type 'exit' to quit.
"""
from dotenv import load_dotenv

from rag.ingest import build_chunks
from rag.store import VectorStore
from rag.generate import answer

load_dotenv()  # read .env if present (for the optional LLM step)

TOP_K = 3


def main():
    print("Loading & indexing the HR handbook…")
    chunks = build_chunks("data/hr_policy.md")
    store = VectorStore().index(chunks)
    print(f"Indexed {len(chunks)} chunks. Ask a question (or 'exit').\n")

    while True:
        try:
            q = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break

        # 1) RETRIEVE the most relevant chunks
        hits = store.search(q, top_k=TOP_K)

        # 2) Show WHAT was retrieved (transparency — this is how you debug RAG)
        print("  ┌ retrieved:")
        for chunk, score in hits:
            print(f"  │  {score:.3f}  {chunk.section}")
        print("  └───────────")

        # 3) GENERATE a grounded answer from those chunks
        print("bot › " + answer(q, hits) + "\n")


if __name__ == "__main__":
    main()
