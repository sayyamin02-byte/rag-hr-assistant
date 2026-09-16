"""
MODULE 5 — Evaluating the RAG system.

You can't improve what you don't measure. Three standard RAG metrics:

1) CONTEXT RELEVANCE  — did RETRIEVAL fetch the right passage? (retriever quality)
   For each test question we labelled the correct handbook section ("gold"). We check:
   is that section among the top-k chunks we retrieved? No LLM needed for this one.

2) FAITHFULNESS (groundedness) — is every claim in the ANSWER actually supported by the
   retrieved context, with nothing invented? (the core anti-hallucination metric)
   We use an LLM as a JUDGE: show it the context + answer and ask YES/NO.

3) ANSWER RELEVANCE — does the answer actually ADDRESS the question? Also judged YES/NO.

We also include a TRAP question whose answer is NOT in the handbook, to check the bot
REFUSES instead of inventing — a correct refusal is faithful.

Run:  python eval.py   (needs the same Groq key in .env)
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
from openai import OpenAI

from rag.ingest import build_chunks
from rag.store import VectorStore
from rag.generate import answer

load_dotenv()

# A tiny labelled test set. "gold" = the section that should answer it (None = not in handbook).
TESTS = [
    {"q": "How many casual leaves per year and can I carry them forward?", "gold": "Casual Leave"},
    {"q": "Do I need a medical certificate for sick leave?",               "gold": "Sick Leave"},
    {"q": "How long is maternity leave and who is eligible?",              "gold": "Maternity & Paternity Leave"},
    {"q": "What is the internet reimbursement and the deadline to claim?", "gold": "Reimbursements"},
    {"q": "What is the health insurance sum insured per family?",          "gold": "Health Insurance"},
    {"q": "What is the notice period for resignation?",                    "gold": None},  # trap: not covered
]
TOP_K = 3


def judge(client, model, system, user) -> bool:
    """Ask the LLM a strict YES/NO question; return True if it says YES."""
    r = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return "YES" in r.choices[0].message.content.strip().upper()[:5]


FAITHFUL_SYS = ("You are a strict grader. Given CONTEXT and an ANSWER, reply with exactly "
                "YES if every factual claim in the ANSWER is supported by the CONTEXT "
                "(a refusal such as 'I couldn't find that' also counts as YES). Otherwise reply NO.")
RELEVANT_SYS = ("You are a strict grader. Reply with exactly YES if the ANSWER addresses the "
                "QUESTION, otherwise NO.")


def main():
    key = os.getenv("LLM_API_KEY")
    if not key:
        print("Add your Groq key to .env first — the judge needs an LLM.")
        return
    client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

    store = VectorStore().index(build_chunks("data/hr_policy.md"))

    ctx_hits = ctx_total = 0
    faith_hits = rel_hits = 0
    print(f"Evaluating {len(TESTS)} questions (top_k={TOP_K}) with judge = {model}\n")

    for t in TESTS:
        q = t["q"]
        hits = store.search(q, top_k=TOP_K)
        retrieved = [c.section for c, _ in hits]

        # 1) Context relevance (only where we have a gold section)
        if t["gold"] is not None:
            ctx_total += 1
            hit = t["gold"] in retrieved
            ctx_hits += 1 if hit else 0
            ctx_mark = "OK " if hit else "MISS"
        else:
            ctx_mark = "n/a"  # trap question — nothing to retrieve

        # 2) & 3) generate, then score faithfulness + answer relevance
        ans = answer(q, hits)
        low = ans.lower()
        is_refusal = ("couldn't find" in low or "could not find" in low or "not in the handbook" in low)

        if t["gold"] is None:
            # For a question we KNOW isn't in the handbook, the correct behaviour is a
            # refusal. We assert that directly instead of asking the LLM judge — the
            # judge tends to conflate "not supported by context" with "bad answer", so
            # it would wrongly penalise a perfect refusal. (Eval design matters!)
            faithful = relevant = is_refusal
        else:
            context = "\n\n".join(c.text for c, _ in hits)
            faithful = judge(client, model, FAITHFUL_SYS, f"CONTEXT:\n{context}\n\nANSWER:\n{ans}")
            relevant = judge(client, model, RELEVANT_SYS, f"QUESTION: {q}\n\nANSWER:\n{ans}")
        faith_hits += faithful
        rel_hits += relevant

        print(f"Q: {q}")
        print(f"   context:{ctx_mark}  faithful:{'YES' if faithful else 'NO '}  relevant:{'YES' if relevant else 'NO '}")
        print(f"   → {ans.splitlines()[0][:90]}")
        print()

    n = len(TESTS)
    print("──────── SCORECARD ────────")
    print(f"Context relevance : {ctx_hits}/{ctx_total}  ({ctx_hits/ctx_total*100:.0f}%)   (retriever found the right section)")
    print(f"Faithfulness      : {faith_hits}/{n}  ({faith_hits/n*100:.0f}%)   (answers grounded, incl. correct refusals)")
    print(f"Answer relevance  : {rel_hits}/{n}  ({rel_hits/n*100:.0f}%)   (answers address the question)")


if __name__ == "__main__":
    main()
