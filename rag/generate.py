"""
STEP 3 — Grounded Generation (prompt synthesis + the LLM call).

This is the "G" in RAG. We take the chunks retrieved in Step 2 and STUFF them into
the prompt as CONTEXT, then instruct the model to answer ONLY from that context.
That instruction is what makes the answer "grounded" / faithful — and why we tell
it to say it doesn't know rather than invent (reducing hallucination).

The LLM is pluggable via any OpenAI-COMPATIBLE endpoint, so you can use:
  - Groq (free, fast)  -> https://api.groq.com/openai/v1   (get a free key)
  - Ollama (local)     -> http://localhost:11434/v1        (any api key string)
  - OpenAI             -> default base_url
Set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in a .env file (see .env.example).
Retrieval in Step 2 needs NO key — only this final answer step does.
"""
import os

from .ingest import Chunk

SYSTEM_PROMPT = (
    "You are an HR policy assistant for Acme Corp. Answer the employee's question "
    "using ONLY the numbered context passages provided. If the answer is not in the "
    "context, say: 'I couldn't find that in the handbook.' Be concise, quote the "
    "specific rule (numbers, days, amounts), and cite the section name you used."
)


def build_prompt(question: str, contexts: list[tuple[Chunk, float]]) -> str:
    """Assemble retrieved chunks into a single grounded prompt."""
    blocks = []
    for i, (chunk, score) in enumerate(contexts, 1):
        blocks.append(f"[{i}] (Section: {chunk.section})\n{chunk.text}")
    context = "\n\n".join(blocks)
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer (grounded in the context, cite the section):"
    )


def answer(question: str, contexts: list[tuple[Chunk, float]]) -> str:
    """Call the LLM with the grounded prompt. Requires the openai package + env vars."""
    try:
        from openai import OpenAI
    except ImportError:
        return "(LLM step skipped: run `pip install openai` and set the .env vars.)"

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return ("(LLM step skipped: no LLM_API_KEY set. Retrieval above already works — "
                "add a free Groq key to .env to get a written answer.)")

    client = OpenAI(api_key=api_key,
                    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        temperature=0,  # deterministic, factual — right for grounded Q&A
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, contexts)},
        ],
    )
    return resp.choices[0].message.content.strip()
