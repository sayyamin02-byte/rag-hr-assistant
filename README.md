# RAG HR Assistant

A minimal, from-scratch **Retrieval-Augmented Generation** chatbot that answers
questions from an internal HR handbook (leaves, insurance, reimbursements).
Built to *show the mechanics* — retrieval is hand-rolled with cosine similarity so
you can see exactly how it works, not hidden behind a framework.

```
ingest (chunk) ──▶ embed + index ──▶ retrieve (cosine top-k) ──▶ ground ──▶ generate
   ingest.py          store.py            store.py            generate.py  generate.py
```

## The pipeline (and the concept each step teaches)

| Step | File | Concept |
|------|------|---------|
| 1. Ingest & chunk | `rag/ingest.py` | document-aware + overlapping chunking, metadata |
| 2. Embed & retrieve | `rag/store.py` | dense embeddings, cosine similarity, top-k search |
| 3. Ground & generate | `rag/generate.py` | prompt synthesis, faithfulness, "say I don't know" |
| Wire-up | `app.py` | the full RAG loop with retrieval transparency |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# See retrieval alone (no API key needed):
python -m rag.store

# Full chatbot (add a free Groq key in .env for the written answer):
cp .env.example .env      # then paste your key
python app.py
```

Retrieval (Steps 1–2) runs fully **local and free** via `sentence-transformers`.
Only the final written answer (Step 3) calls an LLM — pluggable to **Groq (free)**,
**Ollama (local)**, or **OpenAI**.

## Try asking
- "How many casual leaves per year and can I carry them forward?"
- "What's the internet reimbursement and the deadline to claim it?"
- "How long is maternity leave and who is eligible?"
- "Can I encash sick leave?" ← tests grounded refusal (the handbook says no)

## Upgrade — a real vector database (`rag/store_chroma.py`)
`rag/store.py` hand-rolls the index with numpy so you can *see* the mechanic
(embed → normalise → dot product == cosine). `rag/store_chroma.py` swaps that for
**Chroma**, a real vector DB, while keeping the **exact same `.index()` / `.search()`
interface** — so it's a genuine drop-in:

| | `store.py` (numpy) | `store_chroma.py` (Chroma) |
|---|---|---|
| Where vectors live | in memory only | **persisted to disk** (`./.chroma/`) |
| On restart | re-embeds everything | **loads from disk, no re-embed** |
| Search | brute-force over all vectors | **HNSW** approximate-nearest-neighbour index |
| Similarity | cosine (by hand) | cosine (`hnsw:space`) — same scores |

```bash
python -m rag.store_chroma      # see it embed once, then load from disk on re-run
VECTOR_BACKEND=chroma python app.py   # run the whole chatbot on the real DB
#   (Windows PowerShell:  $env:VECTOR_BACKEND="chroma"; python app.py)
```
The concept never changed — same embedding model, same cosine, same top-k. That's
the point: retrieval quality is set by the *embeddings + chunking*, and the vector
DB is just the scalable machinery that stores and searches them.

## Module 3 — Agent version (`agent.py`)
Upgrades the chatbot into an **agent** with two tools it chooses between:
`search_policy` (RAG) and `check_leave_balance` (a pretend HR API). The model
decides which tool(s) to call, we run them, and it writes the final answer — the
**ReAct** (Reason + Act) loop.

```bash
python agent.py   # needs a Groq key in .env (the agent uses the LLM to decide)
```
Killer demo (uses BOTH tools): *"How much casual leave does E101 have, and can it be carried forward?"*

## Module 4 — Multi-agent system (`multi_agent.py`)
A **Router** reads each question and delegates to a specialist (Supervisor pattern):
- **Router** → picks a lane: `policy` / `action` / `escalate`
- **Policy agent** → answers rule questions from the handbook (RAG)
- **Action agent** → runs a transaction (leave-balance lookup) via a tool
- **Escalation agent** → writes a ticket for a human when it's out of scope

```bash
python multi_agent.py
#  "How long is maternity leave?"          → router → policy
#  "Casual leave balance for E101?"        → router → action
#  "I want to resign, what's the process?" → router → escalate (writes a TICKET)
```

## Module 5 — Evaluation (`eval.py`)
Scores the system on the three standard RAG metrics against a small labelled test set:
- **Context relevance** — did retrieval fetch the correct handbook section?
- **Faithfulness** — is every claim grounded in the retrieved context (incl. correct refusals)?
- **Answer relevance** — does the answer address the question?

Faithfulness & relevance use an **LLM-as-judge**; a **trap** question (not in the handbook)
checks the bot *refuses* instead of hallucinating.

```bash
python eval.py
# Context relevance : 5/5 (100%)   Faithfulness : 6/6 (100%)   Answer relevance : 6/6 (100%)
```

## What to build next (extensions for your portfolio)
- [x] Swap the numpy index for **Chroma** (`rag/store_chroma.py`) — real, persistent vector DB.
- [ ] Add **hybrid search** (BM25 keyword + vector) and a **re-ranker**.
- [ ] Add **guardrails**: PII masking + prompt-injection defense + output validation.
- [x] Add **evaluation** (`eval.py`): context relevance, faithfulness, answer relevance.
- [x] Turn it into an **agent** (`agent.py`) and a **multi-agent system** (`multi_agent.py`).
