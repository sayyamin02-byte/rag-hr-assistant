"""
MODULE 4 — Multi-Agent System (a Router + specialist agents).

Instead of ONE agent doing everything, we split the work across small SPECIALISTS,
with a ROUTER that decides who should handle each request.
Why: specialization, smaller focused prompts (saves tokens), and error isolation.

The team (this is the "Supervisor / Router" pattern):
  ROUTER      -> reads the question and picks ONE lane: policy / action / escalate
  POLICY      -> answers rule questions from the handbook            (uses RAG)
  ACTION      -> performs a transaction — check a leave balance      (uses a tool)
  ESCALATION  -> when it can't be handled, writes a ticket for a human

We reuse the tools and RAG we already built in agent.py / rag/, so each specialist
stays small and does one job well.

Run:  python multi_agent.py   (needs the Groq key in .env)
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
from openai import OpenAI

import agent                       # reuse: agent.store, check_leave_balance, TOOLS
from rag.generate import answer as rag_answer

load_dotenv()


def make_client():
    key = os.getenv("LLM_API_KEY")
    if not key:
        return None, None
    client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    return client, os.getenv("LLM_MODEL", "openai/gpt-oss-20b")


# ---------------------------- ROUTER ----------------------------
ROUTER_SYS = (
    "You are the router for an HR assistant. Read the employee's message and reply with "
    "ONE word only:\n"
    "  policy   = a question about rules/benefits (leave rules, insurance, reimbursement, hours)\n"
    "  action   = a request that needs the HR system (check my leave balance)\n"
    "  escalate = anything else / not covered / needs a human (resignation, disputes, exceptions)\n"
    "Reply with just one word: policy, action, or escalate."
)


def route(client, model, q: str) -> str:
    r = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": ROUTER_SYS}, {"role": "user", "content": q}])
    label = r.choices[0].message.content.strip().lower()
    for lane in ("policy", "action", "escalate"):
        if lane in label:
            return lane
    return "escalate"


# ---------------------- POLICY specialist (RAG) ----------------------
def policy_agent(q: str) -> str:
    hits = agent.store.search(q, top_k=3)
    return rag_answer(q, hits)


# ------------------- ACTION specialist (tool call) -------------------
ACTION_SYS = ("You are the HR actions agent. Use the check_leave_balance tool to answer. "
              "If the employee id or leave type is missing, ask for it. Be concise.")
ACTION_TOOLS = [t for t in agent.TOOLS if t["function"]["name"] == "check_leave_balance"]


def action_agent(client, model, q: str) -> str:
    messages = [{"role": "system", "content": ACTION_SYS}, {"role": "user", "content": q}]
    for _ in range(3):
        r = client.chat.completions.create(model=model, temperature=0, messages=messages, tools=ACTION_TOOLS)
        m = r.choices[0].message
        if not m.tool_calls:
            return m.content
        messages.append(m)
        for call in m.tool_calls:
            args = json.loads(call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": str(agent.check_leave_balance(**args))})
    return "Could not complete the action."


# --------------------- ESCALATION specialist ---------------------
ESC_SYS = ("You are the HR escalation agent. This request can't be handled automatically. "
           "Write a SHORT ticket for a human HR rep: a one-line summary and what the employee "
           "needs. Begin with 'TICKET:'.")


def escalate_agent(client, model, q: str) -> str:
    r = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": ESC_SYS}, {"role": "user", "content": q}])
    return r.choices[0].message.content.strip()


# --------------------- the supervisor wiring ---------------------
def handle(client, model, q: str) -> str:
    lane = route(client, model, q)          # 1) router decides the lane
    print(f"  ↳ router → {lane}")  # show the decision (transparency)
    if lane == "policy":
        return policy_agent(q)               # 2a) RAG specialist
    if lane == "action":
        return action_agent(client, model, q)  # 2b) tool specialist
    return escalate_agent(client, model, q)  # 2c) human hand-off


def main():
    client, model = make_client()
    if not client:
        print("Add your Groq key to .env first — the agents need an LLM.")
        return
    print("HR multi-agent ready (employees: E101 Aisha, E102 Rahul). Try:")
    print("  policy   → How long is maternity leave?")
    print("  action   → What's the casual leave balance for E101?")
    print("  escalate → I want to resign, what's the process?   (not in the handbook)")
    print("Type 'exit' to quit.\n")
    while True:
        try:
            q = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        print("bot › " + handle(client, model, q) + "\n")


if __name__ == "__main__":
    main()
