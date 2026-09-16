"""
MODULE 3 — From a RAG chatbot to a simple AGENT.

Plain-English idea:
  - A normal chatbot can only TALK.
  - An AGENT is a chatbot with HANDS: you hand it "tools" (small functions), and it
    decides on its own which tool to use to help you.

Our agent has two tools:
  1) search_policy(query)            -> looks things up in the HR handbook  (this is RAG)
  2) check_leave_balance(emp, type)  -> checks a (pretend) HR database for a balance

The flow (this loop is called "ReAct" = Reason + Act):
  you ask  ->  AI decides which tool  ->  we run the tool  ->  we hand back the result
           ->  AI decides again (another tool? or done?)  ->  ... ->  AI writes the answer

The AI never runs code itself — it only *asks* to call a tool by name with arguments;
our Python code actually runs it and returns the result.
"""
import json
import os
import sys

try:  # make the Windows console print em-dashes/emoji without crashing
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv

from rag.ingest import build_chunks
from rag.store import VectorStore

load_dotenv()

# --- A pretend HR database. In a real system this would be an API/DB call. ---
EMPLOYEES = {
    "E101": {"name": "Aisha", "casual_leave": 7, "sick_leave": 9,  "earned_leave": 12},
    "E102": {"name": "Rahul", "casual_leave": 2, "sick_leave": 10, "earned_leave": 21},
}

# Build the RAG index once when the program starts.
store = VectorStore().index(build_chunks("data/hr_policy.md"))


# ---------------- The two tools (just normal Python functions) ----------------
def search_policy(query: str) -> str:
    """TOOL 1: semantic search over the handbook — returns the top matching passages."""
    hits = store.search(query, top_k=3)
    return "\n\n".join(f"({c.section}) {c.text}" for c, _ in hits)


def check_leave_balance(employee_id: str, leave_type: str) -> str:
    """TOOL 2: look up a specific employee's remaining balance."""
    emp = EMPLOYEES.get(employee_id)
    if not emp:
        return f"No employee found with id {employee_id}."
    key = leave_type.lower().replace(" ", "_")
    if key not in emp:
        return f"Unknown leave type '{leave_type}'. Try casual_leave, sick_leave or earned_leave."
    return f"{emp['name']} ({employee_id}) has {emp[key]} {key.replace('_',' ')} remaining."


# ---- Describe the tools in JSON so the AI knows what it's allowed to call ----
# (name + what it does + what arguments it needs. The AI reads this to decide.)
TOOLS = [
    {"type": "function", "function": {
        "name": "search_policy",
        "description": "Search the HR handbook for policy rules — leave rules, insurance, reimbursements, working hours.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look up"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "check_leave_balance",
        "description": "Get a specific employee's remaining leave balance from the HR system.",
        "parameters": {"type": "object", "properties": {
            "employee_id": {"type": "string", "description": "e.g. E101"},
            "leave_type": {"type": "string", "description": "casual_leave, sick_leave, or earned_leave"}},
            "required": ["employee_id", "leave_type"]}}},
]
FUNCS = {"search_policy": search_policy, "check_leave_balance": check_leave_balance}

SYSTEM = (
    "You are Acme Corp's HR assistant. Use the search_policy tool for policy/rule questions, "
    "and check_leave_balance for an employee's remaining balance. You may use BOTH if needed. "
    "Answer ONLY from tool results; if something isn't available, say so. Cite the policy "
    "section when you used it. Be concise."
)


def run_agent(client, model, user_msg: str) -> str:
    """The agent loop: let the AI call tools until it's ready to answer."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg}]

    for _ in range(5):  # safety cap so it can never loop forever
        resp = client.chat.completions.create(
            model=model, temperature=0, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message

        if not msg.tool_calls:        # no tool requested -> this is the final answer
            return msg.content

        messages.append(msg)          # record the AI's decision to call tool(s)
        for call in msg.tool_calls:   # run each requested tool and feed the result back
            args = json.loads(call.function.arguments)
            print(f"  ⚙ agent calls {call.function.name}({args})")
            result = FUNCS[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})

    return "Stopped after too many steps (safety cap)."


def main():
    from openai import OpenAI
    key = os.getenv("LLM_API_KEY")
    if not key:
        print("This agent needs an LLM to make decisions. Add a free Groq key to .env first.")
        return
    client = OpenAI(api_key=key,
                    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

    print("HR Agent ready (employees: E101 Aisha, E102 Rahul). Try, e.g.:")
    print("  How much casual leave does E101 have, and can it be carried forward?")
    print("  ↑ that one needs BOTH tools. Type 'exit' to quit.\n")
    while True:
        try:
            q = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        print("bot › " + run_agent(client, model, q) + "\n")


if __name__ == "__main__":
    main()
