"""
web.py — a simple browser chat UI for the HR assistant (Streamlit).

Same RAG pipeline as app.py, just with a clickable chat box instead of the
terminal — so you (or a user) can try it without touching code.

Run:
    pip install streamlit
    streamlit run web.py
It opens automatically in your browser at http://localhost:8501
"""
import streamlit as st
from dotenv import load_dotenv

from rag.ingest import build_chunks
from rag.generate import answer
from app import make_store, TOP_K     # reuse the exact same wiring as the CLI

load_dotenv()

st.set_page_config(page_title="HR Assistant (RAG)", page_icon="📄")

# Build + index the handbook ONCE and keep it in memory across clicks.
@st.cache_resource(show_spinner="Loading & indexing the HR handbook…")
def get_store():
    chunks = build_chunks("data/hr_policy.md")
    return make_store().index(chunks), len(chunks)

store, n_chunks = get_store()

st.title("📄 HR Assistant")
st.caption(f"Ask about leaves, insurance, reimbursements. Grounded in the handbook "
           f"({n_chunks} chunks indexed) — it answers only from the document, and says "
           f"so when it doesn't know.")

with st.sidebar:
    st.subheader("Try asking")
    for q in ["How many casual leaves per year and can I carry them forward?",
              "What's the internet reimbursement and the deadline to claim it?",
              "How long is maternity leave and who is eligible?",
              "Can I encash sick leave?"]:
        st.markdown(f"- {q}")
    st.divider()
    st.caption("Retrieval runs locally & free. The written answer needs an LLM key "
               "in `.env` (free Groq key from console.groq.com).")

# Keep the conversation on screen.
if "msgs" not in st.session_state:
    st.session_state.msgs = []
for role, text in st.session_state.msgs:
    st.chat_message(role).markdown(text)

if q := st.chat_input("Ask an HR question…"):
    st.session_state.msgs.append(("user", q))
    st.chat_message("user").markdown(q)

    with st.chat_message("assistant"):
        hits = store.search(q, top_k=TOP_K)                 # 1) RETRIEVE
        with st.expander("🔎 what the bot retrieved (the RAG part)"):
            for chunk, score in hits:                       # 2) show transparency
                st.markdown(f"**{chunk.section}**  ·  score `{score:.3f}`")
                st.caption(chunk.text[:300] + ("…" if len(chunk.text) > 300 else ""))
        reply = answer(q, hits)                             # 3) GENERATE grounded answer
        st.markdown(reply)
    st.session_state.msgs.append(("assistant", reply))
