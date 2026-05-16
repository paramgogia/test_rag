"""
Streamlit chat interface for the Infosys RAG chatbot.

Run with:
    streamlit run app.py
"""
from __future__ import annotations
import streamlit as st

from config import (
    GEMINI_API_KEY, LLM_PROVIDER, SOURCE_FILES,
    OLLAMA_CHAT_MODEL, CHAT_MODEL,
)
from vector_store import VectorStore
from rag_engine import RagEngine, RagAnswer
from exporters import export_pdf, export_excel


ACTIVE_MODEL = OLLAMA_CHAT_MODEL if LLM_PROVIDER == "ollama" else CHAT_MODEL


# ---------- page setup ----------

st.set_page_config(
    page_title="Infosys Financial Analyst",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.small-muted { color: #6b7280; font-size: 0.85rem; }
.source-box {
    background: #f3f4f6; border-left: 3px solid #0a4d8c;
    padding: 8px 12px; margin: 4px 0; border-radius: 4px;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ---------- guards ----------

if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
    st.error(
        "**GEMINI_API_KEY** is not set. "
        "Copy `.env.example` to `.env` and add your key from "
        "https://aistudio.google.com/apikey, "
        "or set `LLM_PROVIDER = \"ollama\"` in `config.py` to run locally."
    )
    st.stop()


# ---------- engine bootstrap (cached) ----------

@st.cache_resource(show_spinner="Loading vector store...")
def get_engine() -> RagEngine:
    store = VectorStore()
    if not store.load():
        st.error("Vector store not found. Run `python ingest.py` first.")
        st.stop()
    return RagEngine(store)


engine = get_engine()


# ---------- session state ----------

if "messages" not in st.session_state:
    st.session_state.messages = []      # list of {role, content, sources?, export_path?, format?}


# ---------- sidebar ----------

with st.sidebar:
    st.title("📊 Infosys Analyst")
    st.caption("RAG over 7 Infosys financial documents")

    st.subheader("Document index")
    for name, fname in SOURCE_FILES.items():
        st.markdown(f"• **{name}**  \n<span class='small-muted'>{fname}</span>",
                    unsafe_allow_html=True)

    st.divider()
    st.subheader("Suggested questions")
    suggestions = [
        "What was Infosys's Q1 FY26 revenue and operating margin?",
        "Compare revenue and operating margin across all four FY26 quarters.",
        "Summarise the FY26 full-year performance for a board report.",
        "Build a table of total assets, equity and cash for every quarter.",
        "How did the BSE 500209 stock price move during FY26?",
        "List the large deal TCV wins reported in each quarter of FY26.",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{hash(s)}"):
            st.session_state.pending_question = s
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        engine.reset()
        st.rerun()


# ---------- main chat ----------

st.title("Infosys Financial Analyst")
st.caption("Ask anything about Infosys's FY25 Annual Report, FY26 quarterly results, "
           "multi-year financials, or FY26 stock price. Follow-ups are supported.")

# Render prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sources"):
                with st.expander(f"📎 Sources ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"<div class='source-box'>"
                            f"<b>{s['source']}</b> — {s['location']} "
                            f"<span class='small-muted'>"
                            f"(relevance: {s.get('score', '?')})</span><br>"
                            f"<span class='small-muted'>{s['snippet']}</span>"
                            f"</div>", unsafe_allow_html=True
                        )
            if msg.get("export_path"):
                fmt = msg.get("format", "")
                label = "📄 Download PDF" if fmt == "pdf" else "📊 Download Excel"
                with open(msg["export_path"], "rb") as f:
                    st.download_button(
                        label=label,
                        data=f.read(),
                        file_name=msg["export_path"].split("/")[-1],
                        mime="application/octet-stream",
                        key=f"dl_{msg['export_path']}",
                    )

# Pick up either a fresh prompt or a sidebar-suggested one
prompt = st.chat_input("Ask a question about Infosys...")
if not prompt and st.session_state.get("pending_question"):
    prompt = st.session_state.pop("pending_question")

if prompt:
    # show user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # generate answer with step-by-step visibility
    with st.chat_message("assistant"):
        status_box = st.status("Thinking…", expanded=True, state="running")
        answer_placeholder = st.empty()

        answer_text = ""
        sources_data: list[dict] = []
        final: RagAnswer | None = None

        try:
            for kind, payload in engine.ask_stream(prompt):
                if kind == "step":
                    name = payload["name"]
                    status = payload["status"]

                    if name == "rewrite" and status == "start":
                        if payload.get("has_history"):
                            status_box.write("🔄 Rewriting follow-up into a standalone question…")
                        else:
                            status_box.write("🧭 First turn — using the question as-is")

                    elif name == "rewrite" and status == "done":
                        if payload.get("rewritten"):
                            status_box.markdown(
                                f"📝 Standalone form: _{payload['standalone']}_"
                            )

                    elif name == "retrieve" and status == "start":
                        status_box.write(
                            f"🔍 Searching FAISS index for top-{payload['top_k']} chunks…"
                        )

                    elif name == "retrieve" and status == "done":
                        sources_data = payload["sources"]
                        unique = len({s["source"] for s in sources_data})
                        status_box.markdown(
                            f"📎 Retrieved **{payload['count']}** chunks from "
                            f"**{unique}** source(s):"
                        )
                        for s in sources_data[:6]:
                            status_box.markdown(
                                f"&nbsp;&nbsp;• **{s['source']}** — {s['location']} "
                                f"`score {s['score']}`"
                            )

                    elif name == "generate" and status == "start":
                        status_box.write(
                            f"🧠 Generating answer with `{payload['model']}` "
                            f"({payload['provider']})…"
                        )

                    elif name == "generate" and status == "done":
                        status_box.write(
                            f"✅ Answer drafted ({payload['length']} chars)"
                        )

                    elif name == "format" and status == "start":
                        status_box.write("🎨 Deciding output format…")

                    elif name == "format" and status == "done":
                        status_box.markdown(f"📄 Format chosen: **{payload['format']}**")

                elif kind == "token":
                    answer_text += payload["text"]
                    answer_placeholder.markdown(answer_text + " ▌")

                elif kind == "final":
                    final = payload

            # render the final answer without the cursor
            answer_placeholder.markdown(answer_text)
            status_box.update(label="Done", state="complete", expanded=False)

        except Exception as e:
            status_box.update(label=f"Error: {e}", state="error")
            err = f"Sorry, something went wrong: `{e}`"
            answer_placeholder.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.stop()

        assert final is not None  # the generator must yield "final"

        # sources expander (full list, persistent across reruns)
        with st.expander(f"📎 Sources ({len(final.sources)})"):
            for s in final.sources:
                st.markdown(
                    f"<div class='source-box'>"
                    f"<b>{s['source']}</b> — {s['location']} "
                    f"<span class='small-muted'>"
                    f"(relevance: {s.get('score', '?')})</span><br>"
                    f"<span class='small-muted'>{s['snippet']}</span>"
                    f"</div>", unsafe_allow_html=True
                )

        # export if needed
        export_path = None
        if final.format == "pdf":
            with st.spinner("📥 Building PDF…"):
                export_path = str(export_pdf(prompt, final.answer, final.sources))
            with open(export_path, "rb") as f:
                st.download_button(
                    "📄 Download PDF report",
                    data=f.read(),
                    file_name=export_path.split("/")[-1],
                    mime="application/pdf",
                    key=f"dl_new_{export_path}",
                )
        elif final.format == "excel":
            with st.spinner("📥 Building Excel…"):
                export_path = str(export_excel(prompt, final.answer, final.sources))
            with open(export_path, "rb") as f:
                st.download_button(
                    "📊 Download Excel sheet",
                    data=f.read(),
                    file_name=export_path.split("/")[-1],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_new_{export_path}",
                )

        # remember for re-render
        st.session_state.messages.append({
            "role": "assistant",
            "content": final.answer,
            "sources": final.sources,
            "format": final.format,
            "export_path": export_path,
        })
