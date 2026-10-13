import hashlib
import os
import tempfile
import time

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(page_title="DocChat - Ask your PDFs", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Config. Backend is chosen with LLM_BACKEND in .env:
#   groq (default) -> free Groq cloud LLM, key read from GROQ_API_KEY in .env
#   ollama         -> fully local, no key
# Embeddings are always local, so there is no OpenAI cost.
# ---------------------------------------------------------------------------
BACKEND = os.getenv("LLM_BACKEND", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODELS = {
    "Llama 3.1 8B (fastest)": "llama-3.1-8b-instant",
    "Llama 3.3 70B (smartest)": "llama-3.3-70b-versatile",
    "GPT-OSS 20B": "openai/gpt-oss-20b",
}
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
HISTORY_TURNS = 6

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions about the user's uploaded documents. "
    "Use ONLY the numbered context excerpts provided. Cite the excerpts you used like [1], [2]. "
    "If the answer is not in the context, say you couldn't find it in the documents. "
    "Be concise and use bullet points or short paragraphs."
)

QUICK_ACTIONS = {
    "📝 Summarize": "Give a concise summary of the uploaded document(s) with the key points as bullets.",
    "🔑 Key facts": "List the most important facts, numbers, dates and deadlines in the document(s).",
    "❓ Suggest questions": "Suggest 5 useful questions I could ask about these documents.",
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("messages", [])      # {"role", "content", "sources"}
ss.setdefault("files", {})         # sha1 -> {"name", "pages", "chunks"}
ss.setdefault("store", None)
ss.setdefault("uploader_key", 0)


# ---------------------------------------------------------------------------
# Cached heavy resources (loaded once per server, not once per upload/rerun)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embeddings(backend: str):
    if backend == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
    )


@st.cache_resource
def load_llm(backend: str, model: str, temperature: float, api_key: str):
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, temperature=temperature, api_key=api_key)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def index_files(new_files):
    """Load, split and embed only the files that are new. Adds to the existing store."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    with st.status("Processing documents...", expanded=True) as status:
        t0 = time.time()
        st.write("Loading embedding model...")
        embeddings = load_embeddings(BACKEND)
        if ss.store is None:
            ss.store = InMemoryVectorStore(embedding=embeddings)

        for digest, f in new_files:
            st.write(f"Reading **{f.name}**...")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(f.getvalue())
                path = tmp.name
            try:
                pages = PyPDFLoader(path).load()
            finally:
                os.remove(path)

            for p in pages:
                p.metadata["source"] = f.name
                p.metadata["page"] = int(p.metadata.get("page", 0)) + 1
            chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]

            if not chunks:
                st.warning(f"{f.name}: no extractable text (scanned PDF?). Skipped.")
                continue

            st.write(f"Embedding {len(chunks)} chunks from **{f.name}**...")
            ss.store.add_documents(chunks)
            ss.files[digest] = {"name": f.name, "pages": len(pages), "chunks": len(chunks)}

        status.update(label=f"Ready in {time.time() - t0:.1f}s", state="complete", expanded=False)


def reset_documents():
    ss.store = None
    ss.files = {}
    ss.messages = []
    ss.uploader_key += 1


# ---------------------------------------------------------------------------
# Answering (retrieve -> single streamed LLM call; no multi-step agent loop)
# ---------------------------------------------------------------------------
def retrieve(question: str, k: int):
    hits = ss.store.similarity_search_with_score(question, k=k)
    return [
        {
            "source": d.metadata.get("source", "?"),
            "page": d.metadata.get("page", "?"),
            "score": float(score),
            "text": d.page_content,
        }
        for d, score in hits
    ]


def stream_answer(llm, question: str, sources: list):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    context = "\n\n".join(
        f"[{i}] ({s['source']}, p.{s['page']})\n{s['text']}" for i, s in enumerate(sources, 1)
    )
    msgs = [SystemMessage(SYSTEM_PROMPT)]
    for m in ss.messages[-HISTORY_TURNS * 2:]:
        msgs.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
    msgs.append(HumanMessage(f"Context:\n{context}\n\nQuestion: {question}"))

    for chunk in llm.stream(msgs):
        if chunk.content:
            yield chunk.content


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}] {s['source']}** - page {s['page']} · relevance {s['score']:.2f}")
            st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


def chat_as_markdown():
    lines = ["# DocChat conversation", ""]
    for m in ss.messages:
        lines.append(f"**{'You' if m['role'] == 'user' else 'Assistant'}:** {m['content']}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 DocChat")
    st.caption("Chat with your PDFs - free and private embeddings.")

    if BACKEND == "groq":
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not found in your .env file. Add it and restart the app.")
            st.stop()
        label = st.selectbox("Model", list(GROQ_MODELS))
        model = GROQ_MODELS[label]
        st.success("Groq API key loaded from .env", icon="🔑")
    else:
        model = OLLAMA_MODEL
        st.info(f"Local Ollama model: {model}", icon="🖥️")

    with st.expander("⚙️ Settings"):
        temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.1)
        k = st.slider("Excerpts retrieved per question", 2, 10, 4)

    st.divider()
    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{ss.uploader_key}",
    )
    if uploaded:
        pending = []
        for f in uploaded:
            digest = hashlib.sha1(f.getvalue()).hexdigest()
            if digest not in ss.files:
                pending.append((digest, f))
        if pending:
            index_files(pending)

    if ss.files:
        st.markdown("**Indexed documents**")
        for info in ss.files.values():
            st.markdown(f"- {info['name']}  \n  <small>{info['pages']} pages · {info['chunks']} chunks</small>",
                        unsafe_allow_html=True)

        st.divider()
        st.markdown("**Quick actions**")
        for label_, prompt_ in QUICK_ACTIONS.items():
            if st.button(label_, use_container_width=True):
                ss.pending_prompt = prompt_

        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("🧹 Clear chat", use_container_width=True):
            ss.messages = []
            st.rerun()
        if c2.button("🗑️ Remove docs", use_container_width=True):
            reset_documents()
            st.rerun()
        if ss.messages:
            st.download_button(
                "⬇️ Download chat",
                chat_as_markdown(),
                file_name="docchat_conversation.md",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
if not ss.files:
    st.header("Ask questions about your documents")
    st.markdown(
        "1. **Upload** one or more PDFs in the sidebar\n"
        "2. **Ask** anything, or use a quick action like *Summarize*\n"
        "3. **Check sources** under every answer to see the exact page it came from"
    )
    st.info("Tip: the first upload is slower because the embedding model loads once; after that it's fast.")
    st.stop()

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources"))

question = st.chat_input("Ask anything about your documents...") or ss.pop("pending_prompt", None)

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            llm = load_llm(BACKEND, model, temperature, GROQ_API_KEY)
            with st.spinner("Searching documents..."):
                sources = retrieve(question, k)
            answer = st.write_stream(stream_answer(llm, question, sources))
            render_sources(sources)
        except Exception as e:
            answer, sources = f"Sorry, something went wrong: `{e}`", []
            st.error(answer)

    ss.messages.append({"role": "user", "content": question})
    ss.messages.append({"role": "assistant", "content": answer, "sources": sources})
