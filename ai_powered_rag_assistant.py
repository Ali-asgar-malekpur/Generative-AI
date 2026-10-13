import os

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

# ---------------------------------------------------------------------------
# Free setup. Choose the backend with LLM_BACKEND in your .env file:
#   LLM_BACKEND=groq    (default) free Groq cloud LLM, needs GROQ_API_KEY
#   LLM_BACKEND=ollama  fully local, no key, needs Ollama installed
# Embeddings are local in both cases, so there is no OpenAI cost.
# ---------------------------------------------------------------------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq").lower()
DOC_DIR = "./doc_files/"
os.makedirs(DOC_DIR, exist_ok=True)

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


## API key: paste it in the sidebar (kept only in this browser session).
## Falls back to GROQ_API_KEY from .env if you set one.
groq_api_key = ""
if LLM_BACKEND == "groq":
    groq_api_key = st.sidebar.text_input(
        "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",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Free key from console.groq.com",
    ).strip()


def get_embeddings():
    if LLM_BACKEND == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm():
    if LLM_BACKEND == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1")  # must be a tool-calling model
    from langchain_groq import ChatGroq
    return ChatGroq(model="openai/gpt-oss-20b", api_key=groq_api_key)


def process_document(path):

    ## load the documents
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    ## split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents=docs)

    ## embeddings and Vector DB
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=get_embeddings()
    )
    st.session_state.vector_store = vector_db

    ## create an agent - tool, llm, prompt
    llm = get_llm()

    @tool
    def retrieve_context(query: str):
        """Retrieve documents relevant to a query from the knowledge base."""
        results = vector_db.similarity_search(query=query, k=3)
        return "\n\n".join(doc.page_content for doc in results)

    system_prompt = """You are a helpful assistant that answers questions using retrieved context.
        My knowledge base consists of the details from the uploaded document.
        ALWAYS use the `retrieve_context` tool for questions requiring external knowledge.
        If the context does not contain the answer, say so."""

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


### upload ui
if LLM_BACKEND == "groq" and not groq_api_key:
    st.info("Paste your Groq API key in the sidebar to begin.")
    st.stop()

if not st.session_state.document_uploaded:
    uploaded =st.file_uploader(label="Select PDF Files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            for file in uploaded:
                with open(os.path.join(DOC_DIR, file.name), "wb") as f:
                    f.write(file.getvalue())

            process_document(DOC_DIR)
        st.rerun()

## chat ui
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    query = st.chat_input("Ask anything related to uploaded documents....")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        with st.spinner("Thinking..."):
            response = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                {"configurable": {"thread_id": "1"}}
            )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role": "ai", "content": answer})
