📄 PDF Chat Intelligence: LangChain & LangGraph RAG Agent

A sophisticated Streamlit application that allows users to upload multiple PDF documents and interact with them using an AI agent. The system uses a **Retrieval-Augmented Generation (RAG)** architecture, leveraging LangChain for document processing and LangGraph for agentic reasoning.

---

## 🚀 Key Features

* **Multi-PDF Support:** Upload and process multiple documents simultaneously.
* **Intelligent Retrieval:** Uses `OpenAIEmbeddings` (text-embedding-3-large) for high-accuracy semantic search.
* **Agentic Reasoning:** Powered by `ChatGroq`, the agent decides when to pull information from your documents using a custom retrieval tool.
* **Persistent Memory:** Includes an `InMemorySaver` to maintain conversation context across multiple queries.
* **Fast Vector Storage:** Utilizes `InMemoryVectorStore` for rapid prototyping and low-latency responses.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Orchestration** | LangChain & LangGraph |
| **LLM** | Groq (LPU Inference Engine) |
| **Embeddings** | OpenAI |
| **Vector DB** | InMemoryVectorStore |
| **Processing** | PyPDF & RecursiveCharacterTextSplitter |

---

## 📋 Prerequisites

Before running the application, ensure you have the following:

1.  **Python 3.9+** installed.
2.  **API Keys:**
    * `GROQ_API_KEY`: For the LLM reasoning.
    * `OPENAI_API_KEY`: For generating document embeddings.

---

## 🧠 System Architecture

The application follows a structured RAG pipeline to ensure accurate information retrieval[cite: 1]:

1. **Document Ingestion:** The `PyPDFDirectoryLoader` reads all files from the local storage[cite: 1].
2. **Chunking:** Documents are split into 1000-character segments with a 200-character overlap to preserve context[cite: 1].
3. **Vectorization:** OpenAI's `text-embedding-3-large` model converts text into high-dimensional vectors[cite: 1].
4. **Agent Logic:** 
    * The user sends a query[cite: 1].
    * The agent evaluates the query against its `system_prompt`[cite: 1].
    * If specific knowledge is needed, the agent calls the `retrieve_context` tool[cite: 1].
    * The tool performs a similarity search ($k=3$) and feeds the text back to the LLM[cite: 1].
5. **Response:** The LLM generates a grounded answer based on the retrieved snippets[cite: 1].

---

## ⚠️ Important Notes

* **Storage:** This version uses an `InMemoryVectorStore`[cite: 1]. If you restart the Streamlit application, the vector database will be cleared[cite: 1].
* **Model Selection:** The code currently uses `openai/gpt-oss-20b` via Groq[cite: 1]. Ensure this model is available or update the model string[cite: 1].
* **Concurrency:** The `thread_id` is currently hardcoded to `1`[cite: 1]. In a multi-user environment, this should be dynamically generated[cite: 1].
* **Directory Setup:** Ensure a `./doc_files/` directory exists in your root folder for document processing[cite: 1].

---

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install Dependencies**
    ```bash
    pip install streamlit langchain langchain-community langchain-openai langchain-groq langgraph pypdf python-dotenv
    ```

3.  **Configure Environment Variables**
    Create a `.env` file in the root directory and add your keys:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    GROQ_API_KEY=your_groq_api_key_here
    ```

4.  **Prepare Directory**
    Ensure a directory named `doc_files` exists to store uploaded PDFs:
    ```bash
    mkdir doc_files
    ```

---

## 🏃 How to Run

Start the Streamlit server with the following command:

```bash
streamlit run app.py
