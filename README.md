---
title: Self Aware RAG
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# Self-Aware-Rag🦾 : Production-Grade Agentic Research Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688) ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED) ![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-black) ![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange)

self-aware-rag is a highly scalable, layout-aware Retrieval-Augmented Generation (RAG) system built to process and query dense academic research. It leverages an agentic architecture with built-in self-reflection to accurately retrieve information from complex documents — including tables, mathematical formulas, and appendices.

---

## 🚀 Key Features

- **Massive Scale Processing** — Successfully ingested a corpus of 249 dense AI research papers (including cutting-edge 2025 releases like DeepSeek-R1), resulting in a high-density index of **42,385 vectors**.
- **Layout-Aware Parsing** — Utilizes `Docling` to intelligently parse PDFs. Unlike standard parsers, it preserves the semantic structure of data tables, equations, and appendices for highly accurate retrieval.
- **Agentic Self-Reflection Loop** — Powered by `LangGraph` and the Groq API, the system uses a stateful agent that critiques its own answers, rewrites search queries, and retries — all autonomously.
- **Multi-Source Retrieval** — Queries Pinecone (local vector store), Wikipedia, and arXiv simultaneously to maximize answer coverage.
- **Cloud-Native Vector Storage** — Embeddings generated via HuggingFace (`google/embeddinggemma-300m`) and stored in a serverless Pinecone vector database for millisecond-latency semantic search.
- **Production-Ready Backend** — Decoupled, asynchronous FastAPI backend with CORS support, fully containerized with Docker.

---

## 🧠 How the Reflection Loop Works

```
User Query
    │
    ▼
┌─────────────┐
│  Retriever  │ ◄─────────────────────────────┐
└──────┬──────┘                               │
       │  (Pinecone + Wikipedia + arXiv)      │
       ▼                                      │
┌─────────────┐                               │
│  Responder  │                               │
└──────┬──────┘                               │
       │  (Cited answer via Groq LLM)         │
       ▼                                      │
┌─────────────┐                               │
│  Reflector  │                               │
└──────┬──────┘                               │
       │                                      │
   ┌───┴────┐                                 │
   │        │                                 │
  YES       NO (attempts < 2)                 │
   │        │                                 │
   ▼        ▼                                 │
┌──────┐ ┌─────────┐                          │
│ Done │ │Rewriter │──────────────────────────┘
└──────┘ └─────────┘  (refined search query)
```

1. **Retrieve** — Pulls documents from Pinecone, Wikipedia, and arXiv.
2. **Generate** — Produces a cited answer with mandatory inline citations (`[0]`, `[1]`, ...).
3. **Reflect** — The LLM evaluates whether the answer is complete, accurate, and properly cited.
4. **Branch:**
   - `YES` or `attempts >= 2` → answer is finalized and returned.
   - `NO` → query is rewritten based on the reflection feedback and the loop retries.

---

## 🛠️ Technology Stack

| Component          | Technology                                 |
| ------------------ | ------------------------------------------ |
| Framework          | FastAPI + Uvicorn                          |
| AI Orchestration   | LangGraph, LangChain                       |
| LLM Provider       | Groq API (`openai/gpt-oss-120b`)           |
| Embeddings         | HuggingFace (`google/embeddinggemma-300m`) |
| Vector Database    | Pinecone (`bionic-rag-cloud`)              |
| Document Parsing   | Docling                                    |
| External Retrieval | Wikipedia + arXiv                          |
| Deployment         | Docker                                     |

---

## 📁 Project Structure

```
.
├── src/
│   ├── agent.py          # RAG state, retrieval, generation, reflection, rewriting
│   ├── graph.py          # LangGraph state machine definition
│   ├── vector_store.py   # Pinecone vector store + HuggingFace embeddings
│   ├── cofig.py          # LLM (Groq) configuration
│   └── cli.py            # Interactive CLI entrypoint
├── main.py               # FastAPI server
├── requirements.txt
├── Dockerfile
├── .env                  # API keys (not committed)
└── README.md
```

---

## ⚙️ Quick Start (Local Development)

### 1. Set up the environment

```bash
cd bionic-rag
python -m venv venv
source venv/bin/activate        # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 3. Pinecone index

Make sure your Pinecone index named `bionic-rag-cloud` exists and is populated before running. The system expects embeddings generated by `google/embeddinggemma-300m`.

### 4. Run the API server

```bash
python main.py
```

Server starts at `http://127.0.0.1:8000`.

### 5. Run the interactive CLI

```bash
python -m src.cli
```

---

## 🐳 Docker

```bash
docker build -t bionic-rag .
docker run -p 8000:8000 --env-file .env bionic-rag
```

---

## 📡 API Reference

### `POST /chat`

**Request body:**

```json
{
  "chat": "Explain the key findings of DeepSeek-R1",
  "thread_id": "session_abc"
}
```

| Field       | Type   | Required | Description                                       |
| ----------- | ------ | -------- | ------------------------------------------------- |
| `chat`      | string | ✅       | The user's question                               |
| `thread_id` | string | ❌       | Session ID for memory continuity (default: `"1"`) |

**Response:**

```json
{
  "res": "DeepSeek-R1 introduces... [0] ... [1]",
  "sources": ["arxiv", "wikipedia", "bionic-rag-cloud"]
}
```

| Field     | Description                                    |
| --------- | ---------------------------------------------- |
| `res`     | Final answer with inline source citations      |
| `sources` | Source identifiers for all retrieved documents |

---

## 🔩 State Schema

```python
class RAGReflectionState(BaseModel):
    question: str          # Original user question
    search_query: str      # Current search query (may be rewritten)
    retrieved_docs: list   # Documents from all sources
    answer: str            # Generated answer
    reflection: str        # LLM self-critique output
    revised: bool          # Whether a retry is needed
    attempts: int          # Number of generation attempts made
```

---

## ⚙️ Configuration

**Change the LLM** — edit `src/cofig.py`:

```python
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
```

**Change the embedding model or Pinecone index** — edit `src/vector_store.py`:

```python
embeddings = HuggingFaceEmbeddings(model="google/embeddinggemma-300m")
PineconeVectorStore(index_name="bionic-rag-cloud", ...)
```

---

## 📄 License

MIT
