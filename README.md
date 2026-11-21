# Multimodal Knowledge Base

[English](README.md) | [简体中文](README.zh-CN.md)

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white) [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An education-focused multimodal knowledge base for course materials. It combines document and image ingestion, hybrid retrieval, and a tool-using RAG agent in a FastAPI service with a React interface.

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Configuration and data](#configuration-and-data)
- [API overview](#api-overview)
- [Development](#development)
- [Security and privacy](#security-and-privacy)
- [Limitations](#limitations)
- [License](#license)

## Features

- Ingest PDF, Word, PowerPoint, text, Markdown, and image files.
- Extract text with Unstructured and PaddleOCR; use Qwen3-VL for image and diagram descriptions.
- Retrieve with BM25 and dense vectors, fuse and diversify candidates, then rerank them.
- Generate source-grounded answers with a DeepSeek-powered agent that can search the active knowledge base, read retrieved evidence, or ask the user for required information.
- Manage accounts, conversations, and workspaces through the web interface. Workspace access is checked by the backend.
- Stream answers over Server-Sent Events (SSE) and show source references in the UI.

## How it works

```text
Documents and images
  → parsing, OCR, and visual analysis
  → confidence checks and hierarchical chunking
  → BM25 + multimodal dense retrieval
  → reciprocal-rank fusion and MMR
  → reranking
  → agent-selected search, source reading, or clarification
  → source and answer checks
  → FastAPI / SSE → React
```

The agent currently has three tools:

| Tool | Scope |
| --- | --- |
| `search_knowledge(query)` | Searches the current authorized workspace and returns ranked evidence snippets. |
| `read_source(evidence_id)` | Reads source text from evidence retrieved during the current request; it cannot open arbitrary file paths. |
| `collect_user_info(question)` | Asks one question when the answer depends on information only the user can provide. |

## Requirements

- Python 3.11
- Node.js and npm
- A reachable Milvus instance for the default vector-store configuration ([deployment options](https://milvus.io/docs/install-overview.md))
- API credentials for the configured language and multimodal model providers

The OCR and document-parsing dependencies are large and may take time to install. Model API usage may incur provider charges.

## Quick start

### 1. Configure the backend

```bash
git clone https://github.com/zhe-w12/Multimodal-knowledge-base.git
cd Multimodal-knowledge-base
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/install_nltk_data.py
cp .env.example .env
```

On Windows PowerShell, use `py -3.11 -m venv .venv`, activate with `.\.venv\Scripts\Activate.ps1`, and create the local configuration with `Copy-Item .env.example .env`. Then run the same `python -m pip install -r requirements.txt` and `python scripts/install_nltk_data.py` commands.

Edit `.env` and set `DEEPSEEK_API_KEY`, `BAILIAN_API_KEY`, and `BAILIAN_WORKSPACE_ID`. Start Milvus and set `MILVUS_ENDPOINT` (and `MILVUS_TOKEN` if required by your deployment). The backend loads `.env` from the repository root. Redis and MySQL integrations are optional and disabled by default; account data uses a local SQLite database unless `AUTH_DATABASE_URL` is configured.

| Variable | Required | Purpose |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | Yes | Primary chat model. |
| `BAILIAN_API_KEY` | Yes for default model providers | Embedding, reranking, and vision services. |
| `BAILIAN_WORKSPACE_ID` | Unless `BAILIAN_API_HOST` is set | Builds the default Bailian service endpoints. |
| `MILVUS_ENDPOINT` | Yes for default vector storage | Address of the Milvus service. |
| `MILVUS_TOKEN` | Deployment-dependent | Authentication token for Milvus, if enabled. |
| `RAG_CORS_ORIGINS` | Recommended | Comma-separated frontend origins allowed to call the API. |
| `RAG_KB_RUNTIME_PATH` | No | Local path for persisted knowledge-base data. |
| `AUTH_DATABASE_URL` | No | Account database URL; defaults to local SQLite. |

Never commit `.env`, API keys, passwords, uploaded materials, or local databases. Keep provider credentials on the backend; do not put them in frontend `VITE_*` variables.

### 2. Start the API

```bash
python -m uvicorn manufacturing_rag.api:app --host 127.0.0.1 --port 8000 --reload
```

The API is available at `http://127.0.0.1:8000`. FastAPI's interactive API reference is at `http://127.0.0.1:8000/docs`, and the health endpoint is `http://127.0.0.1:8000/health`.

### 3. Start the frontend

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. For a backend on another host or port, set `VITE_API_BASE_URL` in the frontend environment and allow the frontend origin in the backend's `RAG_CORS_ORIGINS` setting. While an answer is generating, you can browse documents, switch conversations, or start a new chat without cancelling it. Follow-up questions in the same conversation are queued in order, and you can stop the current answer from the composer.

## Configuration and data

`.env.example` lists the main connection settings. Model names, retrieval parameters, and optional services can be adjusted with environment variables; see [`manufacturing_rag/config.py`](manufacturing_rag/config.py) for the supported settings and defaults.

By default, uploaded knowledge-base data is stored under `tmp/`, and account data and the generated authentication secret are stored under `data/auth/`. These are local runtime data, are excluded from Git, and should be backed up and protected separately. Documents sent for embedding, reranking, visual analysis, or answer generation may be processed by the configured external providers; review their terms and data policies before uploading sensitive material.

## API overview

The main routes include `GET /health`, `POST /answer`, `POST /answer/stream`, and `POST /kb/document`. The complete, current schema is available from the running service at `/docs`.

## Development

Run the backend test suite and build the frontend with:

```bash
pytest -q
cd frontend
npm ci
npm run build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance and [SECURITY.md](SECURITY.md) for private vulnerability reporting.

## Security and privacy

Use HTTPS and explicit allowed origins when deploying outside a local development environment. Protect uploaded files, account data, and generated secrets with appropriate access controls and backups. See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## Limitations

OCR and model-generated descriptions can be wrong, especially for low-resolution, obscured, or complex diagrams. Verify extracted facts against the original material. Answer quality depends on the uploaded sources, model providers, and retrieval configuration.

## License

This project is licensed under the [MIT License](LICENSE). Third-party packages, model services, model weights, and user-provided materials remain subject to their own licenses and terms.
