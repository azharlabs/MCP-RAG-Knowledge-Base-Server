# MCP RAG Knowledge Base Server

FastAPI app and MCP server that expose Retrieval-Augmented Generation (RAG) knowledge bases for chat and document ingestion. Serves a small dashboard, shares a public chat page, and mounts the MCP SSE transport on the same port.

## Features
- Knowledge base CRUD with document ingestion (PDF → text) and vector search (Qdrant).
- Public chat page for a shared knowledge base; secure mode with API key or email/password.
- Dashboard UI to manage knowledge bases, documents, and API keys.
- MCP tools for listing knowledge bases, chatting, and uploading documents; assistant-named aliases kept for compatibility.
- Single-process FastAPI + MCP server; SSE available at `/mcp/sse` and `/sse`.

## Quick start
1) Install deps:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
2) Set env vars (create a `.env` if you like):
- `MODEL_API_KEY` (required) – key for the configured LLM (defaults to Gemini-compatible OpenAI endpoint).
- `MODEL_BASE_URL`/config.yaml overrides if needed.
- Optional: `QDRANT_URL` (otherwise in-memory), `GCS_BUCKET_NAME` (if storage.type=gcs in config.yaml).
3) Run server:
```bash
python serve.py
```
4) Open the dashboard at `http://localhost:8000/` (register/login to get a user API key).
5) Share chat links: `http://localhost:8000/static/chat.html?knowledgeBaseId=<id>`.

## Configuration
- `config.yaml` controls model, storage, qdrant, env name, etc. Loaded via `config_loader.py`.
- Storage: local by default (`storage` dir). Set `storage.type: gcs` and `GCS_BUCKET_NAME` for GCS.
- Qdrant: in-memory by default; set `qdrant.type: remote` and `QDRANT_URL` for a remote instance.

## API (knowledge-base first; assistant paths kept for back-compat)
- Auth: Bearer `<user_api_key>` header.
- Knowledge bases:
  - `GET /api/knowledge-bases` (or `/api/assistants`)
  - `POST /api/knowledge-base` (or `/api/assistant`) with `KnowledgeBaseConfig`
  - `POST /api/knowledge-base/{id}/copy` (or `/api/assistant/{id}/copy`)
  - `DELETE /api/knowledge-base/{id}` (or `/api/assistant/{id}`)
  - `GET /api/knowledge-base/{id}/public` (or `/api/assistant/{id}/public`)
- Documents:
  - `GET /api/knowledge-base/{id}/documents`
  - `POST /api/knowledge-base/{id}/upload` (PDF file)
  - `GET/PUT/POST/DELETE /api/document/{doc_id}` (fetch/update/reindex/delete)
- Chat:
  - `POST /api/chat` with `{"knowledge_base_id": "<id>", "query": "...", "history": [...]}` and optional `api_key`/email/password for secure bases.
- Auth helpers: `POST /api/register`, `POST /api/login`, `GET /api/metrics`.

## MCP usage
Mounted at `/mcp` (SSE at `/mcp/sse`) and `/sse` shortcut.
- `list_knowledge_bases(api_key)` – lists current user’s knowledge bases.
- `list_assistants` – alias for compatibility.
- `chat(knowledge_base_id, query, history?, api_key?)` – chat with a knowledge base.
- `add_document(knowledge_base_id, text, filename?, api_key)` – ingest raw text.
- Resource lists: `knowledge-bases://list` (and `assistants://list` alias).

## Frontend notes
- Dashboard served at `/`; static assets under `/static`.
- Public chat page: `/static/chat.html?knowledgeBaseId=<id>`; falls back to `assistantId` in the query for old links.
- UI text refers to “Knowledge Bases”; assistant terms remain only for compatibility.

## Development tips
- Run lint/format as needed; no formatter enforced here.
- If you change `mcp_servers.py` (core logic), restart the server.
- The MCP client may expect `knowledge_base_id`; assistant_id params are aliases only for older clients.
