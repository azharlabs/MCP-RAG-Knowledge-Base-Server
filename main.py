import os
import shutil
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from knowledge_base_logic import (
    KnowledgeBaseConfig,
    KnowledgeBaseCopyRequest,
    ChatRequest,
    DocUpdate,
    add_text_to_knowledge_base,
    chat_with_knowledge_base,
    chat_with_knowledge_base_with_history,
    create_knowledge_base_logic,
    delete_knowledge_base_logic,
    delete_knowledge_base_document,
    duplicate_knowledge_base_logic,
    fetch_knowledge_base,
    get_knowledge_base_document,
    list_knowledge_bases_logic,
    list_knowledge_base_documents,
    login_user,
    reindex_knowledge_base_document,
    register_user,
    store_file,
    update_knowledge_base_logic,
    update_knowledge_base_document_text,
    get_user_by_api_key,
    get_dashboard_metrics,
    get_user_by_id,
    extract_text_with_extractanything,
)

from mcp_server import server as mcp_server

app = FastAPI(title="MCP RAG")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))

# Mount MCP (FastMCP) on the same FastAPI server (single process/port).
# Primary mount (for dashboard/explicit clients): /mcp/sse
app.mount("/mcp", mcp_server.http_app(transport="sse"))
# Convenience mount so clients requesting `/sse` still reach the MCP SSE endpoint.
app.mount("/sse", mcp_server.http_app(transport="sse", path="/"))

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/register")
async def register(payload: dict):
    email = payload.get("email", "").lower().strip()
    password = payload.get("password", "")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    try:
        return await register_user(email, password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/login")
async def login(payload: dict):
    email = payload.get("email", "").lower().strip()
    password = payload.get("password", "")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    try:
        return await login_user(email, password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/metrics")
async def metrics(request: Request):
    user = await require_user(request)
    return await get_dashboard_metrics(user["id"])

async def require_user(request):
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    else:
        token = request.headers.get("x-api-key", "")
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")
    user = await get_user_by_api_key(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user


async def optional_user(request):
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    else:
        token = request.headers.get("x-api-key", "")
    if not token:
        return None
    return await get_user_by_api_key(token)

# --- Models ---
# --- API Endpoints ---


@app.get("/api/knowledge-bases")
async def list_knowledge_bases(request: Request):
    user = await require_user(request)
    return await list_knowledge_bases_logic(owner_id=user["id"])


@app.get("/api/knowledge-base/{knowledge_base_id}/public")
async def get_knowledge_base_public(knowledge_base_id: str):
    """Public endpoint to get basic knowledge base info for shared chat page"""
    kb = await fetch_knowledge_base(knowledge_base_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    owner_email = None
    if kb.get("owner_id"):
        owner = await get_user_by_id(kb.get("owner_id"))
        if owner:
            owner_email = owner.get("email")

    return {
        "id": kb.get("id"),
        "name": kb.get("name"),
        "secure_enabled": kb.get("secure_enabled", False),
        "owner_email": owner_email,
    }


@app.post("/api/assistant")
async def save_assistant(config: KnowledgeBaseConfig, request: Request):
    user = await require_user(request)
    if config.id:
        await update_knowledge_base_logic(config, owner_id=user["id"])
        return {"id": config.id, "message": "Updated"}
    new_id = await create_knowledge_base_logic(config, owner_id=user["id"])
    return {"id": new_id, "message": "Created"}

@app.post("/api/knowledge-base")
async def save_knowledge_base(config: KnowledgeBaseConfig, request: Request):
    user = await require_user(request)
    if config.id:
        await update_knowledge_base_logic(config, owner_id=user["id"])
        return {"id": config.id, "message": "Updated"}
    new_id = await create_knowledge_base_logic(config, owner_id=user["id"])
    return {"id": new_id, "message": "Created"}


@app.post("/api/knowledge-base/{knowledge_base_id}/copy")
async def copy_knowledge_base(knowledge_base_id: str, payload: KnowledgeBaseCopyRequest, request: Request):
    user = await require_user(request)
    try:
        kb = await fetch_knowledge_base(knowledge_base_id)
        if not kb or kb.get("owner_id") != user["id"]:
            raise HTTPException(status_code=403, detail="Not permitted")
        new_id = await duplicate_knowledge_base_logic(
            knowledge_base_id,
            new_name=payload.name,
            include_docs=payload.include_docs,
        )
        return {"id": new_id, "message": "Copied"}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

@app.delete("/api/knowledge-base/{knowledge_base_id}")
async def delete_knowledge_base(knowledge_base_id: str, request: Request):
    user = await require_user(request)
    kb = await fetch_knowledge_base(knowledge_base_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if kb.get("owner_id") != user["id"]:
        raise HTTPException(status_code=403, detail="Not permitted")
    await delete_knowledge_base_logic(knowledge_base_id, owner_id=user["id"])
    return {"status": "deleted"}

@app.post("/api/knowledge-base/{knowledge_base_id}/upload")
async def upload_file_to_knowledge_base(knowledge_base_id: str, request: Request, file: UploadFile = File(...)):
    user = await require_user(request)
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File too large. Max size is {max_mb:.0f} MB.")
    file_extension = Path(file.filename).suffix or ".dat"
    temp_name = f"temp_{uuid4()}{file_extension}"
    with open(temp_name, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = extract_text_with_extractanything(temp_name)
    except Exception as exc:
        os.remove(temp_name)
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {exc}")

    file_location = store_file(temp_name, knowledge_base_id, file.filename)
    os.remove(temp_name)

    doc_id = await add_text_to_knowledge_base(
        knowledge_base_id,
        text,
        file.filename,
        gcs_link=file_location,
        owner_id=user["id"],
    )
    return {"status": "success", "doc_id": doc_id}

@app.get("/api/knowledge-base/{knowledge_base_id}/documents")
async def get_docs_for_knowledge_base(knowledge_base_id: str, request: Request):
    user = await require_user(request)
    try:
        return await list_knowledge_base_documents(knowledge_base_id, owner_id=user["id"])
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

@app.get("/api/document/{doc_id}")
async def get_single_doc(doc_id: str, request: Request):
    user = await require_user(request)
    doc = await get_knowledge_base_document(doc_id, owner_id=user["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"id": doc["id"], "text": doc.get("extracted_text", "")}

@app.put("/api/document/{doc_id}")
async def update_doc_text(doc_id: str, update: DocUpdate, request: Request):
    user = await require_user(request)
    try:
        await update_knowledge_base_document_text(doc_id, update.extracted_text, owner_id=user["id"])
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "updated"}

@app.post("/api/document/{doc_id}/reindex")
async def reindex_doc(doc_id: str, request: Request):
    user = await require_user(request)
    try:
        await reindex_knowledge_base_document(doc_id, owner_id=user["id"])
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "reindexed"}

@app.delete("/api/document/{doc_id}")
async def delete_doc(doc_id: str, request: Request):
    user = await require_user(request)
    await delete_knowledge_base_document(doc_id, owner_id=user["id"])
    return {"status": "deleted"}

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    user = await optional_user(request)
    user_api_key = user["api_key"] if user else None
    try:
        if req.history:
            return await chat_with_knowledge_base_with_history(
                req.assistant_id,
                req.query,
                req.history,
                req.email,
                req.password,
                req.api_key or user_api_key,
            )
        return await chat_with_knowledge_base(req.assistant_id, req.query, req.email, req.password, req.api_key or user_api_key)
    except ValueError as exc:
        status = 403 if "Unauthorized" in str(exc) else 404
        raise HTTPException(status_code=status, detail=str(exc))

# --- Serve Static ---
# Ensure the 'static' directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
