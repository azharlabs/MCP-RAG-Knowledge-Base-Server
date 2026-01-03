from typing import Dict, List, Optional
from uuid import uuid4

from qdrant_client.http import models

from knowledge_base.auth import get_user_by_api_key, is_authorized
from knowledge_base.chunking import chunk_text
from knowledge_base.defaults import ensure_assistant_defaults
from knowledge_base.models import KnowledgeBaseConfig
from knowledge_base.runtime import COLLECTION_NAME, client, datastore, qdrant
from knowledge_base.settings import EMBEDDING_DIMENSIONS, MODEL_NAME


def get_embedding(text: str) -> List[float]:
    clean_text = text.replace("\n", " ")
    response = client.embeddings.create(input=[clean_text], model="text-embedding-004")
    return response.data[0].embedding


async def ingest_document_logic(
    assistant_id: str,
    doc_id: str,
    text: str,
    filename: str,
    chunk_size: int,
    overlap: int,
    chunking_method: str = "fixed",
):
    chunks = chunk_text(text, chunk_size, overlap, chunking_method=chunking_method)
    points = []
    for chunk in chunks:
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=get_embedding(chunk),
                payload={
                    "assistant_id": assistant_id,
                    "document_id": doc_id,
                    "filename": filename,
                    "text": chunk,
                },
            )
        )
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def delete_vectors_by_doc_id(doc_id: str):
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=doc_id))]
            )
        ),
    )


async def fetch_assistant(assistant_id: str):
    ast = await datastore.get_assistant(assistant_id)
    return await ensure_assistant_defaults(ast, datastore) if ast else None


async def list_assistants_logic(limit: int = 100, owner_id: Optional[str] = None):
    items = await datastore.list_assistants(limit, owner_id=owner_id)
    normalized = []
    for ast in items:
        normalized.append(await ensure_assistant_defaults(ast, datastore))
    return normalized


async def get_dashboard_metrics(owner_id: str):
    knowledge_bases = await list_assistants_logic(owner_id=owner_id)
    doc_count = 0
    for kb in knowledge_bases:
        doc_count += await datastore.count_documents_for_assistant(kb.get("id"))
    user_count = await datastore.count_users()
    return {
        "knowledge_bases": len(knowledge_bases),
        "documents": doc_count,
        "users": user_count,
    }


async def create_assistant_logic(config: KnowledgeBaseConfig, owner_id: str) -> str:
    config.owner_id = owner_id
    if not config.api_key:
        from knowledge_base.utils import generate_api_key

        config.api_key = generate_api_key()
    if config.credentials is None:
        config.credentials = []
    config.embedding_dimensions = config.embedding_dimensions or EMBEDDING_DIMENSIONS
    if config.embedding_dimensions != EMBEDDING_DIMENSIONS:
        config.embedding_dimensions = EMBEDDING_DIMENSIONS
    return await datastore.create_assistant(config)


async def update_assistant_logic(config: KnowledgeBaseConfig, owner_id: str):
    if not config.id:
        raise ValueError("Knowledge base ID is required for update")
    existing = await fetch_assistant(config.id)
    if not existing:
        raise ValueError("Knowledge base not found")
    if existing.get("owner_id") != owner_id:
        raise ValueError("Not permitted to update this knowledge base")
    if existing.get("embedding_dimensions") and config.embedding_dimensions != existing.get(
        "embedding_dimensions"
    ):
        raise ValueError("Embedding dimensions cannot be changed after creation.")
    config.embedding_dimensions = existing.get("embedding_dimensions") or EMBEDDING_DIMENSIONS
    config.owner_id = owner_id
    if not config.api_key:
        config.api_key = existing.get("api_key")
    if config.credentials is None:
        config.credentials = []
    return await datastore.update_assistant(config)


async def add_text_document(
    assistant_id: str,
    text: str,
    filename: str,
    gcs_link: str = "",
    owner_id: Optional[str] = None,
) -> str:
    ast = await fetch_assistant(assistant_id)
    if not ast:
        raise ValueError("Knowledge base not found")
    if ast.get("owner_id") and ast.get("owner_id") != owner_id:
        raise ValueError("Not permitted to modify this knowledge base")

    doc_id = await datastore.create_document(assistant_id, filename, text, gcs_link)

    await ingest_document_logic(
        assistant_id,
        doc_id,
        text,
        filename,
        ast.get("chunk_size", 1000),
        ast.get("overlap", 200),
        ast.get("chunking_method", "fixed"),
    )
    return doc_id


async def list_documents_logic(assistant_id: str, owner_id: Optional[str] = None):
    ast = await fetch_assistant(assistant_id)
    if ast and owner_id and ast.get("owner_id") != owner_id:
        raise ValueError("Not permitted to access this knowledge base")
    return await datastore.list_documents(assistant_id)


async def get_document_logic(doc_id: str, owner_id: Optional[str] = None):
    doc = await datastore.get_document(doc_id)
    if not doc:
        return None
    ast = await fetch_assistant(doc["assistant_id"])
    if owner_id and ast and ast.get("owner_id") != owner_id:
        raise ValueError("Not permitted to access this document")
    return doc


async def update_document_text_logic(doc_id: str, new_text: str, owner_id: Optional[str] = None):
    doc = await datastore.get_document(doc_id)
    if not doc:
        raise ValueError("Document not found")
    ast = await fetch_assistant(doc["assistant_id"])
    if not ast:
        raise ValueError("Knowledge base not found")
    if owner_id and ast.get("owner_id") != owner_id:
        raise ValueError("Not permitted to modify this knowledge base")

    await datastore.update_document_text(doc_id, new_text)
    delete_vectors_by_doc_id(doc_id)
    await ingest_document_logic(
        doc["assistant_id"],
        doc_id,
        new_text,
        doc["filename"],
        ast.get("chunk_size", 1000),
        ast.get("overlap", 200),
        ast.get("chunking_method", "fixed"),
    )


async def reindex_document_logic(doc_id: str, owner_id: Optional[str] = None):
    doc = await datastore.get_document(doc_id)
    if not doc:
        raise ValueError("Document not found")
    ast = await fetch_assistant(doc["assistant_id"])
    if not ast:
        raise ValueError("Knowledge base not found")
    if owner_id and ast.get("owner_id") != owner_id:
        raise ValueError("Not permitted to modify this knowledge base")
    delete_vectors_by_doc_id(doc_id)
    await ingest_document_logic(
        doc["assistant_id"],
        doc_id,
        doc["extracted_text"],
        doc["filename"],
        ast.get("chunk_size", 1000),
        ast.get("overlap", 200),
        ast.get("chunking_method", "fixed"),
    )


async def delete_document_logic(doc_id: str, owner_id: Optional[str] = None):
    doc = await datastore.get_document(doc_id)
    if doc:
        ast = await fetch_assistant(doc["assistant_id"])
        if owner_id and ast and ast.get("owner_id") != owner_id:
            raise ValueError("Not permitted to delete this document")
    delete_vectors_by_doc_id(doc_id)
    await datastore.delete_document(doc_id)


async def delete_assistant_logic(assistant_id: str, owner_id: str):
    ast = await fetch_assistant(assistant_id)
    if not ast:
        raise ValueError("Knowledge base not found")
    if ast.get("owner_id") != owner_id:
        raise ValueError("Not permitted to delete this knowledge base")
    docs = await datastore.documents_for_assistant(assistant_id)
    for doc in docs:
        delete_vectors_by_doc_id(doc["id"])
        await datastore.delete_document(doc["id"])
    await datastore.delete_assistant(assistant_id)


async def chat_with_assistant(
    assistant_id: str,
    query: str,
    email: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
):
    return await chat_with_assistant_with_history(assistant_id, query, None, email, password, api_key)


async def chat_with_assistant_with_history(
    assistant_id: str,
    query: str,
    history: Optional[List[Dict[str, str]]],
    email: Optional[str],
    password: Optional[str],
    api_key: Optional[str],
):
    ast = await fetch_assistant(assistant_id)
    if not ast:
        raise ValueError("Knowledge base not found")
    owner_override = False
    if api_key:
        user = await get_user_by_api_key(api_key)
        if user and user.get("id") == ast.get("owner_id"):
            owner_override = True
    if not is_authorized(ast, email, password, api_key, owner_override):
        raise ValueError("Unauthorized for this knowledge base")

    embedding_source = query
    if ast.get("hyde_enabled"):
        try:
            hyde_prompt = [
                {
                    "role": "system",
                    "content": "Generate a concise hypothetical answer to the user's question to improve document retrieval. Keep it factual and under 120 words.",
                },
                {"role": "user", "content": query},
            ]
            hyde_resp = client.chat.completions.create(model=MODEL_NAME, messages=hyde_prompt)
            hyp = hyde_resp.choices[0].message.content
            if hyp:
                embedding_source = hyp
        except Exception as exc:
            print(f"HyDE generation failed: {exc}")
            embedding_source = query

    q_vec = get_embedding(embedding_source)
    search_limit = ast.get("top_k", 5)
    try:
        if ast.get("reranker_enabled"):
            desired = int(ast.get("reranker_top_n", 3) or 3)
            search_limit = max(search_limit, desired)
    except Exception:
        pass
    search = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vec,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="assistant_id", match=models.MatchValue(value=assistant_id))]
        ),
        limit=search_limit,
    )

    hits = list(search)
    rerank_scores = {}

    if ast.get("reranker_enabled") and hits:
        rerank_model = ast.get("reranker_model") or MODEL_NAME
        desired_n = int(ast.get("reranker_top_n", 3) or 3)
        reranked = []
        for h in hits:
            passage = h.payload.get("text", "")
            try:
                rerank_prompt = [
                    {
                        "role": "system",
                        "content": "Score passage relevance to the query from 0 to 1. Respond with a single number only.",
                    },
                    {"role": "user", "content": f"Query: {query}\n\nPassage:\n{passage}"},
                ]
                resp = client.chat.completions.create(model=rerank_model, messages=rerank_prompt)
                raw = resp.choices[0].message.content.strip()
                score = float(raw.split()[0])
            except Exception as exc:
                print(f"Reranker scoring failed: {exc}")
                score = h.score or 0.0
            reranked.append((score, h))
            rerank_scores[str(h.id)] = score

        reranked.sort(key=lambda x: x[0], reverse=True)
        hits = [h for _, h in reranked[: max(1, desired_n)]]

    context = "\n".join([h.payload["text"] for h in hits])
    sources = list(set([h.payload["filename"] for h in hits]))
    contexts = [
        {"text": h.payload["text"], "filename": h.payload.get("filename", ""), "score": rerank_scores.get(str(h.id), h.score)}
        for h in hits
    ]

    msgs = [{"role": "system", "content": ast["system_prompt"]}]
    if history:
        msgs.extend(history[-10:])
    msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"})

    resp = client.chat.completions.create(model=MODEL_NAME, messages=msgs)
    return {"answer": resp.choices[0].message.content, "sources": sources, "contexts": contexts}


async def duplicate_assistant_logic(
    assistant_id: str,
    new_name: Optional[str] = None,
    include_docs: bool = True,
) -> str:
    ast = await fetch_assistant(assistant_id)
    if not ast:
        raise ValueError("Knowledge base not found")

    base_name = ast["name"]
    copy_name = new_name or f"{base_name} (Copy)"
    config = KnowledgeBaseConfig(
        name=copy_name,
        system_prompt=ast["system_prompt"],
        chunk_size=ast.get("chunk_size", 1000),
        overlap=ast.get("overlap", 200),
        chunking_method=ast.get("chunking_method", "fixed"),
        embedding_dimensions=ast.get("embedding_dimensions", EMBEDDING_DIMENSIONS),
        top_k=ast.get("top_k", 5),
        hyde_enabled=ast.get("hyde_enabled", False),
        reranker_enabled=ast.get("reranker_enabled", False),
        reranker_model=ast.get("reranker_model"),
        reranker_top_n=ast.get("reranker_top_n", 3),
        secure_enabled=ast.get("secure_enabled", False),
        credentials=ast.get("credentials", []),
        owner_id=ast.get("owner_id"),
        api_key=None,
    )
    new_id = await create_assistant_logic(config, owner_id=ast.get("owner_id", ""))

    if include_docs:
        docs = await datastore.documents_for_assistant(assistant_id)
        for doc in docs:
            text = doc.get("extracted_text", "")
            filename = doc.get("filename", "document.txt")
            await add_text_document(new_id, text, filename, gcs_link=doc.get("gcs_link", ""))

    return new_id
