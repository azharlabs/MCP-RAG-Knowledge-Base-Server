import asyncio
import json
import os
import secrets
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import aiosqlite
from bson import ObjectId
from dotenv import load_dotenv
from google.cloud import storage
from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI
from pydantic import BaseModel, root_validator
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config_loader import load_config
from embed_anything import EmbedAnything

# Load environment variables from .env file
load_dotenv()

CONFIG = load_config()
ENV_NAME = CONFIG.get("env", "dev")

# --- Model Settings ---
MODEL_CONFIG = CONFIG.get("model", {})
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_BASE_URL = MODEL_CONFIG.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = MODEL_CONFIG.get("name", "gemini-2.5-flash")

if not MODEL_API_KEY:
    raise RuntimeError("Missing MODEL_API_KEY in environment/.env")

client = OpenAI(api_key=MODEL_API_KEY, base_url=MODEL_BASE_URL)

EMBEDDING_DIMENSIONS = 768
embed_extractor = EmbedAnything()

# --- Storage Settings ---
STORAGE_CONFIG = CONFIG.get("storage", {})
STORAGE_TYPE = STORAGE_CONFIG.get("type", "local")
LOCAL_STORAGE_PATH: Optional[Path] = None
bucket = None
if STORAGE_TYPE == "gcs":
    bucket_name = STORAGE_CONFIG.get("bucket") or os.getenv("GCS_BUCKET_NAME", "")
    if bucket_name:
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
        except Exception:
            bucket = None
            print("GCS not configured correctly; falling back to local storage.")
    else:
        print("GCS bucket name not provided; falling back to local storage.")
        STORAGE_TYPE = "local"

if STORAGE_TYPE == "local":
    LOCAL_STORAGE_PATH = Path(STORAGE_CONFIG.get("base_path", "storage"))
    LOCAL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)


# --- Database Stores ---
class MongoDataStore:
    def __init__(self, mongo_url: str):
        self.client = AsyncIOMotorClient(mongo_url)
        db = self.client["rag_db"]
        self.assistants = db["assistants"]
        self.documents = db["documents"]
        self.users = db["users"]

    async def list_assistants(self, limit: int = 100, owner_id: Optional[str] = None):
        query = {}
        if owner_id:
            query["owner_id"] = owner_id
        items = await self.assistants.find(query).to_list(limit)
        result = []
        for i in items:
            i["id"] = str(i["_id"])
            del i["_id"]
            result.append(await ensure_assistant_defaults(i, self))
        return result

    async def get_assistant(self, assistant_id: str):
        doc = await self.assistants.find_one({"_id": ObjectId(assistant_id)})
        if doc:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
        return await ensure_assistant_defaults(doc, self) if doc else None

    async def create_assistant(self, config: "KnowledgeBaseConfig") -> str:
        data = config.dict(exclude={"id"})
        res = await self.assistants.insert_one(data)
        return str(res.inserted_id)

    async def update_assistant(self, config: "KnowledgeBaseConfig"):
        existing = await self.assistants.find_one({"_id": ObjectId(config.id)})
        data = config.dict(exclude={"id"})
        if existing and not data.get("api_key"):
            data["api_key"] = existing.get("api_key")
        await self.assistants.update_one({"_id": ObjectId(config.id)}, {"$set": data})

    async def delete_assistant(self, assistant_id: str):
        await self.assistants.delete_one({"_id": ObjectId(assistant_id)})

    async def list_documents(self, assistant_id: str):
        docs = await self.documents.find({"assistant_id": assistant_id}, {"extracted_text": 0}).to_list(1000)
        for d in docs:
            d["id"] = str(d["_id"])
            del d["_id"]
        return docs

    async def get_document(self, doc_id: str):
        doc = await self.documents.find_one({"_id": ObjectId(doc_id)})
        if doc:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
        return doc

    async def create_document(self, assistant_id: str, filename: str, text: str, gcs_link: str) -> str:
        res = await self.documents.insert_one({
            "assistant_id": assistant_id,
            "filename": filename,
            "extracted_text": text,
            "gcs_link": gcs_link,
            "created_at": datetime.utcnow()
        })
        return str(res.inserted_id)

    async def update_document_text(self, doc_id: str, text: str):
        await self.documents.update_one({"_id": ObjectId(doc_id)}, {"$set": {"extracted_text": text}})

    async def delete_document(self, doc_id: str):
        await self.documents.delete_one({"_id": ObjectId(doc_id)})

    async def documents_for_assistant(self, assistant_id: str):
        docs = await self.documents.find({"assistant_id": assistant_id}).to_list(1000)
        for d in docs:
            d["id"] = str(d["_id"])
            del d["_id"]
        return docs

    async def set_api_key(self, assistant_id: str, api_key: str):
        await self.assistants.update_one({"_id": ObjectId(assistant_id)}, {"$set": {"api_key": api_key}})

    async def count_users(self) -> int:
        return await self.users.count_documents({})

    async def count_documents_for_assistant(self, assistant_id: str) -> int:
        return await self.documents.count_documents({"assistant_id": assistant_id})

    # User helpers
    async def create_user(self, email: str, password_hash: str, api_key: str) -> str:
        res = await self.users.insert_one({
            "email": email,
            "password_hash": password_hash,
            "api_key": api_key,
            "created_at": datetime.utcnow()
        })
        return str(res.inserted_id)

    async def get_user_by_email(self, email: str):
        user = await self.users.find_one({"email": email})
        if user:
            user["id"] = str(user["_id"])
            del user["_id"]
        return user

    async def get_user_by_api_key(self, api_key: str):
        user = await self.users.find_one({"api_key": api_key})
        if user:
            user["id"] = str(user["_id"])
            del user["_id"]
        return user

    async def get_user_by_id(self, user_id: str):
        from bson import ObjectId
        user = await self.users.find_one({"_id": ObjectId(user_id)})
        if user:
            user["id"] = str(user["_id"])
            del user["_id"]
        return user


class SQLiteDataStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def ensure_tables(self):
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS assistants (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        system_prompt TEXT NOT NULL,
                        chunk_size INTEGER NOT NULL,
                        overlap INTEGER NOT NULL,
                        top_k INTEGER NOT NULL,
                        hyde_enabled INTEGER DEFAULT 0,
                        reranker_enabled INTEGER DEFAULT 0,
                        reranker_model TEXT,
                        reranker_top_n INTEGER DEFAULT 3,
                        secure_enabled INTEGER DEFAULT 0,
                        credentials TEXT DEFAULT '[]',
                        api_key TEXT,
                        owner_id TEXT,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                await self._ensure_column(conn, "assistants", "hyde_enabled", "INTEGER DEFAULT 0")
                await self._ensure_column(conn, "assistants", "reranker_enabled", "INTEGER DEFAULT 0")
                await self._ensure_column(conn, "assistants", "reranker_model", "TEXT")
                await self._ensure_column(conn, "assistants", "reranker_top_n", "INTEGER DEFAULT 3")
                await self._ensure_column(conn, "assistants", "secure_enabled", "INTEGER DEFAULT 0")
                await self._ensure_column(conn, "assistants", "credentials", "TEXT DEFAULT '[]'")
                await self._ensure_column(conn, "assistants", "api_key", "TEXT")
                await self._ensure_column(conn, "assistants", "owner_id", "TEXT")
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        api_key TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        assistant_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        extracted_text TEXT,
                        gcs_link TEXT,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                await conn.commit()
            self._initialized = True

    async def _ensure_column(self, conn, table: str, column: str, column_def: str):
        cursor = await conn.execute(f"PRAGMA table_info({table});")
        cols = [row[1] for row in await cursor.fetchall()]
        if column not in cols:
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def};")

    async def list_assistants(self, limit: int = 100, owner_id: Optional[str] = None):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            if owner_id:
                cursor = await conn.execute(
                    "SELECT id, name, system_prompt, chunk_size, overlap, top_k, hyde_enabled, reranker_enabled, reranker_model, reranker_top_n, secure_enabled, credentials, api_key, owner_id FROM assistants WHERE owner_id=? LIMIT ?",
                    (owner_id, limit),
                )
            else:
                cursor = await conn.execute(
                    "SELECT id, name, system_prompt, chunk_size, overlap, top_k, hyde_enabled, reranker_enabled, reranker_model, reranker_top_n, secure_enabled, credentials, api_key, owner_id FROM assistants LIMIT ?",
                    (limit,),
                )
            rows = await cursor.fetchall()
        return [self._deserialize_assistant_row(r) for r in rows]

    async def get_assistant(self, assistant_id: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT id, name, system_prompt, chunk_size, overlap, top_k, hyde_enabled, reranker_enabled, reranker_model, reranker_top_n, secure_enabled, credentials, api_key, owner_id FROM assistants WHERE id=?",
                (assistant_id,),
            )
            row = await cursor.fetchone()
        return self._deserialize_assistant_row(row) if row else None

    async def create_assistant(self, config: "KnowledgeBaseConfig") -> str:
        await self.ensure_tables()
        ast_id = config.id or uuid4().hex
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT INTO assistants (id, name, system_prompt, chunk_size, overlap, top_k, hyde_enabled, reranker_enabled, reranker_model, reranker_top_n, secure_enabled, credentials, api_key, owner_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ast_id,
                    config.name,
                    config.system_prompt,
                    config.chunk_size,
                    config.overlap,
                    config.top_k,
                    1 if getattr(config, "hyde_enabled", False) else 0,
                    1 if getattr(config, "reranker_enabled", False) else 0,
                    config.reranker_model,
                    config.reranker_top_n,
                    1 if config.secure_enabled else 0,
                    json.dumps(config.credentials or []),
                    config.api_key,
                    config.owner_id,
                    datetime.utcnow().isoformat(),
                ),
            )
            await conn.commit()
        return ast_id

    async def update_assistant(self, config: "KnowledgeBaseConfig"):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                UPDATE assistants SET name=?, system_prompt=?, chunk_size=?, overlap=?, top_k=?, hyde_enabled=?, reranker_enabled=?, reranker_model=?, reranker_top_n=?, secure_enabled=?, credentials=?, api_key=?, owner_id=? WHERE id=?
                """,
                (
                    config.name,
                    config.system_prompt,
                    config.chunk_size,
                    config.overlap,
                    config.top_k,
                    1 if getattr(config, "hyde_enabled", False) else 0,
                    1 if getattr(config, "reranker_enabled", False) else 0,
                    config.reranker_model,
                    config.reranker_top_n,
                    1 if config.secure_enabled else 0,
                    json.dumps(config.credentials or []),
                    config.api_key,
                    config.owner_id,
                    config.id,
                ),
            )
            await conn.commit()

    async def delete_assistant(self, assistant_id: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("DELETE FROM assistants WHERE id=?", (assistant_id,))
            await conn.commit()

    async def list_documents(self, assistant_id: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT id, assistant_id, filename, gcs_link, created_at FROM documents WHERE assistant_id=?",
                (assistant_id,),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_document(self, doc_id: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM documents WHERE id=?",
                (doc_id,),
            )
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def create_document(self, assistant_id: str, filename: str, text: str, gcs_link: str) -> str:
        await self.ensure_tables()
        doc_id = uuid4().hex
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT INTO documents (id, assistant_id, filename, extracted_text, gcs_link, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    assistant_id,
                    filename,
                    text,
                    gcs_link,
                    datetime.utcnow().isoformat(),
                ),
            )
            await conn.commit()
        return doc_id

    async def update_document_text(self, doc_id: str, text: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                "UPDATE documents SET extracted_text=? WHERE id=?",
                (text, doc_id),
            )
            await conn.commit()

    async def delete_document(self, doc_id: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            await conn.commit()

    async def documents_for_assistant(self, assistant_id: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM documents WHERE assistant_id=?",
                (assistant_id,),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def set_api_key(self, assistant_id: str, api_key: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                "UPDATE assistants SET api_key=? WHERE id=?",
                (api_key, assistant_id),
            )
            await conn.commit()

    async def count_users(self) -> int:
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM users")
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def count_documents_for_assistant(self, assistant_id: str) -> int:
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM documents WHERE assistant_id=?", (assistant_id,))
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def create_user(self, email: str, password_hash: str, api_key: str) -> str:
        await self.ensure_tables()
        user_id = uuid4().hex
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT INTO users (id, email, password_hash, api_key, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, email, password_hash, api_key, datetime.utcnow().isoformat()),
            )
            await conn.commit()
        return user_id

    async def get_user_by_email(self, email: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM users WHERE email=?", (email,))
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_user_by_api_key(self, api_key: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM users WHERE api_key=?", (api_key,))
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_user_by_id(self, user_id: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM users WHERE id=?", (user_id,))
            row = await cursor.fetchone()
        return dict(row) if row else None

    def _deserialize_assistant_row(self, row: aiosqlite.Row):
        data = dict(row)
        data["hyde_enabled"] = bool(data.get("hyde_enabled"))
        data["reranker_enabled"] = bool(data.get("reranker_enabled"))
        try:
            data["reranker_top_n"] = int(data.get("reranker_top_n"))
        except Exception:
            data["reranker_top_n"] = 3
        data["secure_enabled"] = bool(data.get("secure_enabled"))
        creds_raw = data.get("credentials") or "[]"
        try:
            data["credentials"] = json.loads(creds_raw)
        except Exception:
            data["credentials"] = []
        return data


DATABASE_CONFIG = CONFIG.get("database", {})
db_type = DATABASE_CONFIG.get("type", "mongo")
if db_type == "mongo":
    MONGO_URL = DATABASE_CONFIG.get("mongo_url") or os.getenv("MONGO_URL", "")
    if not MONGO_URL:
        raise RuntimeError("Missing MONGO_URL for Mongo database configuration")
    datastore = MongoDataStore(MONGO_URL)
elif db_type == "sqlite":
    db_path = Path(DATABASE_CONFIG.get("path", "data/dev.db"))
    datastore = SQLiteDataStore(db_path)
else:
    raise RuntimeError(f"Unsupported database type: {db_type}")


async def ensure_assistant_defaults(ast: dict, store) -> dict:
    if ast is None:
        return ast
    changed = False
    if "hyde_enabled" not in ast:
        ast["hyde_enabled"] = False
        changed = True
    if "reranker_enabled" not in ast:
        ast["reranker_enabled"] = False
        changed = True
    if "reranker_top_n" not in ast or ast.get("reranker_top_n") is None:
        ast["reranker_top_n"] = 3
        changed = True
    if "reranker_model" not in ast:
        ast["reranker_model"] = None
    if "secure_enabled" not in ast:
        ast["secure_enabled"] = False
        changed = True
    if not ast.get("embedding_dimensions"):
        ast["embedding_dimensions"] = EMBEDDING_DIMENSIONS
        changed = True
    if ast.get("credentials") is None:
        ast["credentials"] = []
    if not ast.get("api_key"):
        ast["api_key"] = generate_api_key()
        changed = True
        if hasattr(store, "set_api_key") and ast.get("id"):
            await store.set_api_key(ast["id"], ast["api_key"])
    return ast


# --- User Auth Helpers ---
async def register_user(email: str, password: str):
    existing = await datastore.get_user_by_email(email)
    if existing:
        raise ValueError("User already exists")
    api_key = generate_api_key()
    password_hash = hash_password(password)
    user_id = await datastore.create_user(email, password_hash, api_key)
    return {"id": user_id, "email": email, "api_key": api_key}


async def login_user(email: str, password: str):
    user = await datastore.get_user_by_email(email)
    if not user:
        raise ValueError("Invalid credentials")
    if user.get("password_hash") != hash_password(password):
        raise ValueError("Invalid credentials")
    return {"id": user.get("id"), "email": user.get("email"), "api_key": user.get("api_key")}


async def get_user_by_api_key(api_key: str):
    return await datastore.get_user_by_api_key(api_key)


async def get_user_by_id(user_id: str):
    return await datastore.get_user_by_id(user_id)


# --- Qdrant ---
QDRANT_CONFIG = CONFIG.get("qdrant", {})
qdrant_type = QDRANT_CONFIG.get("type", "memory")
if qdrant_type in ("local", "disk"):
    qdrant_path = QDRANT_CONFIG.get("path") or os.getenv("QDRANT_PATH") or "data/qdrant"
    Path(qdrant_path).mkdir(parents=True, exist_ok=True)
    qdrant = QdrantClient(path=qdrant_path)
elif qdrant_type == "memory":
    qdrant = QdrantClient(":memory:")
else:
    qdrant_url = QDRANT_CONFIG.get("url") or os.getenv("QDRANT_URL", "")
    if not qdrant_url:
        raise RuntimeError("Missing QDRANT_URL for remote Qdrant configuration")
    qdrant = QdrantClient(qdrant_url)

COLLECTION_NAME = "enterprise_rag"
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIMENSIONS, distance=models.Distance.COSINE)
    )


# --- Models ---
class KnowledgeBaseConfig(BaseModel):
    id: Optional[str] = None
    name: str
    system_prompt: str
    chunk_size: int = 1000
    overlap: int = 200
    embedding_dimensions: int = EMBEDDING_DIMENSIONS
    top_k: int = 5
    hyde_enabled: bool = False
    reranker_enabled: bool = False
    reranker_model: Optional[str] = None
    reranker_top_n: int = 3
    secure_enabled: bool = False
    credentials: Optional[List[Dict[str, str]]] = None  # list of {"email","password"}
    api_key: Optional[str] = None
    owner_id: Optional[str] = None


class DocUpdate(BaseModel):
    extracted_text: str


class ChatRequest(BaseModel):
    assistant_id: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    query: str
    history: Optional[List[Dict[str, str]]] = None
    email: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None

    @root_validator(pre=True)
    def normalize_ids(cls, values):
        if not isinstance(values, dict):
            return values
        # Accept either assistant_id/knowledge_base_id (snake) or camelCase variants.
        assistant_id = values.get("assistant_id") or values.get("assistantId")
        knowledge_base_id = values.get("knowledge_base_id") or values.get("knowledgeBaseId")

        if not assistant_id and knowledge_base_id:
            assistant_id = knowledge_base_id
        if assistant_id and not knowledge_base_id:
            knowledge_base_id = assistant_id

        values["assistant_id"] = assistant_id
        values["knowledge_base_id"] = knowledge_base_id

        if not assistant_id:
            raise ValueError("knowledge_base_id (or assistant_id) is required")
        return values


class KnowledgeBaseCopyRequest(BaseModel):
    name: Optional[str] = None
    include_docs: bool = True


# Backwards-compatible aliases
AssistantConfig = KnowledgeBaseConfig
AssistantCopyRequest = KnowledgeBaseCopyRequest


def generate_api_key() -> str:
    return secrets.token_urlsafe(24)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# --- Storage Helpers ---
def store_file(temp_path: str, assistant_id: str, filename: str) -> str:
    if STORAGE_TYPE == "gcs" and bucket:
        blob = bucket.blob(f"{assistant_id}/{filename}")
        blob.upload_from_filename(temp_path)
        return blob.public_url

    if not LOCAL_STORAGE_PATH:
        raise RuntimeError("Local storage path not configured")
    dest_dir = LOCAL_STORAGE_PATH / assistant_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    shutil.copy(temp_path, dest_path)
    return str(dest_path)


def extract_text_with_embedanything(file_path: str) -> str:
    return embed_extractor.extract_text(file_path)


# --- Logic ---
def get_embedding(text: str) -> List[float]:
    clean_text = text.replace("\n", " ")
    response = client.embeddings.create(input=[clean_text], model="text-embedding-004")
    return response.data[0].embedding


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


async def ingest_document_logic(
    assistant_id: str,
    doc_id: str,
    text: str,
    filename: str,
    chunk_size: int,
    overlap: int,
):
    chunks = chunk_text(text, chunk_size, overlap)
    points = []
    for chunk in chunks:
        points.append(models.PointStruct(
            id=str(uuid4()),
            vector=get_embedding(chunk),
            payload={
                "assistant_id": assistant_id,
                "document_id": doc_id,
                "filename": filename,
                "text": chunk
            }
        ))
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def delete_vectors_by_doc_id(doc_id: str):
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=doc_id))])
        )
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
    assistants = await list_assistants_logic(owner_id=owner_id)
    documents_count = 0
    assistants_summary = []
    for ast in assistants:
        doc_count = await datastore.count_documents_for_assistant(ast["id"])
        documents_count += doc_count
        assistants_summary.append({
            "id": ast["id"],
            "name": ast["name"],
            "documents": doc_count,
            "secure_enabled": ast.get("secure_enabled", False),
        })
    total_users = await datastore.count_users()
    return {
        "total_users": total_users,
        "knowledge_bases_count": len(assistants),
        "assistants_count": len(assistants),
        "documents_count": documents_count,
        "knowledge_bases": assistants_summary,
        "assistants": assistants_summary,
    }


async def create_assistant_logic(config: KnowledgeBaseConfig, owner_id: str) -> str:
    config.owner_id = owner_id
    if not config.api_key:
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
    if existing.get("embedding_dimensions") and config.embedding_dimensions != existing.get("embedding_dimensions"):
        raise ValueError("Embedding dimensions cannot be changed after creation.")
    config.embedding_dimensions = existing.get("embedding_dimensions") or EMBEDDING_DIMENSIONS
    config.owner_id = owner_id
    if not config.api_key:
        config.api_key = existing.get("api_key") or generate_api_key()
    if config.credentials is None:
        config.credentials = []
    return await datastore.update_assistant(config)


async def add_text_document(assistant_id: str, text: str, filename: str, gcs_link: str = "", owner_id: Optional[str] = None) -> str:
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
        ast.get("overlap", 200)
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
        ast.get("overlap", 200)
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
        ast.get("overlap", 200)
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
    return await chat_with_assistant_with_history(
        assistant_id,
        query,
        history=None,
        email=email,
        password=password,
        api_key=api_key,
    )


def is_authorized(assistant: dict, email: Optional[str], password: Optional[str], api_key: Optional[str], owner_override: bool) -> bool:
    if not assistant.get("secure_enabled"):
        return True
    if owner_override:
        return True
    if api_key and api_key == assistant.get("api_key"):
        return True
    creds = assistant.get("credentials") or []
    for cred in creds:
        if cred.get("email") == email and cred.get("password") == password:
            return True
    return False


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
                    "content": "Generate a concise hypothetical answer to the user's question to improve document retrieval. Keep it factual and under 120 words."
                },
                {"role": "user", "content": query},
            ]
            hyde_resp = client.chat.completions.create(model=MODEL_NAME, messages=hyde_prompt)
            hyp = hyde_resp.choices[0].message.content
            if hyp:
                embedding_source = hyp
        except Exception as exc:  # best-effort; fall back to direct query embedding
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
        query_filter=models.Filter(must=[models.FieldCondition(key="assistant_id", match=models.MatchValue(value=assistant_id))]),
        limit=search_limit
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
                        "content": "Score passage relevance to the query from 0 to 1. Respond with a single number only."
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nPassage:\n{passage}"
                    },
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
        hits = [h for _, h in reranked[:max(1, desired_n)]]

    context = "\n".join([h.payload["text"] for h in hits])
    sources = list(set([h.payload["filename"] for h in hits]))
    contexts = [
        {
            "text": h.payload["text"],
            "filename": h.payload.get("filename", ""),
            "score": rerank_scores.get(str(h.id), h.score)
        }
        for h in hits
    ]

    msgs = [{"role": "system", "content": ast["system_prompt"]}]
    if history:
        msgs.extend(history[-10:])
    msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"})

    resp = client.chat.completions.create(model=MODEL_NAME, messages=msgs)
    return {
        "answer": resp.choices[0].message.content,
        "sources": sources,
        "contexts": contexts
    }


async def duplicate_assistant_logic(assistant_id: str, new_name: Optional[str] = None, include_docs: bool = True) -> str:
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


# Preferred knowledge base naming (backwards-compatible aliases)
ensure_knowledge_base_defaults = ensure_assistant_defaults
fetch_knowledge_base = fetch_assistant
list_knowledge_bases_logic = list_assistants_logic
create_knowledge_base_logic = create_assistant_logic
update_knowledge_base_logic = update_assistant_logic
delete_knowledge_base_logic = delete_assistant_logic
duplicate_knowledge_base_logic = duplicate_assistant_logic
chat_with_knowledge_base = chat_with_assistant
chat_with_knowledge_base_with_history = chat_with_assistant_with_history
add_text_to_knowledge_base = add_text_document
list_knowledge_base_documents = list_documents_logic
get_knowledge_base_document = get_document_logic
update_knowledge_base_document_text = update_document_text_logic
reindex_knowledge_base_document = reindex_document_logic
delete_knowledge_base_document = delete_document_logic
