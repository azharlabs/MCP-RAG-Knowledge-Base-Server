import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import aiosqlite
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from knowledge_base.defaults import ensure_assistant_defaults


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
        data = {
            "assistant_id": assistant_id,
            "filename": filename,
            "extracted_text": text,
            "gcs_link": gcs_link,
            "created_at": datetime.utcnow().isoformat(),
        }
        res = await self.documents.insert_one(data)
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

    async def create_user(self, email: str, password_hash: str, api_key: str) -> str:
        data = {
            "email": email,
            "password_hash": password_hash,
            "api_key": api_key,
            "created_at": datetime.utcnow().isoformat(),
        }
        res = await self.users.insert_one(data)
        return str(res.inserted_id)

    async def get_user_by_email(self, email: str):
        return await self.users.find_one({"email": email})

    async def get_user_by_api_key(self, api_key: str):
        return await self.users.find_one({"api_key": api_key})

    async def get_user_by_id(self, user_id: str):
        return await self.users.find_one({"_id": ObjectId(user_id)})


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
                        chunking_method TEXT DEFAULT 'fixed',
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
                await self._ensure_column(conn, "assistants", "chunking_method", "TEXT DEFAULT 'fixed'")
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
                    "SELECT id, name, system_prompt, chunk_size, overlap, chunking_method, top_k, hyde_enabled, reranker_enabled, reranker_model, reranker_top_n, secure_enabled, credentials, api_key, owner_id FROM assistants WHERE owner_id=? LIMIT ?",
                    (owner_id, limit),
                )
            else:
                cursor = await conn.execute(
                    "SELECT id, name, system_prompt, chunk_size, overlap, chunking_method, top_k, hyde_enabled, reranker_enabled, reranker_model, reranker_top_n, secure_enabled, credentials, api_key, owner_id FROM assistants LIMIT ?",
                    (limit,),
                )
            rows = await cursor.fetchall()
        return [self._deserialize_assistant_row(r) for r in rows]

    async def get_assistant(self, assistant_id: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT id, name, system_prompt, chunk_size, overlap, chunking_method, top_k, hyde_enabled, reranker_enabled, reranker_model, reranker_top_n, secure_enabled, credentials, api_key, owner_id FROM assistants WHERE id=?",
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
                INSERT INTO assistants (id, name, system_prompt, chunk_size, overlap, chunking_method, top_k, hyde_enabled, reranker_enabled, reranker_model, reranker_top_n, secure_enabled, credentials, api_key, owner_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ast_id,
                    config.name,
                    config.system_prompt,
                    config.chunk_size,
                    config.overlap,
                    config.chunking_method,
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
                UPDATE assistants SET name=?, system_prompt=?, chunk_size=?, overlap=?, chunking_method=?, top_k=?, hyde_enabled=?, reranker_enabled=?, reranker_model=?, reranker_top_n=?, secure_enabled=?, credentials=?, api_key=?, owner_id=? WHERE id=?
                """,
                (
                    config.name,
                    config.system_prompt,
                    config.chunk_size,
                    config.overlap,
                    config.chunking_method,
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
        return [dict(row) for row in rows]

    async def get_document(self, doc_id: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,))
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
            await conn.execute("UPDATE documents SET extracted_text=? WHERE id=?", (text, doc_id))
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
            cursor = await conn.execute("SELECT * FROM documents WHERE assistant_id=?", (assistant_id,))
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def set_api_key(self, assistant_id: str, api_key: str):
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("UPDATE assistants SET api_key=? WHERE id=?", (api_key, assistant_id))
            await conn.commit()

    async def count_users(self) -> int:
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM users")
            row = await cursor.fetchone()
        return int(row[0] or 0)

    async def count_documents_for_assistant(self, assistant_id: str) -> int:
        await self.ensure_tables()
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM documents WHERE assistant_id=?", (assistant_id,))
            row = await cursor.fetchone()
        return int(row[0] or 0)

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
