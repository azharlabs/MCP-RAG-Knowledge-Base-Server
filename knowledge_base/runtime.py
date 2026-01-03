import os
from pathlib import Path
from typing import Optional

from google.cloud import storage
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

from knowledge_base.datastore import MongoDataStore, SQLiteDataStore
from knowledge_base.settings import (
    DATABASE_CONFIG,
    EMBEDDING_DIMENSIONS,
    MODEL_API_KEY,
    MODEL_BASE_URL,
    QDRANT_CONFIG,
    STORAGE_CONFIG,
)

client = OpenAI(api_key=MODEL_API_KEY, base_url=MODEL_BASE_URL)

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

DATABASE_TYPE = DATABASE_CONFIG.get("type", "mongo")
if DATABASE_TYPE == "mongo":
    mongo_url = DATABASE_CONFIG.get("mongo_url") or os.getenv("MONGO_URL", "")
    if not mongo_url:
        raise RuntimeError("Missing MONGO_URL for Mongo database configuration")
    datastore = MongoDataStore(mongo_url)
elif DATABASE_TYPE == "sqlite":
    db_path = Path(DATABASE_CONFIG.get("path", "data/dev.db"))
    datastore = SQLiteDataStore(db_path)
else:
    raise RuntimeError(f"Unsupported database type: {DATABASE_TYPE}")

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
        vectors_config=models.VectorParams(size=EMBEDDING_DIMENSIONS, distance=models.Distance.COSINE),
    )
