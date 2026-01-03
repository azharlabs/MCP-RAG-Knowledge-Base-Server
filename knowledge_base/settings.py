import os

from dotenv import load_dotenv

from config_loader import load_config

load_dotenv()

CONFIG = load_config()
ENV_NAME = CONFIG.get("env", "dev")

MODEL_CONFIG = CONFIG.get("model", {})
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_BASE_URL = MODEL_CONFIG.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = MODEL_CONFIG.get("name", "gemini-2.5-flash")

if not MODEL_API_KEY:
    raise RuntimeError("Missing MODEL_API_KEY in environment/.env")

EMBEDDING_DIMENSIONS = 768

STORAGE_CONFIG = CONFIG.get("storage", {})
DATABASE_CONFIG = CONFIG.get("database", {})
QDRANT_CONFIG = CONFIG.get("qdrant", {})
