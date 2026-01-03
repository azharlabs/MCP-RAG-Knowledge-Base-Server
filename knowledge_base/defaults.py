from typing import Optional

from knowledge_base.settings import EMBEDDING_DIMENSIONS
from knowledge_base.utils import generate_api_key


async def ensure_assistant_defaults(ast: Optional[dict], store) -> Optional[dict]:
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
    if "chunking_method" not in ast or not ast.get("chunking_method"):
        ast["chunking_method"] = "fixed"
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
