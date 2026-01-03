from typing import Dict, List, Optional

from pydantic import BaseModel, root_validator

from knowledge_base.settings import EMBEDDING_DIMENSIONS


class KnowledgeBaseConfig(BaseModel):
    id: Optional[str] = None
    name: str
    system_prompt: str
    chunk_size: int = 1000
    overlap: int = 200
    chunking_method: str = "fixed"
    embedding_dimensions: int = EMBEDDING_DIMENSIONS
    top_k: int = 5
    hyde_enabled: bool = False
    reranker_enabled: bool = False
    reranker_model: Optional[str] = None
    reranker_top_n: int = 3
    secure_enabled: bool = False
    credentials: Optional[List[Dict[str, str]]] = None
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


AssistantConfig = KnowledgeBaseConfig
AssistantCopyRequest = KnowledgeBaseCopyRequest
