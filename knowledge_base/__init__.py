from knowledge_base.auth import (
    get_user_by_api_key,
    get_user_by_id,
    login_user,
    register_user,
)
from knowledge_base.chunking import chunk_text
from knowledge_base.defaults import ensure_assistant_defaults
from knowledge_base.logic import (
    add_text_document,
    chat_with_assistant,
    chat_with_assistant_with_history,
    create_assistant_logic,
    delete_assistant_logic,
    delete_document_logic,
    duplicate_assistant_logic,
    fetch_assistant,
    get_dashboard_metrics,
    get_document_logic,
    ingest_document_logic,
    list_assistants_logic,
    list_documents_logic,
    reindex_document_logic,
    update_assistant_logic,
    update_document_text_logic,
)
from knowledge_base.models import (
    AssistantConfig,
    AssistantCopyRequest,
    ChatRequest,
    DocUpdate,
    KnowledgeBaseConfig,
    KnowledgeBaseCopyRequest,
)
from knowledge_base.storage import extract_text_with_extractanything, store_file

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

__all__ = [
    "AssistantConfig",
    "AssistantCopyRequest",
    "ChatRequest",
    "DocUpdate",
    "KnowledgeBaseConfig",
    "KnowledgeBaseCopyRequest",
    "add_text_document",
    "add_text_to_knowledge_base",
    "chat_with_assistant",
    "chat_with_assistant_with_history",
    "chat_with_knowledge_base",
    "chat_with_knowledge_base_with_history",
    "chunk_text",
    "create_assistant_logic",
    "create_knowledge_base_logic",
    "delete_assistant_logic",
    "delete_document_logic",
    "delete_knowledge_base_document",
    "delete_knowledge_base_logic",
    "duplicate_assistant_logic",
    "duplicate_knowledge_base_logic",
    "ensure_assistant_defaults",
    "ensure_knowledge_base_defaults",
    "extract_text_with_extractanything",
    "fetch_assistant",
    "fetch_knowledge_base",
    "get_dashboard_metrics",
    "get_document_logic",
    "get_knowledge_base_document",
    "get_user_by_api_key",
    "get_user_by_id",
    "ingest_document_logic",
    "list_assistants_logic",
    "list_documents_logic",
    "list_knowledge_base_documents",
    "list_knowledge_bases_logic",
    "login_user",
    "reindex_document_logic",
    "reindex_knowledge_base_document",
    "register_user",
    "store_file",
    "update_assistant_logic",
    "update_document_text_logic",
    "update_knowledge_base_document_text",
    "update_knowledge_base_logic",
]
