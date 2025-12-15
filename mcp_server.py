import os
import sys
from fastmcp import FastMCP, Context

from knowledge_base_logic import (
    add_text_to_knowledge_base,
    chat_with_knowledge_base_with_history,
    get_user_by_api_key,
    list_knowledge_bases_logic,
)

# ------------------------------------------------------------------------------
#  MCP SERVER INSTANCE
# ------------------------------------------------------------------------------
server = FastMCP(
    name="rag-mcp-server",
    instructions="Expose RAG knowledge bases for chat and document ingestion over MCP.",
)


# ------------------------------------------------------------------------------
#  HELPER: Validate API Key from tool argument
# ------------------------------------------------------------------------------
async def validate_api_key(api_key: str):
    """
    The ONLY reliable, cross-client method:
    The client MUST send api_key as a tool argument.

    Example tool call:
    list_knowledge_bases { "api_key": "<user_key>" }
    """
    if not api_key:
        return None, {"error": "Missing api_key. Pass: { \"api_key\": \"YOUR_KEY\" }"}

    user = await get_user_by_api_key(api_key)
    if not user:
        return None, {"error": f"Invalid api_key: {api_key[:6]}..."}

    return user, None


# ------------------------------------------------------------------------------
#  TOOL: LIST KNOWLEDGE BASES (preferred)
# ------------------------------------------------------------------------------
@server.tool
async def list_knowledge_bases(api_key: str = "", ctx: Context = None):
    """
    List all RAG knowledge bases owned by the authenticated user.
    
    This tool retrieves all knowledge bases that belong to the user identified by the provided API key.
    Each knowledge base contains configuration for RAG (Retrieval-Augmented Generation) including
    system prompts, chunking parameters, and vector search settings.
    
    Parameters:
    -----------
    api_key : str, required
        Your user API key for authentication. Get this from the dashboard header
        after logging in at http://localhost:8000/
        
    ctx : Context, optional
        MCP context (automatically provided by the framework)
    
    Returns:
    --------
    list[dict] : List of knowledge base objects, each containing:
        - id: Unique knowledge base identifier
        - name: Knowledge base display name
        - system_prompt: System instructions for the AI
        - chunk_size: Document chunking size
        - overlap: Chunk overlap size
        - top_k: Number of relevant chunks to retrieve
        - secure_enabled: Whether authentication is required
        - owner_id: User ID of the owner
        - created_at: Creation timestamp
        
    dict : Error object if authentication fails:
        - error: Error message
        
    Examples:
    ---------
    Success:
    ```json
    [
        {
            "id": "abc123",
            "name": "Customer Support Knowledge Base",
            "system_prompt": "You are a helpful customer support agent",
            "chunk_size": 1000,
            "overlap": 200,
            "top_k": 5,
            "secure_enabled": true,
            "owner_id": "user_xyz"
        }
    ]
    ```
    
    Error:
    ```json
    {
        "error": "Missing api_key. Pass: { \"api_key\": \"YOUR_KEY\" }"
    }
    ```
    """
    user, error = await validate_api_key(api_key)
    if error:
        return error

    return await list_knowledge_bases_logic(owner_id=user["id"])



# ------------------------------------------------------------------------------
#  TOOL: CHAT (knowledge base)
# ------------------------------------------------------------------------------
@server.tool
async def chat_with_knowledge_base(
    knowledge_base_id: str,
    query: str,
    history: list | None = None,
    api_key: str = "",
    ctx: Context = None
):
    """
    Chat with a RAG knowledge base using natural language queries.
    
    This tool sends a query to a RAG knowledge base which will search through its documents
    and generate a contextual response using retrieved information.
    The model uses vector similarity search to find relevant document chunks and then
    generates an AI response based on those chunks and the system prompt.
    
    Parameters:
    -----------
    knowledge_base_id : str, required
        The unique identifier of the knowledge base to chat with.
        Get this from list_knowledge_bases or the dashboard URL.
        
    query : str, required
        Your question or message to the knowledge base.
        The knowledge base will search its documents and respond accordingly.
        
    history : list[dict], optional
        Conversation history for multi-turn conversations. Each message should have:
        - role: "user" or "assistant"
        - content: The message text
        Example: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
        
    api_key : str, optional
        API key for authentication. Can be either:
        - User-level API key (owner access) - Can chat with any knowledge base you own
        - Knowledge-base-level API key - Can chat with that specific knowledge base only
        For public knowledge bases (secure_enabled=false), this can be omitted.
        Note: Email/password authentication is NOT used in MCP context.
        
    ctx : Context, optional
        MCP context (automatically provided by the framework)
    
    Returns:
    --------
    dict : Response object containing:
        - answer: The AI-generated response text
        - sources: List of document chunks used (if available)
        
    dict : Error object if the request fails:
        - detail: Error message
        
    Examples:
    ---------
    Simple query:
    ```json
    {
        "knowledge_base_id": "abc123",
        "query": "What are your business hours?"
    }
    ```
    
    With conversation history:
    ```json
    {
        "knowledge_base_id": "abc123",
        "query": "What about weekends?",
        "history": [
            {"role": "user", "content": "What are your business hours?"},
            {"role": "assistant", "content": "We're open 9am-5pm Monday-Friday."}
        ]
    }
    ```
    
    Success response:
    ```json
    {
        "answer": "Our business hours are Monday-Friday, 9:00 AM to 5:00 PM EST."
    }
    ```
    
    Error response:
    ```json
    {
        "detail": "Authentication required for this knowledge base"
    }
    ```
    """
    # Call the backend with all optional parameters
    return await chat_with_knowledge_base_with_history(
        assistant_id=knowledge_base_id,
        query=query,
        history=history,
        email=None,  # Not used in MCP context
        password=None,  # Not used in MCP context
        api_key=api_key if api_key else None  # Use provided API key or None
    )


# ------------------------------------------------------------------------------
#  TOOL: ADD DOCUMENT (to knowledge base)
# ------------------------------------------------------------------------------
@server.tool
async def add_document(
    knowledge_base_id: str,
    text: str,
    filename: str = "document.txt",
    api_key: str = "",
    ctx: Context = None
):
    """
    Upload text content to a knowledge base (vector store).
    
    This tool ingests plain text content into a knowledge base's vector database. The text will be:
    1. Split into chunks based on the knowledge base's chunk_size and overlap settings
    2. Converted into vector embeddings
    3. Stored in the vector database (Qdrant)
    4. Made available for retrieval during chat queries
    
    Use this tool to add knowledge, documentation, FAQs, or any text-based information
    that the model should be able to reference when answering questions.
    
    Parameters:
    -----------
    knowledge_base_id : str, required
        The unique identifier of the knowledge base to add the document to.
        Get this from list_knowledge_bases or the dashboard.
        
    text : str, required
        The plain text content to upload. This can be:
        - Documentation
        - FAQ content
        - Product information
        - Policy documents
        - Any text-based knowledge
        
    filename : str, optional (default: "document.txt")
        A descriptive name for this document. Used for tracking and identification.
        Examples: "product_manual.txt", "faq.txt", "pricing_policy.txt"
        
    api_key : str, required
        Your user API key for authentication. Required to verify ownership
        of the knowledge base before allowing document uploads.
        
    ctx : Context, optional
        MCP context (automatically provided by the framework)
    
    Returns:
    --------
    dict : Success response containing:
        - doc_id: Unique identifier for the uploaded document
        
    dict : Error object if authentication fails or upload fails:
        - error: Error message
        
    Examples:
    ---------
    Upload FAQ content:
    ```json
    {
        "knowledge_base_id": "abc123",
        "text": "Q: What are your business hours?\\nA: We're open Monday-Friday, 9am-5pm.\\n\\nQ: Do you offer refunds?\\nA: Yes, within 30 days of purchase.",
        "filename": "customer_faq.txt",
        "api_key": "your-api-key-here"
    }
    ```
    
    Upload product documentation:
    ```json
    {
        "knowledge_base_id": "abc123",
        "text": "Product XYZ is a cloud-based solution that helps teams collaborate...\\n\\nFeatures:\\n- Real-time sync\\n- End-to-end encryption\\n- Mobile apps",
        "filename": "product_xyz_docs.txt",
        "api_key": "your-api-key-here"
    }
    ```
    
    Success response:
    ```json
    {
        "doc_id": "doc_xyz789"
    }
    ```
    
    Error response:
    ```json
    {
        "error": "Missing api_key. Pass: { \\"api_key\\": \\"YOUR_KEY\\" }"
    }
    ```
    
    Notes:
    ------
    - The text will be automatically chunked based on the knowledge base's settings
    - Larger documents may take longer to process
    - Once uploaded, the document is immediately available for retrieval
    - You can upload multiple documents to the same knowledge base
    - Use descriptive filenames to help track your documents
    """
    user, error = await validate_api_key(api_key)
    if error:
        return error

    doc_id = await add_text_to_knowledge_base(
        knowledge_base_id,
        text,
        filename,
        owner_id=user["id"]
    )

    return {"doc_id": doc_id}


# ------------------------------------------------------------------------------
#  RESOURCE VIEW (Optional)
# ------------------------------------------------------------------------------
@server.resource("knowledge-bases://list")
async def knowledge_bases_resource():
    return await list_knowledge_bases_logic()

# ------------------------------------------------------------------------------
#  RUN SERVER
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # If you want FastAPI + MCP on a single port, run `python serve.py` (or `uvicorn main:app`)
    # which mounts this MCP server under `/mcp`.
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "9000"))

    server.run(
        transport="sse",
        host=host,
        port=port,
        log_level="info"
    )
