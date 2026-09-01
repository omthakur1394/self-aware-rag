from typing import List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload for the chat API endpoint."""

    chat: str = Field(..., description="User query or prompt to answer")
    thread_id: str = Field(default="1", description="Session or thread ID for checkpointer memory")


class ChatResponse(BaseModel):
    """Response returned by the chat API endpoint."""

    res: str = Field(..., description="Final generated answer")
    sources: List[str] = Field(default_factory=list, description="List of source document references or URLs")


class HealthResponse(BaseModel):
    """Health check status response."""

    message: str = "Self-Aware RAG API is live and running!"
    status: str = "healthy"
