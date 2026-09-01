from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RAGReflectionState(BaseModel):
    """Represents the internal state of the self-reflective RAG LangGraph workflow."""

    question: str
    search_query: str = Field(default="", description="Optimized query rewritten after failed reflection")
    retrieved_docs: List[Document] = Field(default_factory=list, description="Documents retrieved from local vector store, web, and ArXiv")
    answer: str = Field(default="", description="Generated answer with bracketed citations")
    reflection: str = Field(default="", description="Reflection evaluation result and feedback")
    revised: bool = Field(default=False, description="Flag indicating if the answer required revision")
    attempts: int = Field(default=0, description="Number of generation/reflection attempts made")
