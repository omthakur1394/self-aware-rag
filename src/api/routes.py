from fastapi import APIRouter, HTTPException
from ..schemas.api import ChatRequest, ChatResponse, HealthResponse
from ..schemas.state import RAGReflectionState
from ..graph.workflow import app_graph
from ..core.logger import logger

router = APIRouter()


@router.get("/", response_model=dict)
async def root():
    """Root status check."""
    return {"message": "Self-Aware RAG API is live and running!"}


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(message="Self-Aware RAG API is live and running!", status="healthy")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Processes question through self-reflective RAG LangGraph workflow."""
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        initial_state = {"question": request.chat}
        res = app_graph.invoke(initial_state, config=config)

        sources = []
        if "retrieved_docs" in res and res["retrieved_docs"]:
            sources = [doc.metadata.get("source", "unknown") for doc in res["retrieved_docs"]]

        return ChatResponse(
            res=res.get("answer", "No answer generated."),
            sources=sources,
        )
    except Exception as e:
        logger.error(f"Error executing chat workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
