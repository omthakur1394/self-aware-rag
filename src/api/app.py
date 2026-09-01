from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router


def create_app() -> FastAPI:
    """FastAPI Application factory."""
    application = FastAPI(
        title="Self-Aware RAG API",
        description="Self-Reflective RAG Agent with LangGraph, Pinecone, and Multi-Source Search",
        version="1.0.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)
    return application


app = create_app()
