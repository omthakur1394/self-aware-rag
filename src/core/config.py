import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application configuration and environment variables."""

    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    MONGO_URI: Optional[str] = os.getenv("MONGO_URI")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

    # Model Configurations
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5-nano-2025-08-07")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")

    # Vector Store & Search Configurations
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "bionic-rag-cloud")
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "5"))
    RETRIEVER_FETCH_K: int = int(os.getenv("RETRIEVER_FETCH_K", "20"))
    ARXIV_TOP_K: int = int(os.getenv("ARXIV_TOP_K", "2"))
    MAX_REFLECTION_ATTEMPTS: int = int(os.getenv("MAX_REFLECTION_ATTEMPTS", "2"))

    # Server Configurations
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "7860"))


settings = Settings()
