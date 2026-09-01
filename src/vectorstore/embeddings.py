from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from ..core.config import settings
from ..core.logger import logger

_embeddings_instance: Optional[HuggingFaceEmbeddings] = None


def get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """Returns singleton HuggingFace embeddings model instance."""
    global _embeddings_instance
    target_model = model_name or settings.EMBEDDING_MODEL
    if _embeddings_instance is None:
        logger.info(f"Loading HuggingFace embeddings: {target_model}")
        _embeddings_instance = HuggingFaceEmbeddings(model_name=target_model)
    return _embeddings_instance
