from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import ArxivRetriever
from ..core.config import settings
from ..core.logger import logger

_arxiv_retriever: Optional[ArxivRetriever] = None


def get_arxiv_retriever(top_k: Optional[int] = None) -> ArxivRetriever:
    """Returns an ArxivRetriever instance."""
    global _arxiv_retriever
    k = top_k or settings.ARXIV_TOP_K
    if _arxiv_retriever is None:
        _arxiv_retriever = ArxivRetriever(top_k_results=k)
    return _arxiv_retriever


def search_arxiv(query: str, top_k: Optional[int] = None) -> List[Document]:
    """Retrieves ArXiv research papers safely, returning Documents."""
    retriever = get_arxiv_retriever(top_k=top_k)
    try:
        return retriever.invoke(query)
    except Exception as e:
        logger.warning(f"Arxiv retriever failed: {e}")
        return []
