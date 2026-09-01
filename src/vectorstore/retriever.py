from typing import Optional, List
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_core.documents import Document

from .embeddings import get_embeddings
from ..core.config import settings
from ..core.logger import logger


def get_retriever(
    documents: Optional[List[Document]] = None,
    index_name: Optional[str] = None,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None,
):
    """
    Creates and returns a Pinecone retriever.
    - If documents are passed, creates a PineconeHybridSearchRetriever (dense + sparse BM25).
    - Otherwise, returns a Dense MMR PineconeVectorStore retriever.
    """
    idx_name = index_name or settings.PINECONE_INDEX_NAME
    k = top_k or settings.RETRIEVER_K
    f_k = fetch_k or settings.RETRIEVER_FETCH_K
    embeddings = get_embeddings()

    pinecone_db = PineconeVectorStore(
        index_name=idx_name,
        embedding=embeddings,
        pinecone_api_key=settings.PINECONE_API_KEY,
    )

    dense_retriever = pinecone_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": f_k},
    )

    if documents:
        try:
            bm25_encoder = BM25Encoder().default()
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(idx_name)

            ensemble_retriever = PineconeHybridSearchRetriever(
                embeddings=embeddings,
                sparse_encoder=bm25_encoder,
                index=index,
                top_k=20,
                alpha=0.5,
            )
            return ensemble_retriever
        except Exception as e:
            logger.warning(f"Hybrid retriever fallback to dense retriever due to: {e}")
            return dense_retriever

    return dense_retriever
