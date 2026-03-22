import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder

embeddings = HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")

def get_retriever(documents=None):
    pinecone_db = PineconeVectorStore(
        index_name="bionic-rag-cloud",
        embedding=embeddings,
        pinecone_api_key=os.environ.get("PINECONE_API_KEY")
    )
    
    dense_retriever = pinecone_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    if documents:
        bm25_encoder = BM25Encoder().default() 
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index("bionic-rag-cloud")
        
        ensemble_retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=bm25_encoder,
            index=index,
            top_k=20,
            alpha=0.5 
        )
        
        return ensemble_retriever
    
    return dense_retriever