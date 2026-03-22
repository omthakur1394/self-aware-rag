import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever

embeddings = HuggingFaceEmbeddings(model_name="google/gemma-embedding-300m")

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
        sparse_retriever = BM25Retriever.from_documents(documents)
        sparse_retriever.k = 5
        
        return EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5]
        )
    
    return dense_retriever