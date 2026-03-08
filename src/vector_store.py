import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model="google/embeddinggemma-300m")

def get_retriever():
    return PineconeVectorStore(
        index_name="bionic-rag-cloud",
        embedding=embeddings,
        pinecone_api_key=os.environ.get("PINECONE_API_KEY")
    ).as_retriever()

