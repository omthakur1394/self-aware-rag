from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_community.retrievers import WikipediaRetriever, ArxivRetriever
from .cofig import llm
from .vector_store import get_retriever

retriever = get_retriever()
wiki_retriever = WikipediaRetriever(top_k_results=2)
arxiv_retriever = ArxivRetriever(top_k_results=2)

class RAGReflectionState(BaseModel):
    question: str
    search_query: str = "" 
    retrieved_docs: List[Document] = []
    answer: str = ""
    reflection: str = ""
    revised: bool = False
    attempts: int = 0

def retrieve_docs(state: RAGReflectionState) -> RAGReflectionState:
    query = state.search_query if state.search_query else state.question
    clean_query = query.replace("\x00", "").strip()
    
    local_docs = retriever.invoke(clean_query)
    wiki_docs = wiki_retriever.invoke(clean_query)
    
    try:
        arxiv_docs = arxiv_retriever.invoke(clean_query)
    except Exception:
        arxiv_docs = []
        
    return state.model_copy(update={"retrieved_docs": local_docs + wiki_docs + arxiv_docs})

def generate_answer(state: RAGReflectionState) -> RAGReflectionState:
    context_parts = []
    for i, doc in enumerate(state.retrieved_docs):
        source_name = doc.metadata.get('source', f'Source {i}')
        context_parts.append(f"[{i}] (Source: {source_name}): {doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    prompt = (
        f"Use the following context to answer the question. "
        f"You MUST cite the source index (e.g., [0], [1]) for every fact you state.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{state.question}"
    )
    
    answer = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"answer": answer, "attempts": state.attempts + 1})

def reflect_on_answer(state: RAGReflectionState) -> RAGReflectionState:
    prompt = f"Reflect on the following answer. State YES if complete and accurate with citations, or NO with explanation.\nQuestion: {state.question}\nAnswer: {state.answer}\nRespond like:\nReflection: YES or NO\nExplanation: ..."
    result = llm.invoke(prompt).content
    is_ok = "reflection: yes" in result.lower()
    return state.model_copy(update={"reflection": result, "revised": not is_ok})

def rewrite_query(state: RAGReflectionState) -> RAGReflectionState:
    prompt = f"Original question: {state.question}\nFailed because: {state.reflection}\nWrite a single optimized search query. Return ONLY the query."
    new_query = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"search_query": new_query})

def finalize(state: RAGReflectionState) -> RAGReflectionState:
    return state