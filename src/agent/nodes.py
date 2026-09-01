from ..schemas.state import RAGReflectionState
from ..vectorstore.retriever import get_retriever
from ..tools.search import search_duckduckgo
from ..tools.arxiv import search_arxiv
from .llm import get_llm
from .prompts import get_rag_generation_prompt, get_reflection_prompt, get_rewrite_prompt
from ..core.logger import logger


def retrieve_docs(state: RAGReflectionState) -> RAGReflectionState:
    """Retrieves context documents from local Pinecone index, DuckDuckGo, and ArXiv."""
    query = state.search_query if state.search_query else state.question
    clean_query = query.replace("\x00", "").strip()
    logger.info(f"Retrieving documents for query: '{clean_query}'")

    # Local Pinecone retriever
    try:
        local_retriever = get_retriever()
        local_docs = local_retriever.invoke(clean_query)
    except Exception as e:
        logger.error(f"Error invoking local vector store retriever: {e}")
        local_docs = []

    # DuckDuckGo search tool
    web_docs = search_duckduckgo(clean_query)

    # ArXiv search tool
    arxiv_docs = search_arxiv(clean_query)

    all_docs = local_docs + web_docs + arxiv_docs
    logger.info(f"Retrieved {len(all_docs)} total documents ({len(local_docs)} local, {len(web_docs)} web, {len(arxiv_docs)} arxiv).")

    return state.model_copy(update={"retrieved_docs": all_docs})


def generate_answer(state: RAGReflectionState) -> RAGReflectionState:
    """Generates an answer citing retrieved documents."""
    context_parts = []
    for i, doc in enumerate(state.retrieved_docs):
        source_name = doc.metadata.get("source", f"Source {i}")
        context_parts.append(f"[{i}] (Source: {source_name}): {doc.page_content}")
    context = "\n\n".join(context_parts)

    prompt = get_rag_generation_prompt(context=context, question=state.question)
    llm = get_llm()
    answer = llm.invoke(prompt).content.strip()

    return state.model_copy(update={"answer": answer, "attempts": state.attempts + 1})


def reflect_on_answer(state: RAGReflectionState) -> RAGReflectionState:
    """Critiques the generated answer for grounding and bracketed citations."""
    prompt = get_reflection_prompt(question=state.question, answer=state.answer)
    llm = get_llm()
    result = llm.invoke(prompt).content
    is_ok = "reflection: yes" in result.lower()

    logger.info(f"Reflection critique passed: {is_ok}")
    return state.model_copy(update={"reflection": result, "revised": not is_ok})


def rewrite_query(state: RAGReflectionState) -> RAGReflectionState:
    """Rewrites query when reflection flags answer as insufficient or ungrounded."""
    prompt = get_rewrite_prompt(question=state.question, reflection_feedback=state.reflection)
    llm = get_llm()
    new_query = llm.invoke(prompt).content.strip()
    logger.info(f"Rewritten query for attempt {state.attempts + 1}: '{new_query}'")

    return state.model_copy(update={"search_query": new_query})


def finalize(state: RAGReflectionState) -> RAGReflectionState:
    """Terminal node in the graph returning final state."""
    logger.info(f"Workflow finished after {state.attempts} attempts.")
    return state
