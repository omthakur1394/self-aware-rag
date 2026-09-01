from typing import Optional
from langgraph.graph import StateGraph, END
from ..schemas.state import RAGReflectionState
from ..agent.nodes import (
    retrieve_docs,
    generate_answer,
    reflect_on_answer,
    rewrite_query,
    finalize,
)
from ..database.mongo import get_checkpointer
from ..core.config import settings
from ..core.logger import logger


def should_continue_reflection(state: RAGReflectionState) -> str:
    """Decides whether to route to 'done' or 'rewriter' based on reflection status and max attempts."""
    if not state.revised or state.attempts >= settings.MAX_REFLECTION_ATTEMPTS:
        return "done"
    return "rewriter"


def build_graph(use_checkpointer: bool = True):
    """Constructs and compiles the self-reflective RAG state graph."""
    builder = StateGraph(RAGReflectionState)

    # Register Nodes
    builder.add_node("retriever", retrieve_docs)
    builder.add_node("responder", generate_answer)
    builder.add_node("reflector", reflect_on_answer)
    builder.add_node("rewriter", rewrite_query)
    builder.add_node("done", finalize)

    # Register Edges
    builder.set_entry_point("retriever")
    builder.add_edge("retriever", "responder")
    builder.add_edge("responder", "reflector")
    builder.add_conditional_edges("reflector", should_continue_reflection)
    builder.add_edge("rewriter", "retriever")
    builder.add_edge("done", END)

    checkpointer = get_checkpointer() if use_checkpointer else None
    if checkpointer is not None:
        logger.info("Compiling graph with MongoDB checkpointer.")
        return builder.compile(checkpointer=checkpointer)

    logger.info("Compiling graph without checkpointer.")
    return builder.compile()


app_graph = build_graph(use_checkpointer=True)
