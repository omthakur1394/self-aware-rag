from langgraph.graph import StateGraph, END
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from .agent import (
    RAGReflectionState, retrieve_docs, generate_answer,
    reflect_on_answer, rewrite_query, finalize
)
import os
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

builder = StateGraph(RAGReflectionState)
builder.add_node("retriever", retrieve_docs)
builder.add_node("responder", generate_answer)
builder.add_node("reflector", reflect_on_answer)
builder.add_node("rewriter", rewrite_query)
builder.add_node("done", finalize)

builder.set_entry_point("retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", "reflector")
builder.add_conditional_edges("reflector", lambda s: "done" if not s.revised or s.attempts >= 2 else "rewriter")
builder.add_edge("rewriter", "retriever")
builder.add_edge("done", END)

client = MongoClient(MONGO_URI)
checkpointer = MongoDBSaver(client)
app_graph = builder.compile(checkpointer=checkpointer)