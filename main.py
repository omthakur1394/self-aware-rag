from fastapi import FastAPI 
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class chat_bot(BaseModel):
    chat : str
    thread_id: str = "1"

@app.get("/")
async def root():
    return {"message": "Self-Aware RAG API is live and running!"}

@app.post("/chat")
async def chat(request:chat_bot):
    from src.graph import app_graph
    
    config = {"configurable": {"thread_id": request.thread_id}}
    res = app_graph.invoke({"question": request.chat}, config=config)
    return {
        "res": res["answer"],
        "sources": [doc.metadata.get("source", "unknown") for doc in res["retrieved_docs"]]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)