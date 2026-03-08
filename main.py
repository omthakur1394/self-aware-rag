from fastapi import FastAPI 
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.graph import app_graph
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


@app.post("/chat")
async def chat(request:chat_bot):
    config = {"configurable": {"thread_id": request.thread_id}}
    res = app_graph.invoke({"question": request.chat}, config=config)
    return {
    "res": res["answer"],
    "sources": [doc.metadata.get("source", "unknown") for doc in res["retrieved_docs"]]}
                             

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)




