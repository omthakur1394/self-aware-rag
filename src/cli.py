from .agent import RAGReflectionState
from .graph import app_graph

if __name__ == "__main__":
    while True:
        user_query = input("ask the questions")
        init_state = RAGReflectionState(question=user_query)
        config = {"configurable": {"thread_id": "session_1"}}
        
        result = app_graph.invoke(init_state, config=config)

        print("\n=== Final Answer ===\n", result["answer"])
        print("\n=== Reflection Log ===\n", result["reflection"])
        print("Total Attempts:", result["attempts"])