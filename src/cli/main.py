import sys
from ..schemas.state import RAGReflectionState
from ..graph.workflow import app_graph
from ..core.logger import logger


def run_cli():
    """Interactive command-line interface for the Self-Aware RAG Agent."""
    print("=" * 60)
    print(" Self-Aware RAG Agent CLI ")
    print(" Type 'exit' or 'quit' to exit.")
    print("=" * 60)

    session_id = "session_cli_1"

    while True:
        try:
            user_query = input("\nAsk a question: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break

            init_state = RAGReflectionState(question=user_query)
            config = {"configurable": {"thread_id": session_id}}

            print("\nThinking & Retrieving...")
            result = app_graph.invoke(init_state, config=config)

            print("\n" + "=" * 30 + " Final Answer " + "=" * 30)
            print(result.get("answer", "No answer."))
            print("\n" + "-" * 30 + " Reflection Log " + "-" * 28)
            print(result.get("reflection", "N/A"))
            print(f"\nTotal Reflection/Retrieval Attempts: {result.get('attempts', 1)}")

            if result.get("retrieved_docs"):
                print(f"Sources used ({len(result['retrieved_docs'])}):")
                for i, doc in enumerate(result["retrieved_docs"]):
                    src = doc.metadata.get("source", f"Document #{i+1}")
                    print(f"  [{i}] {src}")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            print(f"\n[Error] {e}")


if __name__ == "__main__":
    run_cli()
