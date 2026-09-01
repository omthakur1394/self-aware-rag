def get_rag_generation_prompt(context: str, question: str) -> str:
    """Constructs prompt for grounded answer generation with citation requirements."""
    return (
        "You are a strict, factual AI research assistant with access to 249 research papers.\n"
        "Answer using ONLY the provided Context in a detailed and comprehensive way.\n"
        "Your answer MUST be at least 3-4 paragraphs long.\n"
        "Cover: main concept, how it works, why it matters, and any key details from the papers.\n"
        "If the context is insufficient, state 'I cannot answer this based on the documents.'\n"
        "Every sentence MUST end with a source index (e.g., [0], [1]).\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )


def get_reflection_prompt(question: str, answer: str) -> str:
    """Constructs prompt for validating answer citations and context faithfulness."""
    return (
        f"Review the Answer for the Question.\n"
        f"1. Does it contain bracketed citations like [0]?\n"
        f"2. Is it based ONLY on the context without outside facts?\n"
        f"Respond ONLY with 'Reflection: YES' or 'Reflection: NO' plus explanation.\n\n"
        f"Question: {question}\nAnswer: {answer}"
    )


def get_rewrite_prompt(question: str, reflection_feedback: str) -> str:
    """Constructs prompt for optimizing search query upon negative reflection."""
    return (
        f"Question: {question}\n"
        f"Failure: {reflection_feedback}\n"
        "Write an optimized search query. Return ONLY the query."
    )
