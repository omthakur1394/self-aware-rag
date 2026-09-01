import os
from typing import Optional
from langchain_openai import ChatOpenAI
from ..core.config import settings
from ..core.logger import logger

_llm_instance: Optional[ChatOpenAI] = None


def get_llm(model: Optional[str] = None, temperature: Optional[float] = None) -> ChatOpenAI:
    """Returns configured ChatOpenAI LLM instance."""
    global _llm_instance
    target_model = model or settings.LLM_MODEL
    target_temp = temperature if temperature is not None else settings.LLM_TEMPERATURE

    if _llm_instance is None:
        logger.info(f"Initializing LLM model: {target_model} (temp={target_temp})")
        if settings.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        _llm_instance = ChatOpenAI(
            model=target_model,
            temperature=target_temp,
        )
    return _llm_instance
