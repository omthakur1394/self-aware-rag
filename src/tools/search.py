from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun
from ..core.logger import logger

_ddg_tool: Optional[DuckDuckGoSearchRun] = None


def get_search_tool() -> DuckDuckGoSearchRun:
    """Returns a DuckDuckGoSearchRun instance."""
    global _ddg_tool
    if _ddg_tool is None:
        _ddg_tool = DuckDuckGoSearchRun()
    return _ddg_tool


def search_duckduckgo(query: str) -> List[Document]:
    """Executes a DuckDuckGo web search safely, returning Documents."""
    tool = get_search_tool()
    try:
        ddg_result = tool.run(query)
        if ddg_result:
            return [Document(page_content=str(ddg_result), metadata={"source": "DuckDuckGo"})]
        return []
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []
