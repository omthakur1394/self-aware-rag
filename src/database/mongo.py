from typing import Optional
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from ..core.config import settings
from ..core.logger import logger

_mongo_client: Optional[MongoClient] = None
_checkpointer: Optional[MongoDBSaver] = None


def get_mongo_client() -> Optional[MongoClient]:
    """Returns a singleton MongoDB client if MONGO_URI is configured."""
    global _mongo_client
    if _mongo_client is None and settings.MONGO_URI:
        try:
            _mongo_client = MongoClient(settings.MONGO_URI)
            logger.info("MongoDB client connected successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            _mongo_client = None
    return _mongo_client


def get_checkpointer() -> Optional[MongoDBSaver]:
    """Returns a MongoDBSaver checkpointer using the active MongoDB client."""
    global _checkpointer
    if _checkpointer is None:
        client = get_mongo_client()
        if client is not None:
            try:
                _checkpointer = MongoDBSaver(client)
                logger.info("MongoDB checkpointer initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize MongoDBSaver: {e}")
                _checkpointer = None
    return _checkpointer
