import uvicorn
from src.api.app import app
from src.core.config import settings

# Expose app for ASGI servers (e.g. uvicorn main:app)
__all__ = ["app"]

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
    )