import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.middleware import RequestMiddleware
from app.api.routes import router
from app.api.websocket import router as ws_router
from app.inference.pipeline import load_model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    setup_logging(debug=settings.debug)
    logger.info("Starting %s", settings.app_name)

    # Preload ML model at startup for fast first-request latency
    try:
        load_model()
        logger.info("Model preloaded successfully")
    except Exception as e:
        logger.warning("Model preload failed (will load on first request): %s", e)

    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description="Low-latency voice attribute inference for gender and age bracket prediction.",
    version="0.1.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(RequestMiddleware)

# Register routes
app.include_router(router)
app.include_router(ws_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
