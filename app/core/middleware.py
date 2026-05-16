import time
import uuid
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)


class RequestMiddleware(BaseHTTPMiddleware):
    """Middleware that adds request IDs and timing to every request."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        logger.info(
            "req=%s method=%s path=%s status=%d ms=%d",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Ms"] = str(elapsed_ms)
        return response
