from __future__ import annotations

import time
from typing import Any, Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiter for write operations (POST/PUT/DELETE) to MCP and decision endpoints."""
    
    def __init__(self, app: Any, requests_per_minute: int = 60) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_times: dict[str, list[float]] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_id = request.client.host if request.client else "unknown"
        
        if request.method in {"POST", "PUT", "DELETE"}:
            now = time.time()
            
            if client_id not in self.request_times:
                self.request_times[client_id] = []
            
            one_minute_ago = now - 60.0
            self.request_times[client_id] = [t for t in self.request_times[client_id] if t > one_minute_ago]
            
            if len(self.request_times[client_id]) >= self.requests_per_minute:
                logger.warning("rate_limit_exceeded", client_id=client_id, method=request.method, path=request.url.path)
                return Response(
                    content='{"error": "Rate limit exceeded"}',
                    status_code=429,
                    media_type="application/json",
                )
            
            self.request_times[client_id].append(now)
        
        response = await call_next(request)
        return response


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses with structured logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_id = request.client.host if request.client else "unknown"
        start_time = time.time()
        
        logger.info(
            "request_start",
            method=request.method,
            path=request.url.path,
            client=client_id,
        )
        
        response = await call_next(request)
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "request_complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            client=client_id,
        )
        
        return response
