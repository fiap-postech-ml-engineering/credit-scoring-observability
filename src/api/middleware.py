"""Middleware de observabilidade: uma linha de log e três métricas por requisição."""

from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.observability.metrics import http_request_duration_seconds, http_requests_total

logger = logging.getLogger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Registra método, rota, status e latência de cada requisição.

    A rota usada como rótulo é o **template** (`/score`), não o caminho
    concreto — rotular por caminho com parâmetros explodiria a cardinalidade
    das séries no Prometheus.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            duration = time.perf_counter() - started
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            http_requests_total.labels(method=request.method, path=path, status="500").inc()
            http_request_duration_seconds.labels(method=request.method, path=path).observe(
                duration
            )
            logger.exception(
                "requisição falhou",
                extra={
                    "method": request.method,
                    "path": path,
                    "duration_seconds": round(duration, 6),
                },
            )
            raise

        duration = time.perf_counter() - started
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)

        http_requests_total.labels(method=request.method, path=path, status=str(status)).inc()
        http_request_duration_seconds.labels(method=request.method, path=path).observe(duration)

        logger.info(
            "requisição atendida",
            extra={
                "method": request.method,
                "path": path,
                "status": status,
                "duration_seconds": round(duration, 6),
            },
        )
        return response
