"""Aplicação FastAPI que serve o modelo de Credit Scoring.

A API tem dois papéis no projeto:

1. **Decisão em tempo real** (`POST /score`) — aplica o mesmo contrato de
   dados do pipeline batch, agora proposta por proposta;
2. **Gateway de métricas** (`GET /metrics`) — único alvo de scrape do
   Prometheus, expondo as métricas ao vivo da API somadas às da última
   execução do pipeline batch (ver `src/observability/metrics.py`).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI

from src.api import inference
from src.api.middleware import ObservabilityMiddleware
from src.api.routes import router
from src.observability.logging_setup import setup_logging
from src.observability.metrics import register_batch_collectors

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Configura logging, registra os collectors e carrega o modelo no startup."""
    setup_logging(json_stdout=True)
    register_batch_collectors()
    inference.load()
    logger.info("API pronta")
    yield
    logger.info("API encerrada")


api = FastAPI(
    title="Credit Scoring — Observabilidade",
    description=(
        "Decisão de crédito com contrato de dados aplicado na borda e métricas "
        "de drift expostas para o Prometheus. FIAP MLE Tech Challenge — Fase 4."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

api.add_middleware(ObservabilityMiddleware)
api.include_router(router)
