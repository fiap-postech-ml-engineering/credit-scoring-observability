"""Rotas da API: decisão de crédito, saúde, métricas e resumo de monitoramento."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api import inference
from src.api.schemas import CreditApplication, HealthResponse, ScoreResponse
from src.config.settings import settings
from src.observability.metrics import REGISTRY, decisions_total, score_distribution

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["operacional"])
def health() -> HealthResponse:
    """Liveness + readiness: a API está de pé e o modelo está carregado?"""
    current = inference.bundle()
    if current is None:
        return HealthResponse(status="degradado", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_candidate=str(current["candidate"]),
        baseline_roc_auc=round(float(current["validation_metrics"]["roc_auc"]), 4),
    )


@router.post("/score", response_model=ScoreResponse, tags=["crédito"])
def score(application: CreditApplication) -> ScoreResponse:
    """Decide uma proposta de crédito.

    A validação dos limites acontece antes daqui, no schema Pydantic: uma
    proposta com idade 15 ou renda nula recebe 422 e **nunca** chega ao
    modelo — é o mesmo portão do pipeline batch, aplicado em tempo real.
    """
    if inference.bundle() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="modelo indisponível — rode o treino para gerar o artefato",
        )

    probability, decision, threshold, candidate = inference.decide(application)

    score_distribution.observe(probability)
    decisions_total.labels(decision=decision).inc()

    return ScoreResponse(
        probability_of_default=round(probability, 6),
        decision=decision,
        threshold=threshold,
        model_candidate=candidate,
    )


@router.get("/metrics", tags=["operacional"])
def metrics() -> Response:
    """Métricas no formato Prometheus.

    Expõe tanto as métricas ao vivo da API quanto as da última execução do
    pipeline batch, lidas dos laudos em `reports/` no momento do scrape.
    """
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@router.get("/monitoring/summary", tags=["operacional"])
def monitoring_summary() -> dict[str, object]:
    """Resumo legível da última execução do pipeline de monitoramento.

    Atalho operacional: responde "o modelo degradou?" sem abrir o Grafana nem
    o dashboard HTML.
    """
    path = settings.reports_dir / "pipeline_runs.json"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="nenhuma execução de monitoramento registrada",
        )

    try:
        runs = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("histórico de execuções ilegível", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="histórico de execuções ilegível",
        ) from exc

    return {
        "batches": [
            {
                "batch": run.get("batch"),
                "status": run.get("status"),
                "n_rows": run.get("n_rows"),
                "alerts": run.get("alerts"),
                "max_psi": (run.get("drift") or {}).get("summary", {}).get("max_psi"),
                "roc_auc": (run.get("drift") or {}).get("performance", {}).get("roc_auc"),
                "roc_auc_delta": (run.get("drift") or {})
                .get("performance_delta", {})
                .get("roc_auc"),
                "silent_degradation": (run.get("drift") or {}).get("silent_degradation"),
            }
            for run in runs
        ],
        "total_alerts": sum(len(run.get("alerts") or []) for run in runs),
    }
