"""Carregamento do modelo e decisão de crédito para a API.

O modelo é carregado uma única vez, no startup (lifespan), e mantido em
memória — carregar o `.joblib` a cada requisição custaria mais que a própria
inferência. Se o artefato não existir, a API sobe mesmo assim e responde 503
em `/score`: um serviço que recusa a subir é pior de operar que um que
reporta honestamente a própria indisponibilidade.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.api.schemas import CreditApplication
from src.config.settings import FEATURES
from src.credit_scoring.train import load_model
from src.observability.metrics import Timer, model_info, scoring_duration_seconds

logger = logging.getLogger(__name__)

_bundle: dict[str, Any] | None = None


def load() -> dict[str, Any] | None:
    """Carrega o modelo em produção. Devolve `None` se o artefato não existir."""
    global _bundle
    try:
        _bundle = load_model()
    except FileNotFoundError:
        logger.warning("modelo não encontrado; /score responderá 503")
        _bundle = None
        return None

    model_info.labels(
        candidate=str(_bundle["candidate"]), threshold=str(_bundle["threshold"])
    ).set(1)
    logger.info(
        "modelo carregado",
        extra={
            "candidate": _bundle["candidate"],
            "threshold": _bundle["threshold"],
            "baseline_roc_auc": round(_bundle["validation_metrics"]["roc_auc"], 4),
        },
    )
    return _bundle


def bundle() -> dict[str, Any] | None:
    return _bundle


def reset() -> None:
    """Descarrega o modelo — usado pelos testes para simular indisponibilidade."""
    global _bundle
    _bundle = None


def to_frame(application: CreditApplication) -> pd.DataFrame:
    """Converte a proposta no DataFrame de uma linha que o modelo espera."""
    row = application.model_dump(by_alias=True)
    return pd.DataFrame([row])[FEATURES]


def decide(application: CreditApplication) -> tuple[float, str, float, str]:
    """Pontua a proposta e devolve (probabilidade, decisão, limiar, candidato)."""
    current = _bundle
    if current is None:
        raise RuntimeError("modelo não carregado")

    frame = to_frame(application)
    with Timer(scoring_duration_seconds):
        probability = float(current["model"].predict_proba(frame)[:, 1][0])

    threshold = float(current["threshold"])
    decision = "recusado" if probability >= threshold else "aprovado"
    return probability, decision, threshold, str(current["candidate"])
