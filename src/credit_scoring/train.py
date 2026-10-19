"""Treino do modelo baseline de Credit Scoring (Etapa 1).

O baseline não tenta ser o melhor modelo possível — ele precisa ser bom o
suficiente para que a degradação simulada nas etapas seguintes seja visível e
mensurável. Dois candidatos são comparados no mesmo split:

* **Regressão Logística** com padronização — referência interpretável, o tipo
  de modelo que um regulador aceita sem discussão;
* **XGBoost** — ganho de poder preditivo, com `scale_pos_weight` para o
  desbalanceamento de ~6,6% de inadimplentes.

O vencedor por ROC AUC de validação é retreinado no conjunto completo de
treino e salvo em `models/credit_model.joblib`. Tudo — parâmetros, métricas,
modelo com assinatura e o próprio dataset de Referência — é registrado no
MLflow, que é o ambiente de rastreamento da Etapa 3.

Nada entra no treino sem passar pelo contrato de dados: `enforce` é chamado
sobre o dataset de Referência antes do split.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config.settings import FEATURES, TARGET, settings
from src.contracts.schemas import CreditReferenceSchema
from src.contracts.validate import enforce
from src.credit_scoring.dataset import load_reference
from src.credit_scoring.metrics import ClassificationMetrics, compute_metrics
from src.observability.tracking import log_training_run

logger = logging.getLogger(__name__)

VALIDATION_SIZE = 0.25


@dataclass
class Candidate:
    """Um modelo candidato e seu desempenho na validação."""

    name: str
    estimator: Pipeline
    params: dict[str, Any]
    metrics: ClassificationMetrics | None = None

    @property
    def validated_metrics(self) -> ClassificationMetrics:
        """As métricas de validação, já avaliadas.

        Acessar antes de `train` rodar o candidato é erro de programação, não
        um caso a tratar — daí o erro explícito em vez de um `Optional` que
        contamina todo o resto do fluxo.
        """
        if self.metrics is None:
            raise RuntimeError(f"candidato '{self.name}' ainda não foi avaliado")
        return self.metrics


def _build_candidates(scale_pos_weight: float) -> list[Candidate]:
    logistic_params: dict[str, Any] = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": settings.random_state,
    }
    xgb_params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "scale_pos_weight": round(scale_pos_weight, 4),
        "eval_metric": "auc",
        "random_state": settings.random_state,
        "n_jobs": -1,
    }

    return [
        Candidate(
            name="logistic_regression",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(**logistic_params)),
                ]
            ),
            params={f"logistic__{k}": v for k, v in logistic_params.items()},
        ),
        Candidate(
            name="xgboost",
            estimator=Pipeline([("clf", XGBClassifier(**xgb_params))]),
            params={f"xgb__{k}": v for k, v in xgb_params.items()},
        ),
    ]


def train(
    reference: pd.DataFrame | None = None,
    model_path: Path | None = None,
    log_to_mlflow: bool = True,
) -> tuple[Pipeline, ClassificationMetrics, list[Candidate]]:
    """Treina, escolhe e persiste o modelo baseline.

    Devolve o modelo vencedor já retreinado no treino completo, suas métricas
    de validação e a lista de candidatos avaliados.
    """
    model_path = model_path or settings.model_path
    reference = load_reference() if reference is None else reference

    # Portão de dados: o treino não começa se a Referência violar o contrato.
    reference = enforce(reference, CreditReferenceSchema, "dataset_referencia")

    x = reference[FEATURES]
    y = reference[TARGET].to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=VALIDATION_SIZE,
        stratify=y,
        random_state=settings.random_state,
    )

    negatives, positives = int((y_train == 0).sum()), int((y_train == 1).sum())
    candidates = _build_candidates(scale_pos_weight=negatives / max(positives, 1))

    for candidate in candidates:
        candidate.estimator.fit(x_train, y_train)
        scores = candidate.estimator.predict_proba(x_val)[:, 1]
        candidate.metrics = compute_metrics(y_val, scores, settings.decision_threshold)
        logger.info(
            "candidato avaliado",
            extra={
                "candidate": candidate.name,
                "roc_auc": round(candidate.metrics.roc_auc, 4),
                "ks": round(candidate.metrics.ks_statistic, 4),
                "pr_auc": round(candidate.metrics.pr_auc, 4),
            },
        )

    winner = max(candidates, key=lambda c: c.validated_metrics.roc_auc)
    logger.info(
        "candidato vencedor",
        extra={"candidate": winner.name, "roc_auc": round(winner.validated_metrics.roc_auc, 4)},
    )

    # Retreino no conjunto completo de treino+validação: o split já cumpriu seu
    # papel de escolher o candidato, e o modelo em produção merece todo o dado.
    winner.estimator.fit(x, y)
    final_scores = winner.estimator.predict_proba(x)[:, 1]
    reference_metrics = compute_metrics(y, final_scores, settings.decision_threshold)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": winner.estimator,
            "features": FEATURES,
            "threshold": settings.decision_threshold,
            "candidate": winner.name,
            "reference_metrics": reference_metrics.to_dict(),
            "validation_metrics": winner.validated_metrics.to_dict(),
        },
        model_path,
    )
    logger.info("modelo salvo", extra={"path": str(model_path), "candidate": winner.name})

    if log_to_mlflow:
        log_training_run(
            winner=winner,
            candidates=candidates,
            reference_metrics=reference_metrics,
            x_sample=x.head(5),
            reference_rows=len(reference),
        )

    return winner.estimator, winner.validated_metrics, candidates


def load_model(model_path: Path | None = None) -> dict[str, Any]:
    """Carrega o artefato salvo pelo treino (modelo + metadados)."""
    model_path = model_path or settings.model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"modelo não encontrado em {model_path} — rode `make train` primeiro"
        )
    return joblib.load(model_path)


def score(df: pd.DataFrame, bundle: dict[str, Any] | None = None) -> np.ndarray:
    """Devolve a probabilidade de inadimplência para cada linha de `df`."""
    bundle = bundle or load_model()
    return bundle["model"].predict_proba(df[bundle["features"]])[:, 1]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _, metrics, _ = train()
    logger.info("validação: %s", metrics.to_dict())
