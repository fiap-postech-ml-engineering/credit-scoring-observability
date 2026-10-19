"""Rastreamento de ciclo de vida do modelo e do monitoramento no MLflow (Etapa 3).

Duas famílias de run convivem no mesmo experimento, distinguidas pela tag
`run_type`:

* `training` — um run por treino do baseline, com os parâmetros e as
  métricas de cada candidato, o modelo vencedor com assinatura e o
  dataset de Referência como artefato de linhagem;
* `monitoring` — um run por lote de produção avaliado, com o resultado do
  contrato de dados, o PSI/KS de cada feature, as métricas de desempenho do
  lote e as flags de alerta.

Com isso o MLflow responde, numa única tela, "este modelo degradou?" e
"desde quando?" — comparando runs `monitoring` ao longo dos lotes contra a
baseline registrada no run `training`.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import mlflow
from mlflow.models import infer_signature
import pandas as pd

from src.config.settings import settings

if TYPE_CHECKING:  # pragma: no cover - evita import circular em runtime
    from src.credit_scoring.metrics import ClassificationMetrics
    from src.credit_scoring.train import Candidate

logger = logging.getLogger(__name__)

# Valores que o MLflow não aceita como métrica (NaN/inf) são descartados com
# aviso em vez de derrubar o pipeline de monitoramento.
_MAX_PARAM_LEN = 500

# O MLflow 3 serializa modelos sklearn com skops, que recusa classes de fora do
# sklearn a menos que sejam declaradas confiáveis. O XGBClassifier vive dentro
# do nosso Pipeline, então precisa ser declarado aqui — é a lista exata de
# tipos que nós mesmos colocamos no modelo, não um "confie em tudo".
SKOPS_TRUSTED_TYPES = [
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def _configure() -> None:
    """Aponta o MLflow para o backend SQLite e garante o experimento do projeto."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    settings.mlflow_artifact_root.mkdir(parents=True, exist_ok=True)

    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(settings.mlflow_experiment) is None:
        client.create_experiment(
            settings.mlflow_experiment,
            artifact_location=settings.mlflow_artifact_root.as_uri(),
        )
    mlflow.set_experiment(settings.mlflow_experiment)


def _clean_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number or number in (float("inf"), float("-inf")):  # NaN/inf
            logger.debug("métrica ignorada por não ser finita", extra={"metric": key})
            continue
        clean[key] = number
    return clean


def _clean_params(params: dict[str, Any]) -> dict[str, str]:
    return {k: str(v)[:_MAX_PARAM_LEN] for k, v in params.items()}


@contextmanager
def run(run_name: str, run_type: str, tags: dict[str, str] | None = None) -> Generator[Any]:
    """Abre um run do MLflow já configurado e tagueado."""
    _configure()
    all_tags = {"run_type": run_type, "project": "credit-scoring-observability"}
    all_tags.update(tags or {})
    with mlflow.start_run(run_name=run_name, tags=all_tags) as active:
        yield active


def log_training_run(
    winner: Candidate,
    candidates: list[Candidate],
    reference_metrics: ClassificationMetrics,
    x_sample: pd.DataFrame,
    reference_rows: int,
) -> str | None:
    """Registra o treino do baseline: candidatos, vencedor, modelo e linhagem."""
    try:
        with run(f"baseline-{winner.name}", "training", {"candidate": winner.name}) as active:
            mlflow.log_params(_clean_params(winner.params))
            mlflow.log_params(
                {
                    "candidate": winner.name,
                    "decision_threshold": settings.decision_threshold,
                    "random_state": settings.random_state,
                    "reference_rows": reference_rows,
                    "n_features": len(x_sample.columns),
                }
            )

            # Métricas de validação de cada candidato, para a escolha ficar auditável.
            for candidate in candidates:
                if candidate.metrics is None:
                    continue
                mlflow.log_metrics(
                    _clean_metrics(
                        {
                            f"val_{candidate.name}_{k}": v
                            for k, v in candidate.metrics.to_dict().items()
                        }
                    )
                )

            # Métricas do vencedor na Referência: é esta a baseline contra a
            # qual todo lote de produção será comparado.
            mlflow.log_metrics(
                _clean_metrics(
                    {f"reference_{k}": v for k, v in reference_metrics.to_dict().items()}
                )
            )

            signature = infer_signature(x_sample, winner.estimator.predict_proba(x_sample))
            mlflow.sklearn.log_model(
                winner.estimator,
                name="model",
                signature=signature,
                input_example=x_sample,
                registered_model_name=settings.mlflow_registered_model,
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )

            if settings.reference_path.exists():
                mlflow.log_artifact(str(settings.reference_path), artifact_path="datasets")

            logger.info("run de treino registrado", extra={"mlflow_run_id": active.info.run_id})
            return active.info.run_id
    except Exception as exc:  # noqa: BLE001 - rastreamento não deve quebrar o treino
        logger.warning("falha ao registrar run de treino no MLflow", extra={"error": str(exc)})
        return None


def log_monitoring_run(
    batch_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[Path] | None = None,
    tags: dict[str, str] | None = None,
) -> str | None:
    """Registra a avaliação de um lote de produção (contrato + drift + desempenho)."""
    try:
        with run(
            f"monitor-{batch_name}", "monitoring", {"batch": batch_name, **(tags or {})}
        ) as active:
            mlflow.log_params(_clean_params(params))
            mlflow.log_metrics(_clean_metrics(metrics))
            for artifact in artifacts or []:
                if Path(artifact).exists():
                    mlflow.log_artifact(str(artifact), artifact_path="reports")
            logger.info(
                "run de monitoramento registrado",
                extra={"mlflow_run_id": active.info.run_id, "batch": batch_name},
            )
            return active.info.run_id
    except Exception as exc:  # noqa: BLE001 - monitoramento não deve quebrar o pipeline
        logger.warning(
            "falha ao registrar run de monitoramento no MLflow",
            extra={"error": str(exc), "batch": batch_name},
        )
        return None
