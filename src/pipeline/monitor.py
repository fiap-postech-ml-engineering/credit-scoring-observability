"""Pipeline de monitoramento de um lote de produção (amarra as Etapas 1–3).

Cada lote passa por cinco tarefas, nesta ordem, e cada tarefa é cronometrada
e logada individualmente:

| # | Tarefa            | O que faz                                              |
|---|-------------------|--------------------------------------------------------|
| 1 | `ingestao`        | lê o CSV do lote                                       |
| 2 | `validacao`       | aplica o contrato Pandera — **bloqueia** se violado    |
| 3 | `inferencia`      | pontua o lote com o modelo em produção                 |
| 4 | `deteccao_drift`  | PSI/KS próprio + relatórios Evidently                  |
| 5 | `publicacao`      | grava laudos JSON/CSV e registra o run no MLflow       |

O encadeamento é o ponto da Etapa 1: se `validacao` falha, as tarefas
seguintes são marcadas como `bloqueada` e o modelo **nunca vê** o lote. Dado
ruim para no portão, não na inferência.

O `PipelineRun` devolvido é a unidade de observabilidade do projeto: é ele
que vira linha em `reports/pipeline_runs.json`, série no Prometheus e run no
MLflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable

import pandas as pd

from src.config.settings import PREDICTION, PREDICTION_PROBA, TARGET, settings
from src.contracts.schemas import CreditInferenceSchema, CreditReferenceSchema
from src.contracts.validate import ValidationResult, save_result_json, validate_batch
from src.credit_scoring.train import load_model, score
from src.drift.detect import DriftReport, detect_drift, save_psi_csv, save_report_json
from src.observability.tracking import log_monitoring_run

logger = logging.getLogger(__name__)

STATUS_SUCCESS = "sucesso"
STATUS_FAILED = "falha"
STATUS_BLOCKED = "bloqueada"


@dataclass
class TaskResult:
    """Resultado de uma tarefa do pipeline — a unidade de status operacional."""

    name: str
    status: str
    duration_seconds: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineRun:
    """Execução completa do pipeline sobre um lote."""

    batch: str
    n_rows: int = 0
    tasks: list[TaskResult] = field(default_factory=list)
    validation: ValidationResult | None = None
    drift: DriftReport | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str = ""
    finished_at_epoch: float = 0.0
    mlflow_run_id: str | None = None
    artifacts: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(t.status == STATUS_FAILED for t in self.tasks):
            return STATUS_FAILED
        if any(t.status == STATUS_BLOCKED for t in self.tasks):
            return STATUS_BLOCKED
        return STATUS_SUCCESS

    @property
    def alerts(self) -> list[str]:
        alerts = list(self.drift.alerts()) if self.drift else []
        if self.validation and not self.validation.passed:
            alerts.insert(0, "contrato_violado")
        return alerts

    @property
    def total_duration_seconds(self) -> float:
        return round(sum(t.duration_seconds for t in self.tasks), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch,
            "status": self.status,
            "n_rows": self.n_rows,
            "alerts": self.alerts,
            "tasks": [t.to_dict() for t in self.tasks],
            "total_duration_seconds": self.total_duration_seconds,
            "validation": self.validation.to_dict() if self.validation else None,
            "drift": self.drift.to_dict() if self.drift else None,
            "mlflow_run_id": self.mlflow_run_id,
            "artifacts": self.artifacts,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "finished_at_epoch": self.finished_at_epoch,
        }


def _run_task(run: PipelineRun, name: str, fn: Callable[[], Any]) -> Any:
    """Executa uma tarefa cronometrando, logando e registrando seu status.

    Em caso de exceção, registra a falha, loga com stack trace e repropaga —
    quem orquestra decide se para tudo ou segue para o próximo lote.
    """
    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        duration = time.perf_counter() - started
        run.tasks.append(
            TaskResult(
                name=name,
                status=STATUS_FAILED,
                duration_seconds=round(duration, 4),
                detail=str(exc),
            )
        )
        logger.error(
            "tarefa do pipeline falhou",
            extra={"batch": run.batch, "task": name, "duration_seconds": round(duration, 4)},
            exc_info=True,
        )
        raise

    duration = time.perf_counter() - started
    run.tasks.append(
        TaskResult(name=name, status=STATUS_SUCCESS, duration_seconds=round(duration, 4))
    )
    logger.info(
        "tarefa do pipeline concluída",
        extra={"batch": run.batch, "task": name, "duration_seconds": round(duration, 4)},
    )
    return result


def _block_remaining(run: PipelineRun, names: list[str], reason: str) -> None:
    for name in names:
        run.tasks.append(TaskResult(name=name, status=STATUS_BLOCKED, detail=reason))
    logger.warning(
        "tarefas bloqueadas pelo portão de dados",
        extra={"batch": run.batch, "blocked_tasks": names, "reason": reason},
    )


def add_predictions(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Anexa `prediction_proba` e `prediction` ao lote, como o Evidently espera."""
    out = df.copy()
    probabilities = score(out, bundle)
    out[PREDICTION_PROBA] = probabilities
    out[PREDICTION] = (probabilities >= bundle["threshold"]).astype(int)
    return out


def monitor_batch(
    batch_name: str,
    batch_path: Path,
    reference: pd.DataFrame,
    bundle: dict[str, Any] | None = None,
    reports_dir: Path | None = None,
    log_to_mlflow: bool = True,
    scenario_tags: dict[str, str] | None = None,
) -> PipelineRun:
    """Roda o pipeline completo de monitoramento sobre um lote de produção."""
    bundle = bundle or load_model()
    reports_dir = reports_dir or settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    run = PipelineRun(batch=batch_name)
    logger.info("pipeline iniciado", extra={"batch": batch_name, "path": str(batch_path)})

    # --- 1. Ingestão -------------------------------------------------------
    batch = _run_task(run, "ingestao", lambda: pd.read_csv(batch_path))
    run.n_rows = len(batch)

    # --- 2. Validação (portão) ---------------------------------------------
    labeled = TARGET in batch.columns
    schema = CreditReferenceSchema if labeled else CreditInferenceSchema

    validation = _run_task(run, "validacao", lambda: validate_batch(batch, schema, batch_name))
    run.validation = validation

    validation_json = save_result_json(validation, reports_dir / f"{batch_name}__validation.json")
    run.artifacts.append(str(validation_json))

    if not validation.passed:
        _block_remaining(
            run,
            ["inferencia", "deteccao_drift", "publicacao"],
            f"contrato {validation.schema_name} violado em {len(validation.rules_violated)} regras",
        )
        run.finished_at = datetime.now(UTC).isoformat()
        run.finished_at_epoch = time.time()
        if log_to_mlflow:
            run.mlflow_run_id = log_monitoring_run(
                batch_name=batch_name,
                params={"status": run.status, "labeled": labeled, "blocked": True},
                metrics={
                    "contract_passed": 0.0,
                    "contract_violations": float(validation.n_failures),
                    "contract_rejection_rate": validation.rejection_rate,
                    "n_rows": float(run.n_rows),
                },
                artifacts=[validation_json],
                tags={"blocked": "true", **(scenario_tags or {})},
            )
        return run

    # --- 3. Inferência -----------------------------------------------------
    scored = _run_task(run, "inferencia", lambda: add_predictions(batch, bundle))

    # --- 4. Detecção de drift ---------------------------------------------
    reference_scored = add_predictions(reference, bundle)
    baseline = bundle.get("validation_metrics") or {}

    drift = _run_task(
        run,
        "deteccao_drift",
        lambda: detect_drift(
            reference=reference_scored,
            current=scored,
            batch_name=batch_name,
            baseline_performance=baseline,
            reports_dir=reports_dir,
        ),
    )
    run.drift = drift

    # --- 5. Publicação -----------------------------------------------------
    def publish() -> None:
        drift_json = save_report_json(drift, reports_dir / f"{batch_name}__drift.json")
        psi_csv = save_psi_csv(drift, reports_dir / f"{batch_name}__psi_ks.csv")
        run.artifacts.extend([str(drift_json), str(psi_csv)])
        run.artifacts.extend(str(p) for p in drift.html_reports.values())

        if log_to_mlflow:
            metrics: dict[str, Any] = {
                "contract_passed": 1.0,
                "contract_violations": 0.0,
                "n_rows": float(run.n_rows),
                "n_drifted_features": float(drift.summary.get("n_drifted", 0)),
                "max_psi": float(drift.summary.get("max_psi", 0.0)),
                "drift_share": float(drift.summary.get("drift_share", 0.0)),
                "dataset_drift": 1.0 if drift.dataset_drift else 0.0,
                "performance_degraded": 1.0 if drift.performance_degraded else 0.0,
                "silent_degradation": 1.0 if drift.silent_degradation else 0.0,
            }
            metrics.update({f"psi_{k}": v for k, v in (drift.to_dict()["psi_by_feature"]).items()})
            metrics.update({f"prod_{k}": v for k, v in drift.performance.items()})
            metrics.update({f"delta_{k}": v for k, v in drift.performance_delta.items()})

            run.mlflow_run_id = log_monitoring_run(
                batch_name=batch_name,
                params={
                    "status": run.status,
                    "labeled": labeled,
                    "blocked": False,
                    "alerts": ",".join(run.alerts) or "nenhum",
                    "psi_alert_threshold": settings.psi_alert_threshold,
                    "auc_degradation_threshold": settings.auc_degradation_threshold,
                },
                metrics=metrics,
                artifacts=[drift_json, psi_csv, *drift.html_reports.values()],
                tags=scenario_tags,
            )

    _run_task(run, "publicacao", publish)

    run.finished_at = datetime.now(UTC).isoformat()
    run.finished_at_epoch = time.time()
    logger.info(
        "pipeline concluído",
        extra={
            "batch": batch_name,
            "status": run.status,
            "alerts": run.alerts,
            "duration_seconds": run.total_duration_seconds,
        },
    )
    return run


def save_runs(runs: list[PipelineRun], path: Path | None = None) -> Path:
    """Grava o histórico de execuções — lido pelo Prometheus e pelo dashboard."""
    path = path or settings.reports_dir / "pipeline_runs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([r.to_dict() for r in runs], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path
