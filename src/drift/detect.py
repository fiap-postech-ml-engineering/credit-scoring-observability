"""Detecção de drift com Evidently AI e geração dos relatórios HTML (Etapa 2).

Três relatórios são produzidos por lote de produção, cada um respondendo a
uma pergunta diferente:

1. **Data Drift** (`DataDriftPreset`) — as features chegaram diferentes do
   que o modelo viu no treino? Responde comparando a distribuição de cada
   feature contra a Referência, com os testes estatísticos do Evidently.
2. **Qualidade da classificação** (`ClassificationPreset`) — o modelo ainda
   acerta? Só roda em lote rotulado; é o único relatório que enxerga
   **concept drift**.
3. **Resumo do dataset** (`DataSummaryPreset`) — sanidade: contagens,
   nulos, constantes. Pega quebra de pipeline que não é drift.

Além do HTML, `detect_drift` devolve um resumo estruturado (`DriftReport`)
com os números que alimentam o MLflow, o Prometheus e o dashboard. A tabela
própria de PSI/KS de `src/drift/statistics.py` entra nesse mesmo resumo, de
modo que o laudo numérico e o relatório visual nunca divergem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any

from evidently import (
    BinaryClassification,
    DataDefinition,
    Dataset,
    MulticlassClassification,
    Report,
)
from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset
import pandas as pd

from src.config.settings import FEATURES, PREDICTION, PREDICTION_PROBA, TARGET, settings
from src.drift.statistics import DriftSummary, dataset_drift_summary, drift_table

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Resumo estruturado da avaliação de drift de um lote."""

    batch_name: str
    n_rows: int
    labeled: bool
    psi_table: pd.DataFrame
    summary: DriftSummary
    performance: dict[str, float] = field(default_factory=dict)
    performance_delta: dict[str, float] = field(default_factory=dict)
    html_reports: dict[str, Path] = field(default_factory=dict)
    evaluated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def drifted_features(self) -> list[str]:
        return list(self.summary["drifted_features"])

    @property
    def dataset_drift(self) -> bool:
        return bool(self.summary["dataset_drift"])

    @property
    def max_psi(self) -> float:
        return float(self.summary["max_psi"])

    @property
    def performance_degraded(self) -> bool:
        """Queda de ROC AUC além do limiar configurado, contra a baseline."""
        delta = self.performance_delta.get("roc_auc")
        if delta is None:
            return False
        return delta <= -settings.auc_degradation_threshold

    @property
    def silent_degradation(self) -> bool:
        """O caso que o enunciado chama de degradação silenciosa.

        O desempenho caiu, mas nenhuma feature acusou drift — sinal de que a
        causa é concept drift, e não mudança na distribuição de entrada.
        """
        return self.performance_degraded and not self.drifted_features

    def alerts(self) -> list[str]:
        """Alertas disparados por este lote, do mais grave para o menos."""
        alerts: list[str] = []
        if self.silent_degradation:
            alerts.append("degradacao_silenciosa")
        if self.performance_degraded:
            alerts.append("queda_de_desempenho")
        if self.dataset_drift:
            alerts.append("drift_de_dataset")
        if self.max_psi >= settings.psi_alert_threshold:
            alerts.append("drift_de_feature")
        return alerts

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_name": self.batch_name,
            "n_rows": self.n_rows,
            "labeled": self.labeled,
            "summary": self.summary,
            "performance": self.performance,
            "performance_delta": self.performance_delta,
            "alerts": self.alerts(),
            "performance_degraded": self.performance_degraded,
            "silent_degradation": self.silent_degradation,
            "psi_by_feature": {row["feature"]: row["psi"] for _, row in self.psi_table.iterrows()}
            if not self.psi_table.empty
            else {},
            "html_reports": {k: str(v) for k, v in self.html_reports.items()},
            "evaluated_at": self.evaluated_at,
        }


def _data_definition(labeled: bool) -> DataDefinition:
    """Descreve as colunas para o Evidently: features, alvo e predição."""
    numerical = [*FEATURES, PREDICTION_PROBA]
    classification: list[BinaryClassification | MulticlassClassification] | None = (
        [
            BinaryClassification(
                target=TARGET,
                prediction_labels=PREDICTION,
                prediction_probas=PREDICTION_PROBA,
                # O alvo é o inteiro 1, não a string "1": passar str levanta
                # KeyError dentro do Evidently. A anotação da biblioteca diz
                # `str | None`, mas o valor aceito em runtime é o próprio
                # rótulo da coluna.
                pos_label=1,  # ty: ignore[invalid-argument-type]
            )
        ]
        if labeled
        else None
    )
    return DataDefinition(
        numerical_columns=numerical,
        categorical_columns=[TARGET, PREDICTION] if labeled else [PREDICTION],
        classification=classification,
    )


def _as_dataset(df: pd.DataFrame, labeled: bool) -> Dataset:
    return Dataset.from_pandas(df, data_definition=_data_definition(labeled))


def _run_report(
    name: str,
    metrics: list[Any],
    current: Dataset,
    reference: Dataset | None,
    batch_name: str,
    reports_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Roda um Report do Evidently, salva o HTML e devolve o resultado em dict."""
    report = Report(metrics=metrics, include_tests=True)
    snapshot = report.run(current_data=current, reference_data=reference)

    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / f"{batch_name}__{name}.html"
    snapshot.save_html(str(html_path))

    logger.info(
        "relatório Evidently gerado",
        extra={"report": name, "batch": batch_name, "path": str(html_path)},
    )
    return html_path, snapshot.dict()


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    batch_name: str,
    baseline_performance: dict[str, float] | None = None,
    reports_dir: Path | None = None,
) -> DriftReport:
    """Compara `current` com `reference` e produz o laudo + os HTMLs.

    Espera que ambos os DataFrames já tenham as colunas de predição
    (`prediction` e `prediction_proba`), adicionadas por
    `src/pipeline/monitor.py`. A presença de `TARGET` em `current` define se
    o relatório de classificação roda — é o que separa um lote rotulado
    (dá para medir concept drift) de um lote só com features.
    """
    reports_dir = reports_dir or settings.reports_dir
    labeled = TARGET in current.columns

    # --- Laudo estatístico próprio (PSI + KS) ------------------------------
    psi_table = drift_table(reference, current, FEATURES)
    summary = dataset_drift_summary(psi_table)

    report = DriftReport(
        batch_name=batch_name,
        n_rows=len(current),
        labeled=labeled,
        psi_table=psi_table,
        summary=summary,
    )

    # --- Relatórios do Evidently -------------------------------------------
    current_ds = _as_dataset(current, labeled)
    reference_ds = _as_dataset(reference, labeled)

    html_path, _ = _run_report(
        "data_drift",
        [DataDriftPreset(drift_share=settings.dataset_drift_share)],
        current_ds,
        reference_ds,
        batch_name,
        reports_dir,
    )
    report.html_reports["data_drift"] = html_path

    html_path, _ = _run_report(
        "data_summary",
        [DataSummaryPreset()],
        current_ds,
        reference_ds,
        batch_name,
        reports_dir,
    )
    report.html_reports["data_summary"] = html_path

    if labeled:
        html_path, _ = _run_report(
            "classification",
            [ClassificationPreset(probas_threshold=settings.decision_threshold)],
            current_ds,
            reference_ds,
            batch_name,
            reports_dir,
        )
        report.html_reports["classification"] = html_path

        from src.credit_scoring.metrics import compute_metrics

        metrics = compute_metrics(
            current[TARGET].to_numpy(),
            current[PREDICTION_PROBA].to_numpy(),
            settings.decision_threshold,
        )
        report.performance = metrics.to_dict()

        if baseline_performance:
            report.performance_delta = {
                key: round(value - float(baseline_performance.get(key, value)), 6)
                for key, value in report.performance.items()
                if key in baseline_performance
            }

    logger.info(
        "detecção de drift concluída",
        extra={
            "batch": batch_name,
            "n_drifted": summary["n_drifted"],
            "max_psi": round(summary["max_psi"], 4),
            "dataset_drift": summary["dataset_drift"],
            "alerts": report.alerts(),
        },
    )
    return report


def save_report_json(report: DriftReport, path: Path | None = None) -> Path:
    """Persiste o laudo em JSON, consumido pelo dashboard consolidado."""
    path = path or settings.reports_dir / f"{report.batch_name}__drift.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_psi_csv(report: DriftReport, path: Path | None = None) -> Path:
    """Exporta a tabela PSI/KS do lote em CSV — o relatório estatístico pedido."""
    path = path or settings.reports_dir / f"{report.batch_name}__psi_ks.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    report.psi_table.to_csv(path, index=False)
    return path
