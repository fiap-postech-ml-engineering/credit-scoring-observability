"""Métricas Prometheus do pipeline, do modelo e da API (Etapa 3).

O desafio arquitetural aqui: o monitoramento de drift é um job **batch**, e o
Prometheus só sabe *raspar* alvos que estão de pé. Um processo que roda, emite
métricas e morre nunca é raspado.

A solução adotada evita subir um Pushgateway: a API é o gateway de métricas.
`DriftMetricsCollector` é um collector customizado que, **no momento do
scrape**, lê os laudos JSON que o pipeline deixou em `reports/` e os traduz
em séries Prometheus. Resultado: um único alvo de scrape (`api:8000/metrics`)
expõe ao mesmo tempo

* as métricas ao vivo da API (requisições, latência, distribuição de scores);
* as métricas da última execução do pipeline batch (contrato, PSI, AUC,
  alertas de drift).

Os laudos em disco são a fonte da verdade, então reiniciar a API não perde
histórico de monitoramento — o que aconteceria com métricas em memória.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Iterator

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Registry próprio em vez do global: mantém o /metrics limpo e torna os
# testes determinísticos (sem métricas vazadas entre casos de teste).
REGISTRY = CollectorRegistry()

# --- Métricas ao vivo da API ----------------------------------------------

http_requests_total = Counter(
    "credit_http_requests_total",
    "Total de requisições HTTP atendidas pela API.",
    ["method", "path", "status"],
    registry=REGISTRY,
)

http_request_duration_seconds = Histogram(
    "credit_http_request_duration_seconds",
    "Latência das requisições HTTP, fim a fim.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)

scoring_duration_seconds = Histogram(
    "credit_scoring_duration_seconds",
    "Latência isolada da inferência, sem o overhead de rede.",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    registry=REGISTRY,
)

score_distribution = Histogram(
    "credit_score_distribution",
    "Distribuição das probabilidades de inadimplência devolvidas pela API.",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

decisions_total = Counter(
    "credit_decisions_total",
    "Decisões de crédito emitidas pela API, por resultado.",
    ["decision"],
    registry=REGISTRY,
)

contract_rejections_total = Counter(
    "credit_contract_rejections_total",
    "Requisições rejeitadas pelo contrato de dados na API.",
    ["reason"],
    registry=REGISTRY,
)

model_info = Gauge(
    "credit_model_info",
    "Metadados do modelo carregado (sempre 1; a informação está nos rótulos).",
    ["candidate", "threshold"],
    registry=REGISTRY,
)


# --- Métricas do pipeline batch, lidas dos laudos no scrape ----------------


class DriftMetricsCollector(Collector):
    """Traduz os laudos JSON de `reports/` em séries Prometheus a cada scrape."""

    def __init__(self, reports_dir: Path | None = None) -> None:
        self.reports_dir = reports_dir or settings.reports_dir

    def _load(self, pattern: str) -> Iterator[dict[str, Any]]:
        if not self.reports_dir.exists():
            return
        for path in sorted(self.reports_dir.glob(pattern)):
            try:
                yield json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "laudo ilegível ignorado no scrape",
                    extra={"path": str(path), "error": str(exc)},
                )

    def collect(self) -> Iterator[Metric]:
        yield from self._collect_contract_metrics()
        yield from self._collect_drift_metrics()

    def _collect_contract_metrics(self) -> Iterator[Metric]:
        passed = GaugeMetricFamily(
            "credit_contract_passed",
            "1 se o lote passou no contrato de dados, 0 se foi bloqueado.",
            labels=["batch"],
        )
        violations = GaugeMetricFamily(
            "credit_contract_violations",
            "Número de linhas reprovadas pelo contrato, por lote.",
            labels=["batch"],
        )
        rejection_rate = GaugeMetricFamily(
            "credit_contract_rejection_rate",
            "Fração das linhas do lote com ao menos uma violação de contrato.",
            labels=["batch"],
        )
        by_rule = GaugeMetricFamily(
            "credit_contract_violations_by_rule",
            "Violações de contrato por regra, por lote.",
            labels=["batch", "rule"],
        )

        for laudo in self._load("*__validation.json"):
            batch = str(laudo.get("batch_name", "desconhecido"))
            passed.add_metric([batch], 1.0 if laudo.get("passed") else 0.0)
            violations.add_metric([batch], float(laudo.get("n_failures", 0)))
            rejection_rate.add_metric([batch], float(laudo.get("rejection_rate", 0.0)))
            for rule, count in (laudo.get("failures_by_rule") or {}).items():
                by_rule.add_metric([batch, str(rule)], float(count))

        yield passed
        yield violations
        yield rejection_rate
        yield by_rule

    def _collect_drift_metrics(self) -> Iterator[Metric]:
        psi = GaugeMetricFamily(
            "credit_feature_psi",
            "Population Stability Index por feature, por lote.",
            labels=["batch", "feature"],
        )
        drifted_count = GaugeMetricFamily(
            "credit_drifted_features_count",
            "Quantidade de features em drift no lote.",
            labels=["batch"],
        )
        dataset_drift = GaugeMetricFamily(
            "credit_dataset_drift",
            "1 se a fração de features em drift cruzou o limiar do dataset.",
            labels=["batch"],
        )
        max_psi = GaugeMetricFamily(
            "credit_max_psi",
            "Maior PSI observado entre as features do lote.",
            labels=["batch"],
        )
        performance = GaugeMetricFamily(
            "credit_model_performance",
            "Métricas de desempenho do modelo no lote de produção.",
            labels=["batch", "metric"],
        )
        performance_delta = GaugeMetricFamily(
            "credit_model_performance_delta",
            "Variação da métrica contra a baseline de validação (negativo = pior).",
            labels=["batch", "metric"],
        )
        alerts = GaugeMetricFamily(
            "credit_drift_alert",
            "1 quando o alerta está ativo para o lote.",
            labels=["batch", "alert"],
        )
        silent = GaugeMetricFamily(
            "credit_silent_degradation",
            "1 quando o desempenho caiu sem nenhuma feature acusar drift.",
            labels=["batch"],
        )

        known_alerts = (
            "degradacao_silenciosa",
            "queda_de_desempenho",
            "drift_de_dataset",
            "drift_de_feature",
        )

        for laudo in self._load("*__drift.json"):
            batch = str(laudo.get("batch_name", "desconhecido"))
            summary = laudo.get("summary") or {}

            for feature, value in (laudo.get("psi_by_feature") or {}).items():
                psi.add_metric([batch, str(feature)], float(value))

            drifted_count.add_metric([batch], float(summary.get("n_drifted", 0)))
            dataset_drift.add_metric([batch], 1.0 if summary.get("dataset_drift") else 0.0)
            max_psi.add_metric([batch], float(summary.get("max_psi", 0.0)))

            for metric, value in (laudo.get("performance") or {}).items():
                try:
                    performance.add_metric([batch, str(metric)], float(value))
                except (TypeError, ValueError):
                    continue

            for metric, value in (laudo.get("performance_delta") or {}).items():
                try:
                    performance_delta.add_metric([batch, str(metric)], float(value))
                except (TypeError, ValueError):
                    continue

            active = set(laudo.get("alerts") or [])
            for alert in known_alerts:
                alerts.add_metric([batch, alert], 1.0 if alert in active else 0.0)

            silent.add_metric([batch], 1.0 if laudo.get("silent_degradation") else 0.0)

        yield psi
        yield drifted_count
        yield dataset_drift
        yield max_psi
        yield performance
        yield performance_delta
        yield alerts
        yield silent


class PipelineRunMetricsCollector(Collector):
    """Expõe o resultado das execuções do pipeline lidas de `reports/pipeline_runs.json`."""

    def __init__(self, reports_dir: Path | None = None) -> None:
        self.reports_dir = reports_dir or settings.reports_dir

    def collect(self) -> Iterator[Metric]:
        path = self.reports_dir / "pipeline_runs.json"
        task_status = GaugeMetricFamily(
            "credit_pipeline_task_status",
            "1 se a tarefa do pipeline terminou com sucesso, 0 se falhou.",
            labels=["batch", "task"],
        )
        task_duration = GaugeMetricFamily(
            "credit_pipeline_task_duration_seconds",
            "Duração da tarefa do pipeline, em segundos.",
            labels=["batch", "task"],
        )
        rows_processed = GaugeMetricFamily(
            "credit_pipeline_rows_processed",
            "Linhas processadas pelo pipeline, por lote.",
            labels=["batch"],
        )
        last_run = GaugeMetricFamily(
            "credit_pipeline_last_run_timestamp_seconds",
            "Instante (epoch) da última execução registrada do pipeline.",
            labels=[],
        )

        if not path.exists():
            yield task_status
            yield task_duration
            yield rows_processed
            yield last_run
            return

        try:
            runs = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("histórico de execuções ilegível", extra={"error": str(exc)})
            runs = []

        newest = 0.0
        for run in runs if isinstance(runs, list) else []:
            batch = str(run.get("batch", "desconhecido"))
            rows_processed.add_metric([batch], float(run.get("n_rows", 0)))
            newest = max(newest, float(run.get("finished_at_epoch", 0.0)))
            for task in run.get("tasks", []):
                name = str(task.get("name", "desconhecida"))
                task_status.add_metric(
                    [batch, name], 1.0 if task.get("status") == "sucesso" else 0.0
                )
                task_duration.add_metric([batch, name], float(task.get("duration_seconds", 0.0)))

        last_run.add_metric([], newest)

        yield task_status
        yield task_duration
        yield rows_processed
        yield last_run


def register_batch_collectors(reports_dir: Path | None = None) -> None:
    """Registra os collectors de batch no registry da aplicação (idempotente)."""
    for collector_cls in (DriftMetricsCollector, PipelineRunMetricsCollector):
        collector = collector_cls(reports_dir)
        try:
            REGISTRY.register(collector)
        except ValueError:
            # Já registrado (reload do uvicorn, ou chamada repetida nos testes).
            logger.debug("collector já registrado", extra={"collector": collector_cls.__name__})


class Timer:
    """Cronômetro de bloco para alimentar os histogramas de latência."""

    def __init__(self, histogram: Histogram) -> None:
        self.histogram = histogram
        self.elapsed = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed = time.perf_counter() - self._start
        self.histogram.observe(self.elapsed)
