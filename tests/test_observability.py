"""Testes da camada de observabilidade: logs estruturados, métricas e pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from src.observability.logging_setup import JsonFormatter, read_events, setup_logging
from src.observability.metrics import (
    DriftMetricsCollector,
    PipelineRunMetricsCollector,
)

pytestmark = pytest.mark.integration


# --- Logging estruturado ----------------------------------------------------


def test_formatter_promove_o_extra_para_o_topo_do_json() -> None:
    record = logging.LogRecord(
        name="teste",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="lote processado",
        args=(),
        exc_info=None,
    )
    record.batch = "2024-05_recessao"
    record.max_psi = 0.002

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "lote processado"
    assert payload["batch"] == "2024-05_recessao"
    assert payload["max_psi"] == 0.002
    assert payload["level"] == "INFO"
    assert "timestamp" in payload


def test_formatter_serializa_objetos_nao_json() -> None:
    """Caminhos e datas aparecem em `extra` o tempo todo; não podem quebrar o log."""
    record = logging.LogRecord(
        name="teste",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="caminho",
        args=(),
        exc_info=None,
    )
    record.path = Path("/tmp/x.csv")
    assert "/tmp/x.csv" in json.loads(JsonFormatter().format(record))["path"]


def test_logs_do_pipeline_sao_lidos_de_volta(tmp_path: Path) -> None:
    log_file = tmp_path / "pipeline.jsonl"
    setup_logging(log_file=log_file)

    logging.getLogger("teste").info("tarefa concluída", extra={"task": "ingestao", "rows": 8000})
    logging.shutdown()

    eventos = read_events(log_file)
    assert any(e["message"] == "tarefa concluída" and e["rows"] == 8000 for e in eventos)


def test_read_events_de_arquivo_inexistente_devolve_vazio(tmp_path: Path) -> None:
    assert read_events(tmp_path / "nao_existe.jsonl") == []


def test_read_events_ignora_linha_corrompida(tmp_path: Path) -> None:
    log_file = tmp_path / "pipeline.jsonl"
    log_file.write_text('{"message": "ok"}\nnão é json\n{"message": "ok2"}\n', encoding="utf-8")
    assert len(read_events(log_file)) == 2


# --- Collectors do Prometheus -----------------------------------------------


def test_collector_de_drift_traduz_o_laudo_em_series(tmp_path: Path) -> None:
    (tmp_path / "lote__drift.json").write_text(
        json.dumps(
            {
                "batch_name": "2024-05_recessao",
                "summary": {
                    "n_drifted": 0,
                    "max_psi": 0.002,
                    "dataset_drift": False,
                    "drift_share": 0.0,
                },
                "psi_by_feature": {"age": 0.0008, "MonthlyIncome": 0.0015},
                "performance": {"roc_auc": 0.8233, "approved_bad_rate": 0.0448},
                "performance_delta": {"roc_auc": -0.0454},
                "alerts": ["degradacao_silenciosa", "queda_de_desempenho"],
                "silent_degradation": True,
            }
        ),
        encoding="utf-8",
    )

    familias = {m.name: m for m in DriftMetricsCollector(tmp_path).collect()}

    assert familias["credit_max_psi"].samples[0].value == 0.002
    assert familias["credit_silent_degradation"].samples[0].value == 1.0

    alertas = {s.labels["alert"]: s.value for s in familias["credit_drift_alert"].samples}
    assert alertas["degradacao_silenciosa"] == 1.0
    assert alertas["drift_de_feature"] == 0.0


def test_collector_de_drift_sobrevive_a_laudo_corrompido(tmp_path: Path) -> None:
    """Um JSON truncado não pode derrubar o endpoint de métricas."""
    (tmp_path / "quebrado__drift.json").write_text("{isso não é json", encoding="utf-8")
    assert list(DriftMetricsCollector(tmp_path).collect())


def test_collector_de_drift_sem_diretorio_nao_quebra(tmp_path: Path) -> None:
    assert list(DriftMetricsCollector(tmp_path / "inexistente").collect())


def test_collector_de_contrato_expoe_violacoes_por_regra(tmp_path: Path) -> None:
    (tmp_path / "lote__validation.json").write_text(
        json.dumps(
            {
                "batch_name": "lote_corrompido",
                "passed": False,
                "n_failures": 167,
                "rejection_rate": 0.2547,
                "failures_by_rule": {"MonthlyIncome:not_nullable": 40, "age:ge(18)": 15},
            }
        ),
        encoding="utf-8",
    )

    familias = {m.name: m for m in DriftMetricsCollector(tmp_path).collect()}

    assert familias["credit_contract_passed"].samples[0].value == 0.0
    assert familias["credit_contract_violations"].samples[0].value == 167.0
    por_regra = {
        s.labels["rule"]: s.value for s in familias["credit_contract_violations_by_rule"].samples
    }
    assert por_regra["MonthlyIncome:not_nullable"] == 40.0


def test_collector_de_execucoes_expoe_status_das_tarefas(tmp_path: Path) -> None:
    (tmp_path / "pipeline_runs.json").write_text(
        json.dumps(
            [
                {
                    "batch": "lote_corrompido",
                    "n_rows": 530,
                    "finished_at_epoch": 1_700_000_000.0,
                    "tasks": [
                        {"name": "ingestao", "status": "sucesso", "duration_seconds": 0.01},
                        {"name": "inferencia", "status": "bloqueada", "duration_seconds": 0.0},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    familias = {m.name: m for m in PipelineRunMetricsCollector(tmp_path).collect()}
    status = {s.labels["task"]: s.value for s in familias["credit_pipeline_task_status"].samples}

    assert status["ingestao"] == 1.0
    assert status["inferencia"] == 0.0  # bloqueada conta como não-sucesso
    assert familias["credit_pipeline_rows_processed"].samples[0].value == 530.0


def test_collector_de_execucoes_sem_historico_nao_quebra(tmp_path: Path) -> None:
    assert list(PipelineRunMetricsCollector(tmp_path).collect())


# --- Pipeline de monitoramento ----------------------------------------------


def test_pipeline_bloqueia_as_tarefas_seguintes_quando_o_contrato_falha(
    synthetic_reference: pd.DataFrame, trained_bundle: dict, tmp_path: Path
) -> None:
    """O teste que prova a Etapa 1 ponta a ponta: dado ruim não chega ao modelo."""
    from src.pipeline.monitor import monitor_batch

    lote_ruim = synthetic_reference.copy()
    lote_ruim.loc[lote_ruim.index[:20], "age"] = 10
    caminho = tmp_path / "lote_ruim.csv"
    lote_ruim.to_csv(caminho, index=False)

    run = monitor_batch(
        batch_name="lote_ruim",
        batch_path=caminho,
        reference=synthetic_reference,
        bundle=trained_bundle,
        reports_dir=tmp_path,
        log_to_mlflow=False,
    )

    assert run.status == "bloqueada"
    assert "contrato_violado" in run.alerts

    por_nome = {t.name: t.status for t in run.tasks}
    assert por_nome["validacao"] == "sucesso"  # a tarefa rodou...
    assert run.validation is not None
    assert run.validation.passed is False  # ...e reprovou o lote
    for tarefa in ("inferencia", "deteccao_drift", "publicacao"):
        assert por_nome[tarefa] == "bloqueada"


def test_pipeline_processa_lote_valido_ate_o_fim(
    synthetic_reference: pd.DataFrame, trained_bundle: dict, tmp_path: Path
) -> None:
    from src.pipeline.monitor import monitor_batch

    caminho = tmp_path / "lote_ok.csv"
    synthetic_reference.to_csv(caminho, index=False)

    run = monitor_batch(
        batch_name="lote_ok",
        batch_path=caminho,
        reference=synthetic_reference,
        bundle=trained_bundle,
        reports_dir=tmp_path,
        log_to_mlflow=False,
    )

    assert run.status == "sucesso"
    assert all(t.status == "sucesso" for t in run.tasks)
    assert run.drift is not None
    assert (tmp_path / "lote_ok__drift.json").exists()
    assert (tmp_path / "lote_ok__psi_ks.csv").exists()
    assert (tmp_path / "lote_ok__data_drift.html").exists()


def test_lote_identico_a_referencia_nao_gera_alerta(
    synthetic_reference: pd.DataFrame, trained_bundle: dict, tmp_path: Path
) -> None:
    """Sem falso positivo: é o que o lote de controle prova na demonstração."""
    from src.pipeline.monitor import monitor_batch

    caminho = tmp_path / "controle.csv"
    synthetic_reference.to_csv(caminho, index=False)

    run = monitor_batch(
        batch_name="controle",
        batch_path=caminho,
        reference=synthetic_reference,
        bundle=trained_bundle,
        reports_dir=tmp_path,
        log_to_mlflow=False,
    )

    assert run.drift is not None
    assert run.drift.drifted_features == []
    assert not run.drift.dataset_drift
    assert not run.drift.silent_degradation


def test_historico_de_execucoes_e_persistido(
    synthetic_reference: pd.DataFrame, trained_bundle: dict, tmp_path: Path
) -> None:
    from src.pipeline.monitor import monitor_batch, save_runs

    caminho = tmp_path / "lote.csv"
    synthetic_reference.to_csv(caminho, index=False)
    run = monitor_batch(
        "lote", caminho, synthetic_reference, trained_bundle, tmp_path, log_to_mlflow=False
    )

    destino = save_runs([run], tmp_path / "pipeline_runs.json")
    historico = json.loads(destino.read_text(encoding="utf-8"))

    assert len(historico) == 1
    assert historico[0]["batch"] == "lote"
    assert "tasks" in historico[0]


def test_dashboard_e_renderizado_com_tudo_que_importa(
    synthetic_reference: pd.DataFrame, trained_bundle: dict, tmp_path: Path
) -> None:
    from src.pipeline.dashboard import render_dashboard
    from src.pipeline.monitor import monitor_batch

    caminho = tmp_path / "lote.csv"
    synthetic_reference.to_csv(caminho, index=False)
    run = monitor_batch(
        "lote", caminho, synthetic_reference, trained_bundle, tmp_path, log_to_mlflow=False
    )

    destino = render_dashboard(
        [run],
        output_path=tmp_path / "dashboard.html",
        baseline=trained_bundle["validation_metrics"],
        candidate=trained_bundle["candidate"],
    )
    html = destino.read_text(encoding="utf-8")

    assert "<svg" in html
    assert "Observabilidade de Credit Scoring" in html
    assert "prefers-color-scheme: dark" in html, "o tema escuro é requisito"
    assert 'lang="pt-BR"' in html
