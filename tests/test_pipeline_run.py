"""Testes do entrypoint ponta a ponta e do rastreamento no MLflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

# --- Demonstração do portão de dados (entregável da Etapa 1) ----------------


@pytest.mark.contracts
def test_demo_do_contrato_grava_laudo_e_manifesto(
    synthetic_reference: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config.settings import settings
    from src.pipeline.run import BAD_BATCH_NAME, demo_contract_block

    referencia = tmp_path / "reference.csv"
    synthetic_reference.to_csv(referencia, index=False)
    monkeypatch.setattr(settings, "reference_path", referencia)

    laudo_path = demo_contract_block(reports_dir=tmp_path)

    laudo = json.loads(laudo_path.read_text(encoding="utf-8"))
    assert laudo["passed"] is False
    assert laudo["n_failures"] > 0
    assert laudo["batch_name"] == BAD_BATCH_NAME

    manifesto = json.loads(
        (tmp_path / f"{BAD_BATCH_NAME}__manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifesto) >= 11
    assert all(item["caught"] for item in manifesto), (
        "toda injeção declarada precisa ser interceptada pelo contrato"
    )


# --- Rastreamento no MLflow -------------------------------------------------


@pytest.fixture
def mlflow_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Aponta o MLflow para um SQLite descartável dentro do tmp_path."""
    from src.config.settings import settings

    monkeypatch.setattr(settings, "mlflow_tracking_uri", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(settings, "mlflow_artifact_root", tmp_path / "artifacts")
    monkeypatch.setattr(settings, "mlflow_experiment", "teste-observabilidade")
    return tmp_path


@pytest.mark.integration
def test_run_de_monitoramento_e_registrado(mlflow_local: Path) -> None:
    import mlflow

    from src.observability.tracking import log_monitoring_run

    run_id = log_monitoring_run(
        batch_name="2024-05_recessao",
        params={"status": "sucesso", "alerts": "degradacao_silenciosa"},
        metrics={"max_psi": 0.002, "delta_roc_auc": -0.0454, "silent_degradation": 1.0},
        tags={"drift_kind": "concept"},
    )

    assert run_id is not None

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_local / 'mlflow.db'}")
    run = mlflow.get_run(run_id)

    assert run.data.metrics["max_psi"] == pytest.approx(0.002)
    assert run.data.metrics["silent_degradation"] == 1.0
    assert run.data.tags["run_type"] == "monitoring"
    assert run.data.tags["drift_kind"] == "concept"


@pytest.mark.integration
def test_metricas_nao_finitas_sao_descartadas_em_vez_de_quebrar(mlflow_local: Path) -> None:
    """Lote de uma classe só produz AUC NaN; isso não pode derrubar o pipeline."""
    import mlflow

    from src.observability.tracking import log_monitoring_run

    run_id = log_monitoring_run(
        batch_name="lote_degenerado",
        params={"status": "sucesso"},
        metrics={"roc_auc": float("nan"), "max_psi": 0.01, "infinito": float("inf")},
    )

    assert run_id is not None
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_local / 'mlflow.db'}")
    metricas = mlflow.get_run(run_id).data.metrics

    assert "roc_auc" not in metricas
    assert "infinito" not in metricas
    assert metricas["max_psi"] == pytest.approx(0.01)


@pytest.mark.integration
def test_falha_de_rastreamento_nao_derruba_o_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observabilidade quebrada degrada a visibilidade, nunca o serviço."""
    from src.config.settings import settings
    from src.observability.tracking import log_monitoring_run

    monkeypatch.setattr(settings, "mlflow_tracking_uri", "postgresql://inexistente:1/none")

    assert log_monitoring_run("lote", {"a": 1}, {"b": 2.0}) is None


@pytest.mark.integration
def test_run_de_treino_registra_modelo_e_candidatos(
    synthetic_reference: pd.DataFrame, mlflow_local: Path, tmp_path: Path
) -> None:
    import mlflow

    from src.credit_scoring.train import train

    train(
        reference=synthetic_reference,
        model_path=tmp_path / "model.joblib",
        log_to_mlflow=True,
    )

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_local / 'mlflow.db'}")
    experimento = mlflow.get_experiment_by_name("teste-observabilidade")
    assert experimento is not None
    runs = mlflow.search_runs(experiment_ids=[experimento.experiment_id])

    assert len(runs) == 1
    linha = runs.iloc[0]
    assert linha["tags.run_type"] == "training"
    # Os dois candidatos precisam estar registrados para a escolha ser auditável.
    assert "metrics.val_xgboost_roc_auc" in runs.columns
    assert "metrics.val_logistic_regression_roc_auc" in runs.columns
    assert "metrics.reference_roc_auc" in runs.columns
