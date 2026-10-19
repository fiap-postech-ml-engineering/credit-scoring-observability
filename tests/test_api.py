"""Testes dos endpoints da API.

O ponto central: o contrato de dados vale **também** na borda em tempo real.
Uma proposta que o pipeline batch rejeitaria recebe 422 aqui, e o modelo
nunca é chamado.
"""

from __future__ import annotations

from typing import Any, Iterator

from fastapi.testclient import TestClient
import pytest

from src.api import inference
from src.api.api import api

pytestmark = pytest.mark.api


PROPOSTA_VALIDA: dict[str, Any] = {
    "revolving_utilization": 0.35,
    "age": 42,
    "late_30_59": 0,
    "debt_ratio": 0.45,
    "monthly_income": 6200.0,
    "open_credit_lines": 9,
    "late_90": 0,
    "real_estate_loans": 1,
    "late_60_89": 0,
    "dependents": 2,
}


@pytest.fixture
def client(trained_bundle: dict, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Cliente com o modelo sintético injetado — sem depender do artefato real."""
    monkeypatch.setattr(inference, "load_model", lambda *a, **k: trained_bundle)
    with TestClient(api) as test_client:
        yield test_client


@pytest.fixture
def client_sem_modelo(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    def _raise(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("modelo ausente")

    monkeypatch.setattr(inference, "load_model", _raise)
    with TestClient(api) as test_client:
        yield test_client


# --- /health ----------------------------------------------------------------


def test_health_reporta_modelo_carregado(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["baseline_roc_auc"] is not None


def test_health_reporta_degradado_sem_modelo(client_sem_modelo: TestClient) -> None:
    """A API sobe mesmo sem modelo e diz a verdade sobre o próprio estado."""
    body = client_sem_modelo.get("/health").json()
    assert body["status"] == "degradado"
    assert body["model_loaded"] is False


# --- /score -----------------------------------------------------------------


def test_score_decide_uma_proposta_valida(client: TestClient) -> None:
    response = client.post("/score", json=PROPOSTA_VALIDA)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["decision"] in ("aprovado", "recusado")
    assert body["threshold"] == 0.5


def test_score_sem_modelo_responde_503(client_sem_modelo: TestClient) -> None:
    response = client_sem_modelo.post("/score", json=PROPOSTA_VALIDA)
    assert response.status_code == 503
    assert "indisponível" in response.json()["detail"]


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("age", 17),
        ("age", 101),
        ("monthly_income", -1.0),
        ("monthly_income", 2_000_000.0),
        ("debt_ratio", -0.1),
        ("debt_ratio", 999.0),
        ("revolving_utilization", 50708.0),
        ("late_30_59", 98),
        ("late_90", 96),
        ("dependents", 50),
        ("real_estate_loans", 99),
    ],
)
def test_contrato_rejeita_na_borda_antes_do_modelo(
    client: TestClient, campo: str, valor: object
) -> None:
    """Mesmos limites do contrato Pandera, aplicados proposta a proposta."""
    response = client.post("/score", json={**PROPOSTA_VALIDA, campo: valor})
    assert response.status_code == 422
    assert any(campo in str(erro["loc"]) for erro in response.json()["detail"])


def test_campo_faltando_e_rejeitado(client: TestClient) -> None:
    payload = {k: v for k, v in PROPOSTA_VALIDA.items() if k != "monthly_income"}
    assert client.post("/score", json=payload).status_code == 422


def test_renda_nula_e_rejeitada(client: TestClient) -> None:
    assert (
        client.post("/score", json={**PROPOSTA_VALIDA, "monthly_income": None}).status_code == 422
    )


def test_perfil_de_risco_alto_recebe_score_maior(client: TestClient) -> None:
    """Sanidade do modelo servido: mais risco, mais probabilidade de inadimplência."""
    seguro = {**PROPOSTA_VALIDA, "revolving_utilization": 0.05, "late_90": 0, "debt_ratio": 0.1}
    arriscado = {**PROPOSTA_VALIDA, "revolving_utilization": 0.98, "late_90": 2, "debt_ratio": 1.9}

    p_seguro = client.post("/score", json=seguro).json()["probability_of_default"]
    p_arriscado = client.post("/score", json=arriscado).json()["probability_of_default"]

    assert p_arriscado > p_seguro


# --- /metrics e /monitoring/summary -----------------------------------------


def test_metrics_expoe_formato_prometheus(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "credit_http_requests_total" in response.text


def test_metrics_contabiliza_as_decisoes(client: TestClient) -> None:
    client.post("/score", json=PROPOSTA_VALIDA)
    assert "credit_decisions_total" in client.get("/metrics").text


def test_metrics_expoe_as_metricas_do_pipeline_batch(client: TestClient, tmp_path) -> None:
    """A API é o gateway: o scrape precisa enxergar o laudo do job batch."""
    import json

    from src.observability.metrics import REGISTRY, DriftMetricsCollector

    (tmp_path / "lote__drift.json").write_text(
        json.dumps(
            {
                "batch_name": "lote_teste",
                "summary": {"n_drifted": 2, "max_psi": 0.42, "dataset_drift": True},
                "psi_by_feature": {"age": 0.42, "MonthlyIncome": 0.31},
                "performance": {"roc_auc": 0.80},
                "performance_delta": {"roc_auc": -0.06},
                "alerts": ["queda_de_desempenho"],
                "silent_degradation": False,
            }
        ),
        encoding="utf-8",
    )

    collector = DriftMetricsCollector(tmp_path)
    REGISTRY.register(collector)
    try:
        texto = client.get("/metrics").text
        assert 'credit_feature_psi{batch="lote_teste",feature="age"} 0.42' in texto
        assert 'credit_max_psi{batch="lote_teste"} 0.42' in texto
        assert 'credit_drift_alert{alert="queda_de_desempenho",batch="lote_teste"} 1.0' in texto
    finally:
        REGISTRY.unregister(collector)


def test_monitoring_summary_sem_execucoes_responde_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.config.settings import settings

    monkeypatch.setattr(settings, "reports_dir", tmp_path)
    assert client.get("/monitoring/summary").status_code == 404
