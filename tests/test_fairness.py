"""Testes da auditoria de equidade por faixa etária.

As métricas de fairness são fáceis de calcular errado em silêncio — trocar o
denominador de "aprovação entre bons pagadores" por "aprovação geral" produz
um número plausível e uma conclusão falsa. Por isso os testes de métrica usam
casos montados à mão, com a resposta conhecida.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.settings import TARGET
from src.governance.fairness import (
    AGE_LABELS,
    age_band,
    audit,
    group_metrics,
    proxy_test,
    retrain_without,
    run_fairness_audit,
    save_fairness,
)

pytestmark = pytest.mark.model


def test_faixas_etarias_cobrem_os_limites_do_contrato() -> None:
    idades = pd.Series([18, 29, 30, 44, 45, 59, 60, 100])
    assert age_band(idades).tolist() == [
        "18-29",
        "18-29",
        "30-44",
        "30-44",
        "45-59",
        "45-59",
        "60+",
        "60+",
    ]


def test_metricas_por_grupo_usam_o_denominador_certo() -> None:
    """Jovens: 4 clientes, 2 bons (1 aprovado) e 2 maus (1 aprovado)."""
    y = np.array([0, 0, 1, 1, 0, 0])
    aprovado = np.array([True, False, True, False, True, True])
    grupos = pd.Series(["18-29"] * 4 + ["60+"] * 2)

    jovens, idosos = group_metrics(y, aprovado, grupos)

    assert jovens.group == "18-29"
    assert jovens.approval_rate == 0.5
    assert jovens.good_payer_approval_rate == 0.5
    assert jovens.bad_payer_approval_rate == 0.5
    assert jovens.approved_bad_rate == 0.5  # 1 mau entre 2 aprovados
    assert idosos.approval_rate == 1.0
    assert idosos.share == pytest.approx(2 / 6, abs=1e-4)


def test_faixa_ausente_no_lote_e_omitida() -> None:
    grupos = pd.Series(["45-59", "45-59"])
    resultado = group_metrics(np.array([0, 1]), np.array([True, False]), grupos)
    assert [g.group for g in resultado] == ["45-59"]


def _lote(idades: list[int], alvo: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"age": idades, TARGET: alvo})


def test_modelo_que_recusa_jovens_bons_pagadores_dispara_os_dois_alertas() -> None:
    lote = _lote([20, 21, 22, 23, 65, 66, 67, 68], [0, 0, 0, 1, 0, 0, 0, 1])
    # Score alto (= recusa) para todos os jovens, baixo para os idosos —
    # exceto o mau pagador idoso, para o AUC ser calculável e não trivial.
    scores = np.array([0.9, 0.9, 0.9, 0.95, 0.1, 0.1, 0.1, 0.6])

    laudo = audit(lote, scores, 0.5, "lote", "teste")

    assert laudo.disparate_impact_ratio == 0.0
    assert laudo.equal_opportunity_gap == 1.0
    assert set(laudo.alerts) == {"impacto_desigual", "oportunidade_desigual"}


def test_modelo_neutro_nao_dispara_alerta() -> None:
    lote = _lote([20, 21, 65, 66], [0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.1, 0.9])

    laudo = audit(lote, scores, 0.5, "lote", "teste")

    assert laudo.disparate_impact_ratio == 1.0
    assert laudo.equal_opportunity_gap == 0.0
    assert laudo.alerts == []


def test_retreino_sem_idade_nao_usa_a_coluna(
    synthetic_reference: pd.DataFrame, trained_bundle: dict
) -> None:
    modelo, features = retrain_without(trained_bundle, synthetic_reference, "age")

    assert "age" not in features
    assert len(features) == len(trained_bundle["features"]) - 1
    # O modelo sem idade pontua um lote que nem tem a coluna `age`.
    probabilidades = modelo.predict_proba(synthetic_reference[features])[:, 1]
    assert probabilidades.shape == (len(synthetic_reference),)


def test_teste_de_proxy_detecta_atributo_codificado(
    synthetic_reference: pd.DataFrame,
) -> None:
    """Se uma feature é função da idade, o proxy tem que acertar quase sempre."""
    referencia = synthetic_reference.copy()
    referencia["NumberOfDependents"] = (referencia["age"] < 30).astype(int)

    assert proxy_test(referencia) > 0.95


def test_auditoria_completa_gera_laudo_serializavel(
    synthetic_reference: pd.DataFrame, trained_bundle: dict, tmp_path: Path
) -> None:
    resultado = run_fairness_audit(
        batches={"lote_sintetico": synthetic_reference},
        reference=synthetic_reference,
        bundle=trained_bundle,
    )
    destino = save_fairness(resultado, tmp_path / "equidade.json")
    laudo = json.loads(destino.read_text(encoding="utf-8"))

    assert laudo["protected_attribute"] == "age"
    assert laudo["age_bands"] == list(AGE_LABELS)
    assert [r["model"] for r in laudo["production_model"]] == ["producao"]
    assert [r["model"] for r in laudo["model_without_age"]] == ["sem_idade"]
    assert 0.0 <= laudo["proxy_test"]["roc_auc"] <= 1.0
