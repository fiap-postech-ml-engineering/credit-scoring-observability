"""Testes de simulação e detecção de drift (Etapa 2).

O teste mais importante deste arquivo é
`test_concept_drift_e_invisivel_para_o_psi`: ele formaliza a tese central do
projeto — PSI e KS não enxergam drift de conceito. Se um dia alguém "melhorar"
a simulação e esse teste quebrar, a demonstração perdeu o sentido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.settings import FEATURES, TARGET, settings
from src.drift.simulate import (
    SCENARIOS,
    _inflate_income,
    _recession_concept_drift,
    _shift_age_younger,
    build_batch,
    scenario_by_name,
)
from src.drift.statistics import (
    classify_psi,
    dataset_drift_summary,
    drift_table,
    feature_drift,
    population_stability_index,
)

pytestmark = pytest.mark.drift


# --- PSI --------------------------------------------------------------------


def test_psi_e_zero_para_distribuicoes_identicas(rng: np.random.Generator) -> None:
    amostra = rng.normal(0, 1, 5000)
    assert population_stability_index(amostra, amostra) == pytest.approx(0.0, abs=1e-9)


def test_psi_cresce_com_o_tamanho_do_deslocamento(rng: np.random.Generator) -> None:
    referencia = rng.normal(0, 1, 10000)
    psis = [
        population_stability_index(referencia, rng.normal(shift, 1, 10000))
        for shift in (0.1, 0.5, 1.0, 2.0)
    ]
    assert psis == sorted(psis), "PSI deveria crescer monotonicamente com o deslocamento"
    assert psis[0] < settings.psi_warn_threshold
    assert psis[-1] > settings.psi_alert_threshold


def test_psi_e_estavel_quanto_ao_tamanho_da_amostra(rng: np.random.Generator) -> None:
    """É por isso que o PSI é o critério, e o KS só apoio: KS reprova tudo com n grande."""
    referencia = rng.normal(0, 1, 50000)
    pequeno = population_stability_index(referencia, rng.normal(0, 1, 1000))
    grande = population_stability_index(referencia, rng.normal(0, 1, 50000))
    assert pequeno < settings.psi_warn_threshold
    assert grande < settings.psi_warn_threshold


def test_psi_lida_com_feature_constante() -> None:
    constante = np.zeros(1000)
    assert population_stability_index(constante, constante) == 0.0


def test_psi_com_amostra_vazia_devolve_nan() -> None:
    assert np.isnan(population_stability_index(np.array([]), np.array([1.0, 2.0])))


@pytest.mark.parametrize(
    ("psi", "severidade", "alerta"),
    [(0.01, "estavel", False), (0.15, "atencao", False), (0.40, "drift", True)],
)
def test_classificacao_das_faixas_de_psi(psi: float, severidade: str, alerta: bool) -> None:
    assert classify_psi(psi) == (severidade, alerta)


def test_feature_drift_reporta_o_deslocamento_da_media(rng: np.random.Generator) -> None:
    referencia = pd.Series(rng.normal(100, 10, 5000))
    atual = pd.Series(rng.normal(150, 10, 5000))

    resultado = feature_drift(referencia, atual, "renda")

    assert resultado.feature == "renda"
    assert resultado.drifted
    assert resultado.severity == "drift"
    assert resultado.mean_shift_pct == pytest.approx(50, abs=5)
    assert resultado.ks_p_value < settings.drift_p_value


# --- Transformações de drift ------------------------------------------------


def test_inflacao_aumenta_a_renda_sem_mexer_no_resto(
    synthetic_reference: pd.DataFrame, rng: np.random.Generator
) -> None:
    inflacionado = _inflate_income(1.5)(synthetic_reference, rng)

    assert inflacionado["MonthlyIncome"].mean() > synthetic_reference["MonthlyIncome"].mean()
    for coluna in ("age", "DebtRatio", TARGET):
        pd.testing.assert_series_equal(inflacionado[coluna], synthetic_reference[coluna])


def test_deslocamento_de_idade_respeita_o_piso_do_contrato(
    synthetic_reference: pd.DataFrame, rng: np.random.Generator
) -> None:
    rejuvenescido = _shift_age_younger(30.0)(synthetic_reference, rng)

    assert rejuvenescido["age"].mean() < synthetic_reference["age"].mean()
    assert rejuvenescido["age"].min() >= 18, "não pode produzir menor de idade"


def test_concept_drift_so_mexe_no_alvo(
    synthetic_reference: pd.DataFrame, rng: np.random.Generator
) -> None:
    """Nenhuma feature muda — só a relação delas com a inadimplência."""
    recessao = _recession_concept_drift(1.8)(synthetic_reference, rng)

    for feature in FEATURES:
        pd.testing.assert_series_equal(recessao[feature], synthetic_reference[feature])

    assert recessao[TARGET].mean() > synthetic_reference[TARGET].mean()


def test_concept_drift_e_invisivel_para_o_psi(
    synthetic_reference: pd.DataFrame, rng: np.random.Generator
) -> None:
    """A tese do projeto, como teste: PSI e KS ficam cegos para concept drift."""
    recessao = _recession_concept_drift(1.8)(synthetic_reference, rng)

    tabela = drift_table(synthetic_reference, recessao, FEATURES)
    resumo = dataset_drift_summary(tabela)

    assert resumo["n_drifted"] == 0
    assert resumo["max_psi"] < settings.psi_warn_threshold
    assert not resumo["dataset_drift"]


def test_concept_drift_so_piora_quem_estava_bom(
    synthetic_reference: pd.DataFrame, rng: np.random.Generator
) -> None:
    """Rótulos viram apenas de 0 para 1 — ninguém "desinadimple" numa recessão."""
    recessao = _recession_concept_drift(1.8)(synthetic_reference, rng)
    antes, depois = synthetic_reference[TARGET].to_numpy(), recessao[TARGET].to_numpy()

    assert not ((antes == 1) & (depois == 0)).any()


# --- Cenários ---------------------------------------------------------------


def test_cenarios_cobrem_controle_data_drift_e_concept_drift() -> None:
    tipos = {s.kind for s in SCENARIOS}
    assert "nenhum" in tipos, "falta o lote de controle, que prova a ausência de falso positivo"
    assert "data" in tipos
    assert "concept" in tipos


def test_cenario_de_controle_nao_transforma_nada() -> None:
    assert scenario_by_name("2024-01_estavel").transform is None


def test_nome_de_cenario_desconhecido_levanta() -> None:
    with pytest.raises(KeyError):
        scenario_by_name("2099-13_inexistente")


def test_lotes_sao_deterministicos(synthetic_reference: pd.DataFrame, tmp_path) -> None:
    """Mesma semente, mesmo lote — a demonstração precisa ser reproduzível."""
    cenario = scenario_by_name("2024-03_inflacao_forte")

    primeiro, _ = build_batch(cenario, synthetic_reference, tmp_path / "a")
    segundo, _ = build_batch(cenario, synthetic_reference, tmp_path / "b")

    pd.testing.assert_frame_equal(primeiro, segundo)


def test_lotes_com_data_drift_continuam_validos_no_contrato(
    synthetic_reference: pd.DataFrame, tmp_path
) -> None:
    """Drift não é dado inválido: o contrato deve deixar passar e o detector acusar.

    Confundir as duas coisas é o erro clássico — um lote em drift é dado
    legítimo de um mundo que mudou, não lixo a ser barrado.
    """
    from src.contracts.schemas import CreditReferenceSchema
    from src.contracts.validate import validate_batch

    for nome in ("2024-03_inflacao_forte", "2024-04_novo_perfil", "2024-05_recessao"):
        lote, _ = build_batch(scenario_by_name(nome), synthetic_reference, tmp_path / nome)
        assert validate_batch(lote, CreditReferenceSchema, nome).passed, (
            f"lote {nome} deveria passar no contrato: drift é dado válido"
        )
