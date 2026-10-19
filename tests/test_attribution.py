"""Testes do experimento de atribuição causal.

O valor deste experimento depende inteiramente do desenho estar limpo. Os
testes aqui verificam as propriedades que sustentam a conclusão, não os
números em si: se a condição de controle começar a mover o AUC, ou se "só
concept drift" começar a mexer nas features, a atribuição deixa de valer.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config.settings import FEATURES, TARGET
from src.drift.attribution import (
    Condition,
    _apply,
    decompose,
    run_attribution,
    save_attribution,
)

pytestmark = pytest.mark.drift


# --- Propriedades do desenho experimental -----------------------------------


def test_controle_nao_altera_nada(synthetic_reference: pd.DataFrame) -> None:
    """Sem mecanismo ligado, a amostra tem que sair idêntica à que entrou."""
    resultado = _apply(synthetic_reference, data_drift=False, concept_drift=False, seed=5)
    pd.testing.assert_frame_equal(resultado, synthetic_reference)


def test_so_data_drift_preserva_os_rotulos(synthetic_reference: pd.DataFrame) -> None:
    """`P(Y|X)` intacta: nenhum rótulo muda, só os valores das features."""
    resultado = _apply(synthetic_reference, data_drift=True, concept_drift=False, seed=5)

    pd.testing.assert_series_equal(resultado[TARGET], synthetic_reference[TARGET])
    assert not resultado["MonthlyIncome"].equals(synthetic_reference["MonthlyIncome"])


def test_so_concept_drift_preserva_as_features(synthetic_reference: pd.DataFrame) -> None:
    """`P(X)` intacta: nenhuma feature muda, só a relação com o alvo."""
    resultado = _apply(synthetic_reference, data_drift=False, concept_drift=True, seed=5)

    for feature in FEATURES:
        pd.testing.assert_series_equal(resultado[feature], synthetic_reference[feature])
    assert resultado[TARGET].sum() > synthetic_reference[TARGET].sum()


def test_condicao_combinada_move_features_e_rotulos(
    synthetic_reference: pd.DataFrame,
) -> None:
    resultado = _apply(synthetic_reference, data_drift=True, concept_drift=True, seed=5)

    assert not resultado["MonthlyIncome"].equals(synthetic_reference["MonthlyIncome"])
    assert resultado[TARGET].sum() > synthetic_reference[TARGET].sum()


def test_condicoes_sao_deterministicas(synthetic_reference: pd.DataFrame) -> None:
    """Mesma semente, mesmo resultado — senão a comparação entre condições não vale."""
    primeiro = _apply(synthetic_reference, True, True, seed=5)
    segundo = _apply(synthetic_reference, True, True, seed=5)
    pd.testing.assert_frame_equal(primeiro, segundo)


# --- Decomposição -----------------------------------------------------------


def test_decomposicao_desconta_o_ruido_amostral() -> None:
    """O efeito de cada mecanismo é medido contra o controle, não contra zero."""
    conditions = [
        Condition("controle", "", False, False, auc_delta=-0.0100),
        Condition("so_data_drift", "", True, False, auc_delta=-0.0400),
        Condition("so_concept_drift", "", False, True, auc_delta=-0.0600),
        Condition("ambos", "", True, True, auc_delta=-0.1100),
    ]

    resultado = decompose(conditions)

    assert resultado["ruido_amostral"] == pytest.approx(-0.01)
    assert resultado["efeito_data_drift"] == pytest.approx(-0.03)
    assert resultado["efeito_concept_drift"] == pytest.approx(-0.05)
    assert resultado["efeito_conjunto"] == pytest.approx(-0.10)
    # Interação = conjunto − soma dos isolados.
    assert resultado["interacao"] == pytest.approx(-0.02)


def test_shares_somam_um() -> None:
    conditions = [
        Condition("controle", "", False, False, auc_delta=0.0),
        Condition("so_data_drift", "", True, False, auc_delta=-0.03),
        Condition("so_concept_drift", "", False, True, auc_delta=-0.05),
        Condition("ambos", "", True, True, auc_delta=-0.10),
    ]
    resultado = decompose(conditions)
    assert resultado["share_data_drift"] + resultado["share_concept_drift"] == pytest.approx(1.0)
    assert resultado["share_concept_drift"] > resultado["share_data_drift"]


def test_decomposicao_sem_efeito_nao_divide_por_zero() -> None:
    conditions = [
        Condition(nome, "", False, False, auc_delta=0.0)
        for nome in ("controle", "so_data_drift", "so_concept_drift", "ambos")
    ]
    resultado = decompose(conditions)
    assert resultado["share_data_drift"] == 0.0
    assert resultado["share_concept_drift"] == 0.0


# --- Experimento completo ---------------------------------------------------


@pytest.fixture
def amostra_com_sinal_forte() -> pd.DataFrame:
    """Amostra grande com a inadimplência fortemente dependente da RENDA.

    O `synthetic_reference` padrão é pequeno e de sinal fraco (AUC ~0,6): nele
    o efeito do data drift se perde no ruído amostral e o experimento não tem
    o que medir. Aqui a renda é determinante, então inflacioná-la quebra de
    verdade os limiares nominais que o modelo aprendeu — que é exatamente o
    mecanismo que o experimento precisa conseguir enxergar.
    """
    import numpy as np

    generator = np.random.default_rng(7)
    n = 6000

    frame = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": generator.uniform(0.0, 1.0, n).round(4),
            "age": generator.integers(25, 75, n),
            "NumberOfTime30-59DaysPastDueNotWorse": generator.integers(0, 4, n),
            "DebtRatio": generator.uniform(0.0, 2.0, n).round(4),
            "MonthlyIncome": generator.uniform(1500.0, 15000.0, n).round(2),
            "NumberOfOpenCreditLinesAndLoans": generator.integers(1, 20, n),
            "NumberOfTimes90DaysLate": generator.integers(0, 3, n),
            "NumberRealEstateLoansOrLines": generator.integers(0, 5, n),
            "NumberOfTime60-89DaysPastDueNotWorse": generator.integers(0, 3, n),
            "NumberOfDependents": generator.integers(0, 5, n),
        }
    )

    # Renda baixa domina o risco: é o limiar nominal que a inflação corrói.
    risco = (
        3.0 * (1.0 - frame["MonthlyIncome"] / 15000.0)
        + 1.2 * frame["RevolvingUtilizationOfUnsecuredLines"]
        + 0.8 * frame["NumberOfTimes90DaysLate"]
        - 2.0
    )
    frame[TARGET] = (generator.random(n) < 1 / (1 + np.exp(-risco))).astype(int)

    return frame[[TARGET, *FEATURES]].drop_duplicates().reset_index(drop=True)


@pytest.mark.slow
def test_experimento_completo_sobre_dado_sintetico(
    amostra_com_sinal_forte: pd.DataFrame, tmp_path
) -> None:
    """Espelha o desenho real: o pool é holdout, nunca visto pelo treino.

    Usar a mesma amostra como treino e pool daria AUC dentro da amostra e o
    controle apareceria "melhor" que a baseline — exatamente o viés que a
    separação Referência/pool futuro existe para evitar.
    """
    import joblib

    from src.credit_scoring.train import train

    metade = len(amostra_com_sinal_forte) // 2
    referencia = amostra_com_sinal_forte.iloc[:metade].reset_index(drop=True)
    pool = amostra_com_sinal_forte.iloc[metade:].reset_index(drop=True)

    model_path = tmp_path / "model.joblib"
    train(reference=referencia, model_path=model_path, log_to_mlflow=False)
    bundle = joblib.load(model_path)

    conditions = run_attribution(
        reference=referencia,
        pool=pool,
        bundle=bundle,
        n_rows=len(pool),
    )

    assert [c.name for c in conditions] == [
        "controle",
        "so_data_drift",
        "so_concept_drift",
        "ambos",
    ]

    por_nome = {c.name: c for c in conditions}

    # Asserções sobre ORDENAÇÃO, não sobre valores absolutos: com algumas
    # centenas de linhas sintéticas o ruído amostral do AUC é legitimamente
    # grande (no pipeline real, com lotes de 8.000, o controle fica em
    # +0,0005). O que precisa valer em qualquer escala é a ordem dos efeitos.
    assert por_nome["controle"].roc_auc > por_nome["so_data_drift"].roc_auc
    assert por_nome["controle"].roc_auc > por_nome["so_concept_drift"].roc_auc
    assert por_nome["ambos"].roc_auc < por_nome["controle"].roc_auc

    # Só data drift: PSI acusa.
    assert por_nome["so_data_drift"].n_drifted_features > 0
    # Só concept drift: PSI fica cego — a tese do projeto.
    assert por_nome["so_concept_drift"].n_drifted_features == 0
    assert por_nome["so_concept_drift"].max_psi < 0.1
    # E ainda assim a inadimplência sobe.
    assert por_nome["so_concept_drift"].default_rate > por_nome["controle"].default_rate
    # Data drift puro não mexe no rótulo: a inadimplência fica igual ao controle.
    assert por_nome["so_data_drift"].default_rate == por_nome["controle"].default_rate

    destino = save_attribution(conditions, tmp_path / "atribuicao.json")
    payload = json.loads(destino.read_text(encoding="utf-8"))
    assert len(payload["conditions"]) == 4
    assert "interacao" in payload["decomposition"]
