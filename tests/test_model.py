"""Testes do dataset, das métricas e do treino do baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.settings import FEATURES, TARGET
from src.credit_scoring.dataset import LATE_COLUMNS, SENTINEL_LATE_CODES, clean
from src.credit_scoring.metrics import approved_bad_rate, compute_metrics, ks_statistic
from src.credit_scoring.train import score, train

pytestmark = pytest.mark.model


# --- Métricas ---------------------------------------------------------------


def test_ks_de_um_classificador_perfeito_e_um() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == pytest.approx(1.0)


def test_ks_de_uma_classe_so_e_nan() -> None:
    assert np.isnan(ks_statistic(np.zeros(10), np.random.default_rng(0).random(10)))


def test_inadimplencia_da_carteira_aprovada() -> None:
    #           aprovados (pred=0): índices 0,1,2 -> alvos 0,1,0 -> 1/3
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    assert approved_bad_rate(y_true, y_pred) == pytest.approx(1 / 3)


def test_inadimplencia_da_carteira_aprovada_sem_aprovados_e_nan() -> None:
    assert np.isnan(approved_bad_rate(np.array([0, 1]), np.array([1, 1])))


def test_metricas_de_classificador_perfeito() -> None:
    y = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y, np.array([0.05, 0.1, 0.9, 0.95]), threshold=0.5)

    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.approved_bad_rate == pytest.approx(0.0)
    assert metrics.n_rows == 4


def test_metricas_toleram_lote_de_uma_classe_so() -> None:
    """Acontece de verdade em lote pequeno; não pode derrubar o monitoramento."""
    metrics = compute_metrics(np.zeros(10), np.random.default_rng(0).random(10))
    assert np.isnan(metrics.roc_auc)
    assert metrics.default_rate == 0.0


# --- Limpeza do dataset -----------------------------------------------------


@pytest.fixture
def lote_bruto() -> pd.DataFrame:
    """Reproduz os defeitos reais do Give Me Some Credit."""
    base = {
        "SeriousDlqin2yrs": [0, 1, 0, 0, 1, 0],
        "RevolvingUtilizationOfUnsecuredLines": [0.5, 0.2, 50708.0, 0.3, 0.4, 0.5],
        "age": [45, 0, 52, 38, 61, 45],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 0, 98, 0, 0],
        "DebtRatio": [0.3, 0.5, 0.2, 0.4, 329664.0, 0.3],
        "MonthlyIncome": [5000.0, 4000.0, np.nan, 6000.0, 3000.0, 5000.0],
        "NumberOfOpenCreditLinesAndLoans": [8, 5, 10, 7, 9, 8],
        "NumberOfTimes90DaysLate": [0, 0, 0, 0, 1, 0],
        "NumberRealEstateLoansOrLines": [1, 0, 2, 1, 32, 1],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 0, 0, 0, 0],
        "NumberOfDependents": [2.0, np.nan, 1.0, 0.0, 3.0, 2.0],
    }
    return pd.DataFrame(base)


def test_limpeza_produz_lote_que_passa_no_contrato(lote_bruto: pd.DataFrame) -> None:
    from src.contracts.schemas import CreditReferenceSchema
    from src.contracts.validate import validate_batch

    limpo, _ = clean(lote_bruto)
    assert validate_batch(limpo, CreditReferenceSchema, "limpo").passed


def test_limpeza_descarta_idade_invalida_e_sentinelas(lote_bruto: pd.DataFrame) -> None:
    limpo, stats = clean(lote_bruto)

    assert stats.dropped_invalid_age == 1  # a linha com age = 0
    assert stats.dropped_sentinel_late == 1  # a linha com o código 98
    assert stats.dropped_duplicates == 1  # a última linha repete a primeira
    for coluna in LATE_COLUMNS:
        assert not limpo[coluna].isin(SENTINEL_LATE_CODES).any()


def test_limpeza_imputa_em_vez_de_descartar_renda_nula(lote_bruto: pd.DataFrame) -> None:
    limpo, stats = clean(lote_bruto)
    assert stats.imputed_income >= 1
    assert limpo["MonthlyIncome"].notna().all()


def test_limpeza_winsoriza_as_razoes_absurdas(lote_bruto: pd.DataFrame) -> None:
    limpo, stats = clean(lote_bruto)
    assert stats.clipped_revolving >= 1
    assert stats.clipped_debt_ratio >= 1
    assert limpo["DebtRatio"].max() <= 10.0
    assert limpo["RevolvingUtilizationOfUnsecuredLines"].max() <= 10.0


def test_estatisticas_de_limpeza_fecham_a_conta(lote_bruto: pd.DataFrame) -> None:
    limpo, stats = clean(lote_bruto)
    assert stats.rows_in == len(lote_bruto)
    assert stats.rows_out == len(limpo)
    assert stats.rows_dropped == stats.rows_in - stats.rows_out
    assert 0.0 <= stats.drop_rate <= 1.0


# --- Treino -----------------------------------------------------------------


def test_treino_salva_o_artefato_com_metadados(
    synthetic_reference: pd.DataFrame, tmp_path
) -> None:
    import joblib

    model_path = tmp_path / "model.joblib"
    train(reference=synthetic_reference, model_path=model_path, log_to_mlflow=False)

    bundle = joblib.load(model_path)
    for chave in ("model", "features", "threshold", "candidate", "validation_metrics"):
        assert chave in bundle
    assert bundle["features"] == FEATURES
    assert bundle["candidate"] in ("logistic_regression", "xgboost")


def test_treino_avalia_os_dois_candidatos(synthetic_reference: pd.DataFrame, tmp_path) -> None:
    _, _, candidatos = train(
        reference=synthetic_reference,
        model_path=tmp_path / "model.joblib",
        log_to_mlflow=False,
    )
    assert {c.name for c in candidatos} == {"logistic_regression", "xgboost"}
    assert all(c.metrics is not None for c in candidatos)


def test_treino_recusa_referencia_que_viola_o_contrato(
    synthetic_reference: pd.DataFrame, tmp_path
) -> None:
    """O portão também protege o treino, não só a inferência."""
    from src.contracts.validate import DataContractViolationError

    ruim = synthetic_reference.copy()
    ruim.loc[ruim.index[0], "age"] = 9

    with pytest.raises(DataContractViolationError):
        train(reference=ruim, model_path=tmp_path / "model.joblib", log_to_mlflow=False)


def test_modelo_treinado_pontua_entre_zero_e_um(
    trained_bundle: dict, synthetic_reference: pd.DataFrame
) -> None:
    scores = score(synthetic_reference, trained_bundle)
    assert len(scores) == len(synthetic_reference)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_modelo_aprende_algo_acima_do_acaso(
    trained_bundle: dict, synthetic_reference: pd.DataFrame
) -> None:
    scores = score(synthetic_reference, trained_bundle)
    metrics = compute_metrics(synthetic_reference[TARGET].to_numpy(), scores)
    assert metrics.roc_auc > 0.6, "o baseline precisa discriminar para o drift ser mensurável"
