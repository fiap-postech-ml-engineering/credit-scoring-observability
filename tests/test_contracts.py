"""Testes dos data contracts (Etapa 1).

O que importa aqui não é só "o contrato aceita dado bom", mas sobretudo "o
contrato **rejeita** cada defeito específico que ele promete rejeitar". Um
contrato que passa tudo é pior que nenhum, porque dá falsa segurança.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts.bad_batch import INJECTIONS, build_bad_batch
from src.contracts.schemas import CreditInferenceSchema, CreditReferenceSchema
from src.contracts.validate import (
    DataContractViolationError,
    enforce,
    save_result_json,
    validate_batch,
)

pytestmark = pytest.mark.contracts


def test_lote_valido_passa_no_contrato(synthetic_reference: pd.DataFrame) -> None:
    result = validate_batch(synthetic_reference, CreditReferenceSchema, "sintetico")
    assert result.passed
    assert result.n_failures == 0
    assert result.rules_violated == []
    assert result.rejection_rate == 0.0


def test_schema_de_inferencia_dispensa_o_alvo(synthetic_reference: pd.DataFrame) -> None:
    sem_alvo = synthetic_reference.drop(columns=["SeriousDlqin2yrs"])
    assert validate_batch(sem_alvo, CreditInferenceSchema, "sem_alvo").passed


def test_schema_de_referencia_exige_o_alvo(synthetic_reference: pd.DataFrame) -> None:
    sem_alvo = synthetic_reference.drop(columns=["SeriousDlqin2yrs"])
    assert not validate_batch(sem_alvo, CreditReferenceSchema, "sem_alvo").passed


@pytest.mark.parametrize(
    ("coluna", "valor", "regra_esperada"),
    [
        ("age", 15, "age:greater_than_or_equal_to(18)"),
        ("age", 130, "age:less_than_or_equal_to(100)"),
        ("MonthlyIncome", np.nan, "MonthlyIncome:not_nullable"),
        ("MonthlyIncome", -500.0, "MonthlyIncome:greater_than_or_equal_to(0.0)"),
        ("DebtRatio", 50_000.0, "DebtRatio:less_than_or_equal_to(10.0)"),
        (
            "RevolvingUtilizationOfUnsecuredLines",
            9_999.0,
            "RevolvingUtilizationOfUnsecuredLines:less_than_or_equal_to(10.0)",
        ),
        (
            "NumberOfTime30-59DaysPastDueNotWorse",
            98,
            "NumberOfTime30-59DaysPastDueNotWorse:less_than_or_equal_to(20)",
        ),
        ("NumberOfTimes90DaysLate", 96, "NumberOfTimes90DaysLate:less_than_or_equal_to(20)"),
        ("SeriousDlqin2yrs", 2, "SeriousDlqin2yrs:isin([0, 1])"),
        ("NumberOfDependents", np.nan, "NumberOfDependents:not_nullable"),
    ],
)
def test_cada_regra_rigida_intercepta_sua_violacao(
    synthetic_reference: pd.DataFrame, coluna: str, valor: object, regra_esperada: str
) -> None:
    """Uma linha corrompida por vez — a regra certa precisa disparar."""
    lote = synthetic_reference.copy()
    lote[coluna] = lote[coluna].astype("float64")
    lote.loc[lote.index[0], coluna] = valor

    result = validate_batch(lote, CreditReferenceSchema, "corrompido")

    assert not result.passed
    assert regra_esperada in result.rules_violated


def test_contrato_rejeita_duplicatas(synthetic_reference: pd.DataFrame) -> None:
    com_duplicatas = pd.concat(
        [synthetic_reference, synthetic_reference.iloc[:5]], ignore_index=True
    )
    result = validate_batch(com_duplicatas, CreditReferenceSchema, "duplicado")

    assert not result.passed
    assert any("duplicadas" in regra for regra in result.rules_violated)


def test_contrato_rejeita_coluna_desconhecida(synthetic_reference: pd.DataFrame) -> None:
    """`strict=True` impede que PII entre de carona num lote."""
    com_pii = synthetic_reference.copy()
    com_pii["cpf"] = "000.000.000-00"

    result = validate_batch(com_pii, CreditReferenceSchema, "com_pii")

    assert not result.passed
    assert any("column_in_schema" in regra for regra in result.rules_violated)


def test_contrato_rejeita_lote_vazio() -> None:
    vazio = pd.DataFrame(columns=list(CreditReferenceSchema.to_schema().columns))
    assert not validate_batch(vazio, CreditReferenceSchema, "vazio").passed


def test_enforce_bloqueia_a_ingestao(synthetic_reference: pd.DataFrame) -> None:
    """O portão levanta exceção: o dado ruim não segue para o modelo."""
    lote = synthetic_reference.copy()
    lote.loc[lote.index[0], "age"] = 10

    with pytest.raises(DataContractViolationError) as exc_info:
        enforce(lote, CreditReferenceSchema, "corrompido")

    assert "corrompido" in str(exc_info.value)
    assert exc_info.value.result.passed is False


def test_enforce_devolve_o_lote_coagido(synthetic_reference: pd.DataFrame) -> None:
    validado = enforce(synthetic_reference, CreditReferenceSchema, "sintetico")
    assert len(validado) == len(synthetic_reference)


def test_lote_corrompido_dispara_todas_as_injecoes(synthetic_reference: pd.DataFrame) -> None:
    """Teste de regressão do contrato: afrouxar uma regra quebra aqui."""
    lote, manifesto = build_bad_batch(source=synthetic_reference, n_rows=400)
    result = validate_batch(lote, CreditReferenceSchema, "lote_corrompido")

    assert not result.passed
    assert len(manifesto) == len(INJECTIONS)

    disparadas = set(result.rules_violated)
    nao_interceptadas = [
        (defeito.name, regra)
        for defeito in manifesto
        for regra in defeito.expected_rules
        if regra not in disparadas
    ]
    assert nao_interceptadas == []


def test_laudo_em_json_tem_o_que_o_dashboard_consome(
    synthetic_reference: pd.DataFrame, reports_dir
) -> None:
    import json

    lote, _ = build_bad_batch(source=synthetic_reference, n_rows=300)
    result = validate_batch(lote, CreditReferenceSchema, "lote_corrompido")
    caminho = save_result_json(result, reports_dir / "laudo.json")

    laudo = json.loads(caminho.read_text(encoding="utf-8"))
    for chave in ("batch_name", "passed", "n_rows", "n_failures", "failures_by_rule"):
        assert chave in laudo
    assert laudo["passed"] is False
    assert laudo["failures_by_rule"]
