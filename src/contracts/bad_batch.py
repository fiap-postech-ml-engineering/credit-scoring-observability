"""Gerador de lotes inválidos para demonstrar o contrato bloqueando a ingestão.

A Etapa 1 pede explicitamente que um lote com erros seja forçado contra o
contrato. Em vez de corromper o dado de forma genérica, aqui cada violação é
injetada de propósito, uma por regra do contrato, em linhas conhecidas — o
laudo de validação resultante mapeia 1:1 com as regras de
`src/contracts/schemas.py` e serve como teste de regressão do portão.

Cada entrada de `INJECTIONS` é um defeito realista de pipeline de crédito:
campo não preenchido no onboarding, idade digitada errada, sentinela de
sistema legado vazando como número, reprocessamento duplicando linhas.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

import numpy as np
import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InjectedDefect:
    """Um defeito injetado, como aparece no manifesto do lote."""

    name: str
    description: str
    expected_rules: tuple[str, ...]

    def to_dict(self) -> dict[str, str | list[str]]:
        return {
            "name": self.name,
            "description": self.description,
            "expected_rules": list(self.expected_rules),
        }


@dataclass(frozen=True)
class Injection:
    """Um defeito injetado, com a regra do contrato que deve pegá-lo."""

    name: str
    description: str
    expected_rules: tuple[str, ...]
    apply: Callable[[pd.DataFrame], pd.DataFrame]


def _nullify_income(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[:40], "MonthlyIncome"] = np.nan
    return df


def _minor_applicants(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[40:55], "age"] = [15, 16, 17, 0, 12, 3, 17, 16, 15, 11, 9, 17, 14, 16, 2]
    return df


def _impossible_age(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[55:60], "age"] = [142, 199, 320, 115, 101]
    return df


def _negative_income(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[60:70], "MonthlyIncome"] = -1500.0
    return df


def _sentinel_late_codes(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[70:80], "NumberOfTime30-59DaysPastDueNotWorse"] = 98
    df.loc[df.index[80:90], "NumberOfTimes90DaysLate"] = 96
    return df


def _absurd_debt_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[90:100], "DebtRatio"] = 329664.0
    return df


def _absurd_revolving(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[100:110], "RevolvingUtilizationOfUnsecuredLines"] = 50708.0
    return df


def _invalid_target(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[110:120], "SeriousDlqin2yrs"] = 2
    return df


def _null_dependents(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[120:135], "NumberOfDependents"] = np.nan
    return df


def _duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, df.iloc[200:230]], ignore_index=True)


def _unexpected_column(df: pd.DataFrame) -> pd.DataFrame:
    df["cpf"] = [f"{i % 97:011d}" for i in range(len(df))]
    return df


INJECTIONS: tuple[Injection, ...] = (
    Injection(
        "renda_nula",
        "40 propostas sem MonthlyIncome preenchido no onboarding",
        ("MonthlyIncome:not_nullable",),
        _nullify_income,
    ),
    Injection(
        "menor_de_idade",
        "15 solicitantes com menos de 18 anos (inclui age=0 do dado bruto)",
        ("age:greater_than_or_equal_to(18)",),
        _minor_applicants,
    ),
    Injection(
        "idade_impossivel",
        "5 idades acima de 100 — erro de digitação no cadastro",
        ("age:less_than_or_equal_to(100)",),
        _impossible_age,
    ),
    Injection(
        "renda_negativa",
        "10 propostas com renda negativa (sinal invertido na integração)",
        ("MonthlyIncome:greater_than_or_equal_to(0.0)",),
        _negative_income,
    ),
    Injection(
        "sentinela_atraso",
        "20 linhas com os códigos 96/98 de sistema legado vazando como contagem",
        (
            "NumberOfTime30-59DaysPastDueNotWorse:less_than_or_equal_to(20)",
            "NumberOfTimes90DaysLate:less_than_or_equal_to(20)",
        ),
        _sentinel_late_codes,
    ),
    Injection(
        "endividamento_absurdo",
        "10 linhas com DebtRatio de 329.664x a renda",
        ("DebtRatio:less_than_or_equal_to(10.0)",),
        _absurd_debt_ratio,
    ),
    Injection(
        "utilizacao_absurda",
        "10 linhas com utilização de crédito de 50.708x o limite",
        ("RevolvingUtilizationOfUnsecuredLines:less_than_or_equal_to(10.0)",),
        _absurd_revolving,
    ),
    Injection(
        "alvo_invalido",
        "10 linhas com alvo = 2 num problema binário",
        ("SeriousDlqin2yrs:isin([0, 1])",),
        _invalid_target,
    ),
    Injection(
        "dependentes_nulos",
        "15 linhas sem número de dependentes",
        ("NumberOfDependents:not_nullable",),
        _null_dependents,
    ),
    Injection(
        "coluna_inesperada",
        "coluna 'cpf' (PII!) adicionada por um upstream desavisado",
        ("CreditReferenceContract:column_in_schema",),
        _unexpected_column,
    ),
    # Último de propósito: duplicar depois da coluna extra garante que as
    # linhas repetidas sejam idênticas em todas as colunas.
    Injection(
        "linhas_duplicadas",
        "30 linhas reenviadas por reprocessamento do lote",
        ("CreditReferenceContract:o lote contém linhas duplicadas",),
        _duplicate_rows,
    ),
)


def build_bad_batch(
    source: pd.DataFrame | None = None,
    n_rows: int = 500,
    injections: tuple[Injection, ...] = INJECTIONS,
) -> tuple[pd.DataFrame, list[InjectedDefect]]:
    """Monta um lote inválido a partir de dados válidos, aplicando `injections`.

    Devolve o lote e o manifesto dos defeitos injetados, para que o
    relatório consiga comparar "o que foi injetado" com "o que o contrato
    pegou".
    """
    if source is None:
        source = pd.read_csv(settings.reference_path)

    batch = source.sample(
        n=min(n_rows, len(source)), random_state=settings.random_state
    ).reset_index(drop=True)

    manifest: list[InjectedDefect] = []
    for injection in injections:
        batch = injection.apply(batch)
        manifest.append(
            InjectedDefect(
                name=injection.name,
                description=injection.description,
                expected_rules=injection.expected_rules,
            )
        )

    logger.info(
        "lote inválido gerado",
        extra={"rows": len(batch), "injections": len(manifest)},
    )
    return batch, manifest
