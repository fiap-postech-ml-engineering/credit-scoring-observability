"""Data contracts (Pandera) do pipeline de Credit Scoring.

Este módulo é a fronteira entre "dado bruto" e "dado que o modelo pode
consumir". Nada entra no treino nem na inferência sem passar por aqui.

As regras foram derivadas da análise do dataset bruto Give Me Some Credit,
que viola várias delas de propósito — ver `docs/data_contracts.md`:

* `MonthlyIncome` tem 29.731 nulos (19,8% das linhas);
* `age` tem valor mínimo 0 (impossível para um contrato de crédito);
* existem 609 linhas duplicadas;
* os contadores de atraso usam 96/98 como códigos sentinela, não como
  quantidade real de ocorrências;
* `DebtRatio` e `RevolvingUtilizationOfUnsecuredLines` chegam a 329.664 e
  50.708 — proporções que não fazem sentido como razão.

Dois schemas são expostos:

* `CreditReferenceSchema` — contrato completo, com alvo. Usado na construção
  do dataset de Referência e na ingestão de lotes rotulados de produção.
* `CreditInferenceSchema` — mesmo contrato sem o alvo. Usado na API e em
  lotes de produção ainda não rotulados.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

# Limites de negócio dos contratos. Centralizados para que documentação,
# testes e geração de lotes inválidos leiam exatamente os mesmos números.
MIN_AGE = 18
MAX_AGE = 100
MAX_MONTHLY_INCOME = 1_000_000.0
MAX_DEBT_RATIO = 10.0
MAX_REVOLVING_UTILIZATION = 10.0
MAX_LATE_COUNT = 20
MAX_DEPENDENTS = 20
MAX_OPEN_CREDIT_LINES = 60
MAX_REAL_ESTATE_LOANS = 30


class CreditInferenceSchema(pa.DataFrameModel):
    """Contrato das features. Toda coluna é obrigatória e não aceita nulos."""

    # Regra rígida 1: utilização de crédito é uma proporção — não pode ser
    # negativa nem atingir as dezenas de milhares que aparecem no dado bruto.
    revolving_utilization: Series[float] = pa.Field(
        alias="RevolvingUtilizationOfUnsecuredLines",
        ge=0.0,
        le=MAX_REVOLVING_UTILIZATION,
        nullable=False,
    )

    # Regra rígida 2: só maiores de idade podem contratar crédito, e idade
    # acima de 100 é erro de cadastro. Intercepta os `age == 0` do bruto.
    age: Series[int] = pa.Field(ge=MIN_AGE, le=MAX_AGE, nullable=False)

    # Regra rígida 3: contadores de atraso não aceitam os sentinelas 96/98.
    late_30_59: Series[int] = pa.Field(
        alias="NumberOfTime30-59DaysPastDueNotWorse",
        ge=0,
        le=MAX_LATE_COUNT,
        nullable=False,
    )

    # Regra rígida 4: endividamento é razão dívida/renda; acima de 10x já é
    # outlier estrutural (o bruto chega a 329.664).
    debt_ratio: Series[float] = pa.Field(
        alias="DebtRatio", ge=0.0, le=MAX_DEBT_RATIO, nullable=False
    )

    # Regra rígida 5: renda é obrigatória para decisão de crédito. É a regra
    # que o lote bruto mais viola (19,8% de nulos).
    monthly_income: Series[float] = pa.Field(
        alias="MonthlyIncome", ge=0.0, le=MAX_MONTHLY_INCOME, nullable=False
    )

    open_credit_lines: Series[int] = pa.Field(
        alias="NumberOfOpenCreditLinesAndLoans",
        ge=0,
        le=MAX_OPEN_CREDIT_LINES,
        nullable=False,
    )

    late_90: Series[int] = pa.Field(
        alias="NumberOfTimes90DaysLate", ge=0, le=MAX_LATE_COUNT, nullable=False
    )

    real_estate_loans: Series[int] = pa.Field(
        alias="NumberRealEstateLoansOrLines",
        ge=0,
        le=MAX_REAL_ESTATE_LOANS,
        nullable=False,
    )

    late_60_89: Series[int] = pa.Field(
        alias="NumberOfTime60-89DaysPastDueNotWorse",
        ge=0,
        le=MAX_LATE_COUNT,
        nullable=False,
    )

    dependents: Series[int] = pa.Field(
        alias="NumberOfDependents", ge=0, le=MAX_DEPENDENTS, nullable=False
    )

    # O `ty` não modela a classe `Config` aninhada do Pandera, que é a forma
    # documentada de configurar um DataFrameModel. Os ignores abaixo valem só
    # para esse descompasso entre a biblioteca e o verificador.
    class Config:  # ty: ignore[invalid-attribute-override]
        """`strict` recusa colunas desconhecidas; `coerce` normaliza dtypes."""

        strict = True
        coerce = True
        name = "CreditInferenceContract"

    @pa.dataframe_check(name="sem_duplicatas", error="o lote contém linhas duplicadas")
    def no_duplicate_rows(cls, df: pd.DataFrame) -> bool:
        """Regra rígida 6: duplicata é dupla contagem do mesmo tomador."""
        return not df.duplicated().any()

    @pa.dataframe_check(name="lote_nao_vazio", error="o lote chegou vazio")
    def not_empty(cls, df: pd.DataFrame) -> bool:
        """Regra rígida 7: lote vazio é falha silenciosa de ingestão."""
        return len(df) > 0


class CreditReferenceSchema(CreditInferenceSchema):
    """Contrato completo: features + alvo. Usado no treino e em lotes rotulados."""

    # Regra rígida 8: o alvo é estritamente binário.
    target: Series[int] = pa.Field(alias="SeriousDlqin2yrs", isin=[0, 1], nullable=False)

    class Config:  # ty: ignore[invalid-attribute-override]
        strict = True
        coerce = True
        name = "CreditReferenceContract"
