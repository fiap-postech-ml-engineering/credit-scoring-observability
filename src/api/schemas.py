"""Schemas Pydantic de request/response da API de Credit Scoring.

Os limites declarados aqui são **os mesmos** do contrato Pandera em
`src/contracts/schemas.py`, importados de lá para não haver duas fontes de
verdade. Uma proposta rejeitada pelo contrato no pipeline batch é rejeitada
pela API com 422, e vice-versa.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.contracts.schemas import (
    MAX_AGE,
    MAX_DEBT_RATIO,
    MAX_DEPENDENTS,
    MAX_LATE_COUNT,
    MAX_MONTHLY_INCOME,
    MAX_OPEN_CREDIT_LINES,
    MAX_REAL_ESTATE_LOANS,
    MAX_REVOLVING_UTILIZATION,
    MIN_AGE,
)


class CreditApplication(BaseModel):
    """Uma proposta de crédito submetida para decisão.

    Os nomes em `snake_case` são a interface pública; o alias é o nome da
    coluna que o modelo espera. Nenhum campo de PII (CPF, nome, endereço)
    existe neste schema — ver a seção de LGPD no README.
    """

    model_config = ConfigDict(populate_by_name=True)

    revolving_utilization: float = Field(
        ge=0.0,
        le=MAX_REVOLVING_UTILIZATION,
        serialization_alias="RevolvingUtilizationOfUnsecuredLines",
        description="Utilização do crédito rotativo, como proporção do limite.",
    )
    age: int = Field(ge=MIN_AGE, le=MAX_AGE, description="Idade do solicitante, em anos.")
    late_30_59: int = Field(
        ge=0,
        le=MAX_LATE_COUNT,
        serialization_alias="NumberOfTime30-59DaysPastDueNotWorse",
        description="Ocorrências de atraso entre 30 e 59 dias.",
    )
    debt_ratio: float = Field(
        ge=0.0,
        le=MAX_DEBT_RATIO,
        serialization_alias="DebtRatio",
        description="Razão entre dívidas mensais e renda mensal.",
    )
    monthly_income: float = Field(
        ge=0.0,
        le=MAX_MONTHLY_INCOME,
        serialization_alias="MonthlyIncome",
        description="Renda mensal declarada.",
    )
    open_credit_lines: int = Field(
        ge=0,
        le=MAX_OPEN_CREDIT_LINES,
        serialization_alias="NumberOfOpenCreditLinesAndLoans",
        description="Linhas de crédito e empréstimos abertos.",
    )
    late_90: int = Field(
        ge=0,
        le=MAX_LATE_COUNT,
        serialization_alias="NumberOfTimes90DaysLate",
        description="Ocorrências de atraso de 90 dias ou mais.",
    )
    real_estate_loans: int = Field(
        ge=0,
        le=MAX_REAL_ESTATE_LOANS,
        serialization_alias="NumberRealEstateLoansOrLines",
        description="Financiamentos imobiliários e linhas com garantia real.",
    )
    late_60_89: int = Field(
        ge=0,
        le=MAX_LATE_COUNT,
        serialization_alias="NumberOfTime60-89DaysPastDueNotWorse",
        description="Ocorrências de atraso entre 60 e 89 dias.",
    )
    dependents: int = Field(
        ge=0,
        le=MAX_DEPENDENTS,
        serialization_alias="NumberOfDependents",
        description="Número de dependentes.",
    )


class ScoreResponse(BaseModel):
    """Decisão de crédito para uma proposta."""

    probability_of_default: float = Field(
        description="Probabilidade estimada de inadimplência grave em 2 anos."
    )
    decision: str = Field(description="`aprovado` ou `recusado`, conforme o limiar vigente.")
    threshold: float = Field(description="Limiar de decisão aplicado.")
    model_candidate: str = Field(description="Algoritmo do modelo em produção.")


class HealthResponse(BaseModel):
    """Estado de saúde do serviço."""

    status: str
    model_loaded: bool
    model_candidate: str | None = None
    baseline_roc_auc: float | None = None
