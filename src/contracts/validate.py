"""Portão de validação: decide se um lote de dados pode entrar no pipeline.

A função central é `enforce`, que valida um DataFrame contra um schema
Pandera e **levanta exceção** quando o contrato é violado — bloqueando a
ingestão em vez de deixar dado ruim chegar ao modelo. `validate_batch` faz a
mesma checagem sem levantar, para quem precisa inspecionar o resultado (API,
testes, relatórios).

A validação sempre roda em modo `lazy=True`: o objetivo não é parar no
primeiro erro, e sim produzir o laudo completo de tudo que está errado no
lote, agrupado por regra violada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors
import pandera.pandas as pa

logger = logging.getLogger(__name__)


class DataContractViolationError(Exception):
    """Lote rejeitado pelo contrato de dados. Interrompe a ingestão."""

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        super().__init__(
            f"lote '{result.batch_name}' rejeitado pelo contrato "
            f"{result.schema_name}: {result.n_failures} violações em "
            f"{len(result.rules_violated)} regras ({', '.join(result.rules_violated)})"
        )


@dataclass
class ValidationResult:
    """Laudo de uma validação de lote."""

    batch_name: str
    schema_name: str
    passed: bool
    n_rows: int
    n_failures: int
    failure_cases: pd.DataFrame = field(default_factory=pd.DataFrame)
    validated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def rules_violated(self) -> list[str]:
        """Nomes das regras violadas, sem repetição e em ordem estável."""
        if self.failure_cases.empty:
            return []
        pairs = self.failure_cases[["column", "check"]].astype(str).drop_duplicates()
        return sorted(f"{c}:{k}" for c, k in pairs.itertuples(index=False))

    @property
    def failures_by_rule(self) -> dict[str, int]:
        """Contagem de linhas reprovadas por (coluna, regra)."""
        if self.failure_cases.empty:
            return {}
        pairs = self.failure_cases[["column", "check"]].astype(str)
        counts: dict[str, int] = {}
        for column, check in pairs.itertuples(index=False):
            counts[f"{column}:{check}"] = counts.get(f"{column}:{check}", 0) + 1
        return counts

    @property
    def rejection_rate(self) -> float:
        """Fração de linhas com ao menos uma violação."""
        if self.failure_cases.empty or self.n_rows == 0:
            return 0.0
        offenders = self.failure_cases["index"].dropna().nunique()
        return float(offenders) / float(self.n_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_name": self.batch_name,
            "schema_name": self.schema_name,
            "passed": self.passed,
            "n_rows": self.n_rows,
            "n_failures": self.n_failures,
            "rejection_rate": round(self.rejection_rate, 6),
            "rules_violated": self.rules_violated,
            "failures_by_rule": self.failures_by_rule,
            "validated_at": self.validated_at,
        }


def _empty_failure_cases() -> pd.DataFrame:
    return pd.DataFrame(columns=["schema_context", "column", "check", "failure_case", "index"])


def _normalize_failure_cases(exc: SchemaErrors | SchemaError) -> pd.DataFrame:
    """Devolve as `failure_cases` do Pandera com as colunas que usamos sempre presentes."""
    cases = getattr(exc, "failure_cases", None)
    if cases is None or not isinstance(cases, pd.DataFrame) or cases.empty:
        return _empty_failure_cases()
    out = cases.copy()
    for col in ("schema_context", "column", "check", "failure_case", "index"):
        if col not in out.columns:
            out[col] = None
    return out


def validate_batch(
    df: pd.DataFrame,
    schema: type[pa.DataFrameModel],
    batch_name: str = "batch",
) -> ValidationResult:
    """Valida `df` contra `schema` e devolve o laudo, sem levantar exceção."""
    schema_name = getattr(getattr(schema, "Config", None), "name", schema.__name__)
    n_rows = len(df)

    try:
        schema.validate(df, lazy=True)
    except (SchemaErrors, SchemaError) as exc:
        cases = _normalize_failure_cases(exc)
        result = ValidationResult(
            batch_name=batch_name,
            schema_name=schema_name,
            passed=False,
            n_rows=n_rows,
            n_failures=len(cases),
            failure_cases=cases,
        )
        logger.warning(
            "contrato violado",
            extra={
                "batch": batch_name,
                "schema": schema_name,
                "n_rows": n_rows,
                "n_failures": result.n_failures,
                "failures_by_rule": result.failures_by_rule,
            },
        )
        return result

    logger.info(
        "contrato aprovado",
        extra={"batch": batch_name, "schema": schema_name, "n_rows": n_rows},
    )
    return ValidationResult(
        batch_name=batch_name,
        schema_name=schema_name,
        passed=True,
        n_rows=n_rows,
        n_failures=0,
        failure_cases=_empty_failure_cases(),
    )


def enforce(
    df: pd.DataFrame,
    schema: type[pa.DataFrameModel],
    batch_name: str = "batch",
) -> pd.DataFrame:
    """Valida `df` e o devolve coagido ao schema; levanta se o contrato falhar.

    Este é o portão usado pelo pipeline: se o lote não passa, a ingestão para
    aqui e o dado ruim nunca chega ao modelo.
    """
    result = validate_batch(df, schema, batch_name)
    if not result.passed:
        raise DataContractViolationError(result)
    return schema.validate(df, lazy=True)


def save_result_json(result: ValidationResult, path: Path) -> Path:
    """Persiste o laudo em JSON (consumido pelo dashboard e pelo CI)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
