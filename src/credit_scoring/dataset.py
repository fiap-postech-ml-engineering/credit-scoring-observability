"""Ingestão e preparo dos dados do Give Me Some Credit (Kaggle).

Três responsabilidades, nesta ordem:

1. **Ingestão** (`download_raw`) — baixa `cs-training.csv` via kagglehub. Não
   exige credenciais para este dataset específico (mirror público).
2. **Transformação** (`clean`) — aplica as correções mínimas para o lote
   passar no contrato de `src/contracts/schemas.py`, registrando quantas
   linhas cada passo descartou ou corrigiu. É esse log que a Etapa 3
   consome como métrica de saúde do pipeline.
3. **Separação** (`prepare_datasets`) — divide o dado limpo em **Referência**
   (usado no treino do baseline) e **pool futuro** (matéria-prima dos lotes
   de produção com drift, em `src/drift/simulate.py`). A separação é
   estratificada no alvo e o pool futuro nunca é visto pelo treino.

Decisões de limpeza (justificadas em `docs/data_contracts.md`):

* duplicatas são removidas — representariam dupla contagem do mesmo tomador;
* `age` fora de [18, 100] é descartada — erro de cadastro, não outlier;
* linhas com os códigos sentinela 96/98 nos contadores de atraso são
  descartadas: no dataset original elas significam "sem histórico
  avaliável", e não "96 atrasos" — manter viraria ruído;
* `DebtRatio` e `RevolvingUtilizationOfUnsecuredLines` são winsorizados no
  teto do contrato: são razões com cauda absurda, mas a linha em si é válida;
* `MonthlyIncome` nula é imputada pela mediana da faixa de idade, e
  `NumberOfDependents` nula por 0 (moda) — imputação explícita e auditável,
  nunca silenciosa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import FEATURES, TARGET, settings
from src.contracts.schemas import (
    MAX_AGE,
    MAX_DEBT_RATIO,
    MAX_DEPENDENTS,
    MAX_MONTHLY_INCOME,
    MAX_REAL_ESTATE_LOANS,
    MAX_REVOLVING_UTILIZATION,
    MIN_AGE,
)

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "brycecf/give-me-some-credit-dataset"
RAW_FILENAME = "cs-training.csv"

# Códigos sentinela dos contadores de atraso no dataset original.
SENTINEL_LATE_CODES = (96, 98)

LATE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


@dataclass
class CleaningStats:
    """Quantas linhas cada passo da transformação descartou ou corrigiu.

    Exposto como métrica de pipeline na Etapa 3: se `rows_out` cair de
    repente, a ingestão quebrou mesmo que nenhuma exceção tenha sido lançada.
    """

    rows_in: int = 0
    rows_out: int = 0
    dropped_duplicates: int = 0
    dropped_invalid_age: int = 0
    dropped_sentinel_late: int = 0
    dropped_duplicates_post: int = 0
    clipped_debt_ratio: int = 0
    clipped_revolving: int = 0
    clipped_real_estate: int = 0
    clipped_income: int = 0
    imputed_income: int = 0
    imputed_dependents: int = 0
    steps: list[str] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    @property
    def drop_rate(self) -> float:
        return self.rows_dropped / self.rows_in if self.rows_in else 0.0

    def to_dict(self) -> dict[str, float | int | list[str]]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": self.rows_dropped,
            "drop_rate": round(self.drop_rate, 6),
            "dropped_duplicates": self.dropped_duplicates,
            "dropped_invalid_age": self.dropped_invalid_age,
            "dropped_sentinel_late": self.dropped_sentinel_late,
            "dropped_duplicates_post": self.dropped_duplicates_post,
            "clipped_debt_ratio": self.clipped_debt_ratio,
            "clipped_revolving": self.clipped_revolving,
            "clipped_real_estate": self.clipped_real_estate,
            "clipped_income": self.clipped_income,
            "imputed_income": self.imputed_income,
            "imputed_dependents": self.imputed_dependents,
            "steps": self.steps,
        }


def download_raw(output_path: Path | None = None, force: bool = False) -> Path:
    """Garante `data/raw/credit.csv`, baixando do Kaggle apenas se necessário."""
    output_path = output_path or settings.raw_path
    if output_path.exists() and not force:
        logger.info("dataset bruto já presente", extra={"path": str(output_path)})
        return output_path

    logger.info("baixando dataset do Kaggle", extra={"dataset": KAGGLE_DATASET})
    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    df = pd.read_csv(dataset_dir / RAW_FILENAME, index_col=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("dataset bruto salvo", extra={"path": str(output_path), "rows": len(df)})
    return output_path


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """Aplica a transformação mínima para o lote satisfazer o contrato."""
    stats = CleaningStats(rows_in=len(df))
    out = df.copy()

    before = len(out)
    out = out.drop_duplicates()
    stats.dropped_duplicates = before - len(out)
    stats.steps.append(f"drop_duplicates: -{stats.dropped_duplicates}")

    before = len(out)
    out = out[out["age"].between(MIN_AGE, MAX_AGE)]
    stats.dropped_invalid_age = before - len(out)
    stats.steps.append(f"drop_invalid_age: -{stats.dropped_invalid_age}")

    before = len(out)
    sentinel_mask = np.zeros(len(out), dtype=bool)
    for col in LATE_COLUMNS:
        sentinel_mask |= out[col].isin(SENTINEL_LATE_CODES).to_numpy()
    out = out[~sentinel_mask]
    stats.dropped_sentinel_late = before - len(out)
    stats.steps.append(f"drop_sentinel_late: -{stats.dropped_sentinel_late}")

    # Winsorização das razões com cauda absurda: a linha é válida, o valor não.
    stats.clipped_revolving = int(
        (out["RevolvingUtilizationOfUnsecuredLines"] > MAX_REVOLVING_UTILIZATION).sum()
    )
    out["RevolvingUtilizationOfUnsecuredLines"] = out["RevolvingUtilizationOfUnsecuredLines"].clip(
        0.0, MAX_REVOLVING_UTILIZATION
    )
    stats.steps.append(f"clip_revolving: {stats.clipped_revolving}")

    stats.clipped_debt_ratio = int((out["DebtRatio"] > MAX_DEBT_RATIO).sum())
    out["DebtRatio"] = out["DebtRatio"].clip(0.0, MAX_DEBT_RATIO)
    stats.steps.append(f"clip_debt_ratio: {stats.clipped_debt_ratio}")

    stats.clipped_real_estate = int(
        (out["NumberRealEstateLoansOrLines"] > MAX_REAL_ESTATE_LOANS).sum()
    )
    out["NumberRealEstateLoansOrLines"] = out["NumberRealEstateLoansOrLines"].clip(
        0, MAX_REAL_ESTATE_LOANS
    )
    stats.steps.append(f"clip_real_estate: {stats.clipped_real_estate}")

    # Imputação explícita e auditável da renda: mediana da faixa de idade.
    stats.imputed_income = int(out["MonthlyIncome"].isna().sum())
    age_band = pd.cut(out["age"], bins=range(15, 105, 5))
    band_median = out.groupby(age_band, observed=True)["MonthlyIncome"].transform("median")
    out["MonthlyIncome"] = (
        out["MonthlyIncome"].fillna(band_median).fillna(out["MonthlyIncome"].median())
    )
    stats.steps.append(f"impute_income_by_age_band: {stats.imputed_income}")

    stats.clipped_income = int((out["MonthlyIncome"] > MAX_MONTHLY_INCOME).sum())
    out["MonthlyIncome"] = out["MonthlyIncome"].clip(0.0, MAX_MONTHLY_INCOME)
    stats.steps.append(f"clip_income: {stats.clipped_income}")

    stats.imputed_dependents = int(out["NumberOfDependents"].isna().sum())
    out["NumberOfDependents"] = (
        out["NumberOfDependents"].fillna(0).clip(0, MAX_DEPENDENTS).astype("int64")
    )
    stats.steps.append(f"impute_dependents: {stats.imputed_dependents}")

    ordered = [TARGET, *FEATURES] if TARGET in out.columns else list(FEATURES)
    out = out[ordered]

    # Winsorização e imputação colapsam linhas que antes diferiam só na cauda,
    # criando duplicatas novas. Sem esta segunda passada o contrato (que
    # proíbe duplicatas) reprovaria o próprio dataset de Referência.
    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    stats.dropped_duplicates_post = before - len(out)
    stats.steps.append(f"drop_duplicates_post_transform: -{stats.dropped_duplicates_post}")

    stats.rows_out = len(out)
    logger.info("transformação concluída", extra=stats.to_dict())
    return out, stats


def prepare_datasets(
    reference_path: Path | None = None,
    future_pool_path: Path | None = None,
    future_size: float = 0.4,
    force: bool = False,
) -> tuple[Path, Path, CleaningStats | None]:
    """Produz o dataset de Referência e o pool futuro (base dos lotes de produção).

    O split é estratificado no alvo, então Referência e pool futuro partem da
    mesma distribuição — qualquer drift observado depois é, por construção,
    introduzido de propósito em `src/drift/simulate.py`, e não artefato da
    separação.
    """
    reference_path = reference_path or settings.reference_path
    future_pool_path = future_pool_path or settings.production_dir / "future_pool.csv"

    if reference_path.exists() and future_pool_path.exists() and not force:
        logger.info("datasets de referência e pool futuro já presentes")
        return reference_path, future_pool_path, None

    raw_path = download_raw(force=force)
    raw = pd.read_csv(raw_path)
    clean_df, stats = clean(raw)

    reference, future_pool = train_test_split(
        clean_df,
        test_size=future_size,
        stratify=clean_df[TARGET],
        random_state=settings.random_state,
    )

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    future_pool_path.parent.mkdir(parents=True, exist_ok=True)
    reference.reset_index(drop=True).to_csv(reference_path, index=False)
    future_pool.reset_index(drop=True).to_csv(future_pool_path, index=False)

    logger.info(
        "datasets preparados",
        extra={
            "reference_rows": len(reference),
            "future_pool_rows": len(future_pool),
            "reference_path": str(reference_path),
            "future_pool_path": str(future_pool_path),
        },
    )
    return reference_path, future_pool_path, stats


def load_reference() -> pd.DataFrame:
    """Carrega o dataset de Referência, preparando-o se ainda não existir."""
    if not settings.reference_path.exists():
        prepare_datasets()
    return pd.read_csv(settings.reference_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ref, pool, _ = prepare_datasets()
    logger.info("pronto: referência=%s pool=%s", ref, pool)
