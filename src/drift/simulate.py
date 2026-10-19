"""Simulação da passagem do tempo: lotes de produção que degradam (Etapa 2).

O pool futuro produzido por `src/credit_scoring/dataset.py` é a matéria-prima
de cinco lotes mensais que contam uma história econômica coerente, do mês
estável à recessão. Cada cenário declara **que tipo** de drift introduz, e a
distinção importa:

* **Data drift** (desvio de covariáveis) — a distribuição das features muda,
  mas a relação `P(inadimplência | features)` continua valendo. É o que os
  testes estatísticos (PSI, KS) enxergam.
* **Concept drift** — as features chegam com a mesma distribuição, mas a
  *relação* com o alvo mudou: o mesmo perfil de cliente passou a dar
  prejuízo. PSI e KS ficam **cegos** para isso; só o monitoramento de
  desempenho com rótulo pega.

O último lote (`2024-05_recessao`) existe justamente para demonstrar esse
ponto cego — é o cenário em que o modelo erra mais e os testes de data drift
não acusam quase nada. É a "degradação silenciosa" que a diretoria da fintech
suspeita no enunciado.

Os cenários são determinísticos: mesma `random_state`, mesmos lotes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.config.settings import TARGET, settings
from src.contracts.schemas import (
    MAX_DEBT_RATIO,
    MAX_MONTHLY_INCOME,
    MAX_REVOLVING_UTILIZATION,
    MIN_AGE,
)

logger = logging.getLogger(__name__)

DriftTransform = Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]


@dataclass(frozen=True)
class DriftScenario:
    """Um lote mensal de produção e o drift que ele carrega."""

    name: str
    month: str
    kind: str  # "nenhum" | "data" | "data+covariavel" | "concept"
    description: str
    expected_drift_features: tuple[str, ...]
    transform: DriftTransform | None = None
    n_rows: int = 8000
    seed_offset: int = 0
    tags: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transformações de data drift (mudam o valor das features)
# ---------------------------------------------------------------------------


def _inflate_income(factor: float) -> DriftTransform:
    """Inflação nominal da renda: o número sobe, o poder de compra não.

    Degrada o modelo de verdade porque ele aprendeu *limiares nominais* de
    renda no dataset de Referência — R$ 5.000 em 2024 não compram o que
    compravam no treino, mas o modelo trata o valor como se comprassem.
    """

    def apply(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = df.copy()
        # Ruído multiplicativo: a inflação não atinge todos igualmente.
        noise = rng.normal(loc=factor, scale=factor * 0.08, size=len(out))
        out["MonthlyIncome"] = (out["MonthlyIncome"] * noise).clip(0.0, MAX_MONTHLY_INCOME)
        return out

    return apply


def _increase_debt_ratio(factor: float) -> DriftTransform:
    """Endividamento das famílias sobe (juros altos, crédito rotativo mais caro)."""

    def apply(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = df.copy()
        noise = rng.normal(loc=factor, scale=factor * 0.10, size=len(out))
        out["DebtRatio"] = (out["DebtRatio"] * noise).clip(0.0, MAX_DEBT_RATIO)
        out["RevolvingUtilizationOfUnsecuredLines"] = (
            out["RevolvingUtilizationOfUnsecuredLines"] * noise
        ).clip(0.0, MAX_REVOLVING_UTILIZATION)
        return out

    return apply


def _shift_age_younger(years: float) -> DriftTransform:
    """Novo perfil de cliente: a base rejuvenesce (canal digital, conta paga)."""

    def apply(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = df.copy()
        shift = rng.normal(loc=years, scale=3.0, size=len(out))
        out["age"] = (out["age"] - shift).round().clip(MIN_AGE, 100).astype("int64")
        return out

    return apply


def _chain(*transforms: DriftTransform) -> DriftTransform:
    """Compõe transformações na ordem dada, reusando o mesmo gerador."""

    def apply(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = df
        for transform in transforms:
            out = transform(out, rng)
        return out

    return apply


# ---------------------------------------------------------------------------
# Transformação de concept drift (muda P(y|x), não a distribuição de x)
# ---------------------------------------------------------------------------


def _recession_concept_drift(intensity: float = 1.0) -> DriftTransform:
    """Recessão: um perfil antes seguro passa a dar prejuízo.

    O baseline aprendeu, no dado pré-crise, que cliente **jovem sem
    patrimônio imobiliário** não é especialmente arriscado — a idade pesa
    pouco e a ausência de financiamento imobiliário quase nada. Na recessão
    é exatamente esse grupo que perde o emprego primeiro e não tem colateral
    para queimar.

    A transformação vira rótulos `0 -> 1` nesse subgrupo, **sem tocar em
    nenhuma feature**. Ou seja: PSI e KS não têm o que detectar, e ainda
    assim o modelo passa a errar — porque ele aprovaria exatamente essas
    pessoas. É o ponto cego do monitoramento só-de-dados.
    """

    def apply(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = df.copy()
        if TARGET not in out.columns:
            return out

        # Vulnerabilidade à recessão: jovem, sem colateral, usando rotativo.
        youth = np.clip((45.0 - out["age"].to_numpy(dtype=float)) / 25.0, 0.0, 1.0)
        no_collateral = (out["NumberRealEstateLoansOrLines"].to_numpy() == 0).astype(float)
        revolving = np.clip(
            out["RevolvingUtilizationOfUnsecuredLines"].to_numpy(dtype=float) / 1.0, 0.0, 1.0
        )
        dependents = np.clip(out["NumberOfDependents"].to_numpy(dtype=float) / 3.0, 0.0, 1.0)

        vulnerability = youth * (0.45 + 0.55 * no_collateral) * (0.35 + 0.65 * revolving)
        vulnerability = np.clip(vulnerability * (0.7 + 0.3 * dependents), 0.0, 1.0)

        flip_probability = np.clip(0.55 * intensity * vulnerability, 0.0, 0.95)

        currently_good = out[TARGET].to_numpy() == 0
        flips = currently_good & (rng.random(len(out)) < flip_probability)
        out.loc[flips, TARGET] = 1

        logger.info(
            "concept drift aplicado",
            extra={
                "flipped_labels": int(flips.sum()),
                "default_rate_before": round(float(df[TARGET].mean()), 4),
                "default_rate_after": round(float(out[TARGET].mean()), 4),
            },
        )
        return out

    return apply


# ---------------------------------------------------------------------------
# Os cinco lotes mensais
# ---------------------------------------------------------------------------

SCENARIOS: tuple[DriftScenario, ...] = (
    DriftScenario(
        name="2024-01_estavel",
        month="2024-01",
        kind="nenhum",
        description=(
            "Mês de controle: amostra i.i.d. do pool futuro, sem nenhuma alteração. "
            "Serve para provar que o detector não dá falso positivo."
        ),
        expected_drift_features=(),
        transform=None,
        seed_offset=1,
    ),
    DriftScenario(
        name="2024-02_inflacao_leve",
        month="2024-02",
        kind="data",
        description="Início do ciclo inflacionário: renda nominal +8%.",
        expected_drift_features=("MonthlyIncome",),
        transform=_inflate_income(1.08),
        seed_offset=2,
    ),
    DriftScenario(
        name="2024-03_inflacao_forte",
        month="2024-03",
        kind="data",
        description=(
            "Inflação acelera (renda nominal +45%) e o endividamento das famílias "
            "sobe 60% com os juros altos. Duas features importantes deslocadas."
        ),
        expected_drift_features=(
            "MonthlyIncome",
            "DebtRatio",
            "RevolvingUtilizationOfUnsecuredLines",
        ),
        transform=_chain(_inflate_income(1.45), _increase_debt_ratio(1.60)),
        seed_offset=3,
    ),
    DriftScenario(
        name="2024-04_novo_perfil",
        month="2024-04",
        kind="data+covariavel",
        description=(
            "Campanha no canal digital traz uma base 12 anos mais jovem, somada à "
            "inflação e ao endividamento já instalados. Três features em drift."
        ),
        expected_drift_features=(
            "MonthlyIncome",
            "DebtRatio",
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
        ),
        transform=_chain(
            _inflate_income(1.50),
            _increase_debt_ratio(1.65),
            _shift_age_younger(12.0),
        ),
        seed_offset=4,
    ),
    DriftScenario(
        name="2024-05_recessao",
        month="2024-05",
        kind="concept",
        description=(
            "Recessão: as features chegam com a MESMA distribuição da Referência, "
            "mas jovens sem patrimônio e com rotativo alto — que o modelo aprovaria — "
            "passam a inadimplir. Drift de conceito puro: a inadimplência da carteira "
            "aprovada quase triplica e o PSI máximo fica em 0,002."
        ),
        expected_drift_features=(),
        transform=_recession_concept_drift(intensity=1.8),
        seed_offset=5,
        tags={"silent_degradation": "true"},
    ),
)


def build_batch(
    scenario: DriftScenario,
    pool: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Gera e persiste o lote de um cenário."""
    output_dir = output_dir or settings.production_dir
    rng = np.random.default_rng(settings.random_state + scenario.seed_offset)

    sample = pool.sample(
        n=min(scenario.n_rows, len(pool)),
        random_state=settings.random_state + scenario.seed_offset,
    ).reset_index(drop=True)

    batch = scenario.transform(sample, rng) if scenario.transform else sample

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{scenario.name}.csv"
    batch.to_csv(path, index=False)

    logger.info(
        "lote de produção gerado",
        extra={
            "batch": scenario.name,
            "kind": scenario.kind,
            "rows": len(batch),
            "default_rate": round(float(batch[TARGET].mean()), 4),
            "path": str(path),
        },
    )
    return batch, path


def build_all_batches(
    pool_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Gera os cinco lotes mensais a partir do pool futuro."""
    pool_path = pool_path or settings.production_dir / "future_pool.csv"
    if not pool_path.exists():
        from src.credit_scoring.dataset import prepare_datasets

        _, pool_path, _ = prepare_datasets()

    pool = pd.read_csv(pool_path)
    return {s.name: build_batch(s, pool, output_dir)[1] for s in SCENARIOS}


def scenario_by_name(name: str) -> DriftScenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(f"cenário desconhecido: {name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name, path in build_all_batches().items():
        logger.info("%s -> %s", name, path)
