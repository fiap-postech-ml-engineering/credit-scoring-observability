"""Testes estatísticos de drift: PSI e Kolmogorov-Smirnov.

O Evidently cobre a detecção com relatório visual, mas o enunciado pede um
**relatório estatístico** próprio evidenciando a mudança de distribuição. Este
módulo é essa evidência numérica, calculada de forma explícita e auditável —
dá para conferir cada número na mão.

As duas medidas são complementares:

* **PSI (Population Stability Index)** é a medida que a área de risco de
  crédito usa. Mede quanto a massa da distribuição migrou entre faixas,
  comparando a Referência com a produção. Não depende do tamanho da amostra,
  o que importa: com 8.000 linhas por lote, qualquer teste de hipótese
  rejeita a igualdade por diferenças irrelevantes.
* **KS (Kolmogorov-Smirnov)** dá o p-valor formal da hipótese "as duas
  amostras vêm da mesma distribuição". Sensível ao tamanho da amostra, então
  é lido como sinal de apoio, não como critério isolado.

Faixas de PSI usadas (convenção de mercado em credit scoring, configuráveis
em `settings`):

| PSI           | Leitura                                  |
|---------------|------------------------------------------|
| < 0,10        | estável                                  |
| 0,10 – 0,25   | atenção, monitorar                       |
| > 0,25        | drift relevante — investigar o modelo    |
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import TypedDict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Evita log(0) e divisão por zero quando uma faixa fica vazia em um dos lados.
_EPSILON = 1e-6
DEFAULT_BINS = 10


@dataclass(frozen=True)
class FeatureDrift:
    """Resultado dos testes para uma feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    reference_mean: float
    current_mean: float
    mean_shift_pct: float
    severity: str  # "estavel" | "atencao" | "drift"
    drifted: bool

    def to_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = DEFAULT_BINS,
) -> float:
    """PSI entre duas amostras, com faixas definidas pelos quantis da Referência.

    Usar os quantis da Referência (e não faixas de largura fixa) é o que faz o
    índice ser comparável entre features de escalas completamente diferentes,
    como `age` e `MonthlyIncome`.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))

    # Features muito concentradas (ex.: contadores quase sempre zero) podem
    # colapsar em uma única faixa; sem ao menos duas bordas não há PSI.
    if len(edges) < 2:
        return 0.0

    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / len(reference), _EPSILON, None)
    cur_pct = np.clip(cur_counts / len(current), _EPSILON, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify_psi(psi: float) -> tuple[str, bool]:
    """Traduz um PSI em severidade legível e na flag booleana de alerta."""
    if not np.isfinite(psi):
        return "indefinido", False
    if psi >= settings.psi_alert_threshold:
        return "drift", True
    if psi >= settings.psi_warn_threshold:
        return "atencao", False
    return "estavel", False


def feature_drift(reference: pd.Series, current: pd.Series, name: str) -> FeatureDrift:
    """Roda PSI e KS para uma feature e classifica a severidade."""
    ref_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)

    psi = population_stability_index(ref_values, cur_values)
    severity, drifted = classify_psi(psi)

    if len(ref_values) and len(cur_values):
        ks_result = ks_2samp(ref_values, cur_values)
        ks_stat, ks_p = float(ks_result.statistic), float(ks_result.pvalue)
        ref_mean, cur_mean = float(ref_values.mean()), float(cur_values.mean())
    else:
        ks_stat = ks_p = ref_mean = cur_mean = float("nan")

    shift_pct = (
        (cur_mean - ref_mean) / ref_mean * 100.0
        if np.isfinite(ref_mean) and abs(ref_mean) > _EPSILON
        else float("nan")
    )

    return FeatureDrift(
        feature=name,
        psi=round(psi, 6),
        ks_statistic=round(ks_stat, 6),
        ks_p_value=ks_p,
        reference_mean=round(ref_mean, 4),
        current_mean=round(cur_mean, 4),
        mean_shift_pct=round(shift_pct, 2) if np.isfinite(shift_pct) else float("nan"),
        severity=severity,
        drifted=drifted,
    )


def drift_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Tabela de drift feature por feature, ordenada do pior PSI para o melhor."""
    rows = [
        feature_drift(reference[f], current[f], f).to_dict()
        for f in features
        if f in reference.columns and f in current.columns
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("psi", ascending=False).reset_index(drop=True)


class DriftSummary(TypedDict):
    """Veredito agregado sobre um lote inteiro.

    É um `TypedDict` e não um dict solto porque este resumo viaja para o
    laudo JSON, para o MLflow, para o Prometheus e para o dashboard — cada
    consumidor lendo chaves fixas.
    """

    n_features: int
    n_drifted: int
    drift_share: float
    dataset_drift: bool
    max_psi: float
    drifted_features: list[str]


def dataset_drift_summary(table: pd.DataFrame) -> DriftSummary:
    """Agrega a tabela num veredito sobre o dataset inteiro."""
    if table.empty:
        return DriftSummary(
            n_features=0,
            n_drifted=0,
            drift_share=0.0,
            dataset_drift=False,
            max_psi=0.0,
            drifted_features=[],
        )

    drifted = table[table["drifted"]]
    share = len(drifted) / len(table)
    return DriftSummary(
        n_features=int(len(table)),
        n_drifted=int(len(drifted)),
        drift_share=round(float(share), 4),
        dataset_drift=bool(share >= settings.dataset_drift_share),
        max_psi=float(table["psi"].max()),
        drifted_features=list(drifted["feature"]),
    )
