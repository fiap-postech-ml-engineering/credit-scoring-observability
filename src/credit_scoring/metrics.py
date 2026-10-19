"""Métricas de qualidade do modelo de crédito, compartilhadas por treino e monitoramento.

O mesmo cálculo roda no treino (sobre o dataset de Referência) e no
monitoramento (sobre cada lote de produção rotulado). Reaproveitar a função
é o que torna comparável a frase "o AUC caiu 0,07 em relação à Referência".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Métricas de desempenho num conjunto rotulado."""

    roc_auc: float
    pr_auc: float
    ks_statistic: float
    brier: float
    precision: float
    recall: float
    f1: float
    default_rate: float
    approval_rate: float
    approved_bad_rate: float
    n_rows: int

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS de separação entre bons e maus pagadores (prática padrão em crédito).

    É o maior afastamento entre a taxa de verdadeiros positivos e a de falsos
    positivos ao longo da curva ROC.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def approved_bad_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Inadimplência *dentro da carteira aprovada* — o prejuízo que a fintech come.

    É a métrica de negócio que o drift de conceito ataca primeiro: o modelo
    continua aprovando a mesma fatia de clientes, mas uma parcela maior deles
    deixa de pagar. Nenhum teste de distribuição de features enxerga isso.
    """
    approved = y_pred == 0
    if not approved.any():
        return float("nan")
    return float(y_true[approved].mean())


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Calcula o conjunto completo de métricas para um lote rotulado."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    single_class = len(np.unique(y_true)) < 2

    return ClassificationMetrics(
        roc_auc=float("nan") if single_class else float(roc_auc_score(y_true, y_score)),
        pr_auc=float("nan") if single_class else float(average_precision_score(y_true, y_score)),
        ks_statistic=ks_statistic(y_true, y_score),
        brier=float(brier_score_loss(y_true, y_score)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        default_rate=float(y_true.mean()),
        approval_rate=float(1.0 - y_pred.mean()),
        approved_bad_rate=approved_bad_rate(y_true, y_pred),
        n_rows=int(len(y_true)),
    )
