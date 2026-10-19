"""Atribuição causal da degradação: quanto é data drift e quanto é concept drift.

O enunciado pede uma análise causal do problema. "O AUC caiu 0,045" não é
diagnóstico — é sintoma. O diagnóstico responde **por qual mecanismo** caiu,
porque a resposta operacional é diferente em cada caso:

* **data drift** (as features mudaram, `P(X)` ≠ referência, mas `P(Y|X)` vale):
  o modelo continua certo sobre o mundo, só está operando fora do domínio em
  que foi treinado. Recalibrar limiar ou reponderar amostra costuma resolver;
* **concept drift** (`P(Y|X)` mudou): o que o modelo aprendeu virou mentira.
  Nenhuma recalibração salva — tem que retreinar com dado do regime novo.

Normalmente separar as duas coisas em produção é difícil, porque só se observa
o efeito combinado. Aqui é possível fazer de forma limpa: nós controlamos o
processo gerador em `src/drift/simulate.py`, então dá para rodar o
experimento fatorial — aplicar cada mecanismo **isolado** sobre a mesma
amostra-base, com a mesma semente, e medir o efeito de cada um.

É o desenho contrafactual clássico: mesma amostra, mesmo modelo, só muda o
mecanismo ligado. A diferença entre o efeito conjunto e a soma dos isolados é
a **interação** — o quanto os dois mecanismos se potencializam.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import TARGET, settings
from src.credit_scoring.metrics import compute_metrics
from src.credit_scoring.train import load_model, score
from src.drift.simulate import (
    _chain,
    _increase_debt_ratio,
    _inflate_income,
    _recession_concept_drift,
    _shift_age_younger,
)
from src.drift.statistics import dataset_drift_summary, drift_table
from src.observability.logging_setup import safe_extra

logger = logging.getLogger(__name__)


@dataclass
class Condition:
    """Uma condição do experimento: quais mecanismos estão ligados."""

    name: str
    description: str
    data_drift: bool
    concept_drift: bool
    roc_auc: float = 0.0
    auc_delta: float = 0.0
    approved_bad_rate: float = 0.0
    default_rate: float = 0.0
    max_psi: float = 0.0
    n_drifted_features: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply(sample: pd.DataFrame, data_drift: bool, concept_drift: bool, seed: int) -> pd.DataFrame:
    """Aplica os mecanismos pedidos, sempre a partir da mesma amostra-base."""
    out = sample.copy()

    # A semente é a mesma em todas as condições: a diferença entre elas é o
    # mecanismo ligado, nunca o sorteio.
    if data_drift:
        rng = np.random.default_rng(seed)
        out = _chain(
            _inflate_income(1.50),
            _increase_debt_ratio(1.65),
            _shift_age_younger(12.0),
        )(out, rng)

    if concept_drift:
        rng = np.random.default_rng(seed)
        out = _recession_concept_drift(intensity=1.8)(out, rng)

    return out


def run_attribution(
    reference: pd.DataFrame | None = None,
    pool: pd.DataFrame | None = None,
    bundle: dict[str, Any] | None = None,
    n_rows: int = 8000,
    seed: int = 5,
) -> list[Condition]:
    """Roda o experimento fatorial 2×2 e devolve as quatro condições medidas."""
    bundle = bundle or load_model()
    reference = reference if reference is not None else pd.read_csv(settings.reference_path)
    pool = pool if pool is not None else pd.read_csv(settings.production_dir / "future_pool.csv")

    baseline = bundle["validation_metrics"]
    base_sample = pool.sample(
        n=min(n_rows, len(pool)), random_state=settings.random_state + seed
    ).reset_index(drop=True)

    conditions = [
        Condition(
            "controle",
            "Nenhum mecanismo ligado — mede só o ruído amostral.",
            data_drift=False,
            concept_drift=False,
        ),
        Condition(
            "so_data_drift",
            "Só as features mudam (inflação, endividamento, idade). P(Y|X) intacta.",
            data_drift=True,
            concept_drift=False,
        ),
        Condition(
            "so_concept_drift",
            "Features idênticas à referência; só a relação com a inadimplência muda.",
            data_drift=False,
            concept_drift=True,
        ),
        Condition(
            "ambos",
            "Os dois mecanismos simultâneos — o cenário real de uma recessão.",
            data_drift=True,
            concept_drift=True,
        ),
    ]

    for condition in conditions:
        batch = _apply(base_sample, condition.data_drift, condition.concept_drift, seed)
        scores = score(batch, bundle)
        metrics = compute_metrics(batch[TARGET].to_numpy(), scores, settings.decision_threshold)
        summary = dataset_drift_summary(drift_table(reference, batch, list(bundle["features"])))

        condition.roc_auc = round(metrics.roc_auc, 4)
        condition.auc_delta = round(metrics.roc_auc - float(baseline["roc_auc"]), 4)
        condition.approved_bad_rate = round(metrics.approved_bad_rate, 4)
        condition.default_rate = round(metrics.default_rate, 4)
        condition.max_psi = round(float(summary["max_psi"]), 4)
        condition.n_drifted_features = int(summary["n_drifted"])

        logger.info("condição medida", extra=safe_extra(condition.to_dict()))

    return conditions


def decompose(conditions: list[Condition]) -> dict[str, float]:
    """Decompõe a queda de AUC nos efeitos isolados e na interação."""
    by_name = {c.name: c for c in conditions}
    controle = by_name["controle"].auc_delta
    so_data = by_name["so_data_drift"].auc_delta - controle
    so_concept = by_name["so_concept_drift"].auc_delta - controle
    ambos = by_name["ambos"].auc_delta - controle

    total = abs(so_data) + abs(so_concept)
    return {
        "ruido_amostral": round(controle, 4),
        "efeito_data_drift": round(so_data, 4),
        "efeito_concept_drift": round(so_concept, 4),
        "efeito_conjunto": round(ambos, 4),
        "interacao": round(ambos - so_data - so_concept, 4),
        "share_data_drift": round(abs(so_data) / total, 4) if total else 0.0,
        "share_concept_drift": round(abs(so_concept) / total, 4) if total else 0.0,
    }


def save_attribution(conditions: list[Condition], path: Path | None = None) -> Path:
    """Persiste o experimento e a decomposição em JSON."""
    path = path or settings.reports_dir / "atribuicao_causal.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "conditions": [c.to_dict() for c in conditions],
        "decomposition": decompose(conditions),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    from src.observability.logging_setup import setup_logging

    setup_logging()
    results = run_attribution()
    destino = save_attribution(results)

    header = f"{'condição':20} {'AUC':>7} {'ΔAUC':>8} {'inadimpl.':>10} {'aprov.ruim':>11} {'PSI máx':>8} {'nDrift':>7}"
    logger.info("\n%s\n%s", header, "-" * len(header))
    for item in results:
        logger.info(
            "%-20s %7.4f %+8.4f %10.4f %11.4f %8.4f %7d",
            item.name,
            item.roc_auc,
            item.auc_delta,
            item.default_rate,
            item.approved_bad_rate,
            item.max_psi,
            item.n_drifted_features,
        )
    logger.info("decomposição: %s", json.dumps(decompose(results), ensure_ascii=False))
    logger.info("laudo salvo em %s", destino)
