"""Auditoria de equidade por faixa etária: o viés medido, não só discutido.

O Give Me Some Credit não traz gênero, raça ou estado civil, mas traz
**idade** — critério protegido em várias jurisdições e tema sensível na oferta
de crédito no Brasil. Então dá para sair do "o modelo é justo?" para números:

* **paridade demográfica** — a taxa de aprovação é parecida entre as faixas?
  Resumida pela *razão de impacto desigual* (menor taxa / maior taxa); a
  convenção dos "quatro quintos" trata razão abaixo de 0,8 como sinal de
  impacto desigual. É uma convenção de auditoria (EEOC, EUA), não um limite
  legal brasileiro — usamos como limiar de alerta, documentado;
* **igualdade de oportunidade** — entre os clientes que **de fato pagam**, a
  chance de ser aprovado é parecida entre as faixas? É a métrica que importa
  para o bom pagador: paridade demográfica pode ser violada por uma diferença
  legítima de risco, mas recusar mais um bom pagador só pela faixa etária não;
* **custo do erro por faixa** — inadimplência dentro da carteira aprovada de
  cada faixa: quem paga a conta quando o modelo erra.

Dois experimentos respondem à pergunta "tirar a idade do modelo resolve?":

1. **modelo sem idade** — o mesmo estimador, retreinado na mesma Referência
   sem a coluna `age`, medido nos mesmos lotes;
2. **teste de proxy** — um classificador tenta prever a faixa etária a partir
   das *outras* nove features. Se acerta bem, a idade está codificada nelas e
   remover a coluna não remove a informação.

O lote `2024-01_estavel` é a linha de base honesta: vem do pool futuro, nunca
visto no treino, e não tem drift. Medir equidade na própria Referência
mediria o modelo sobre dados que ele memorizou.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.config.settings import FEATURES, TARGET, settings
from src.credit_scoring.train import load_model, score
from src.observability.logging_setup import safe_extra

logger = logging.getLogger(__name__)

PROTECTED = "age"

# Faixas etárias da auditoria. Os cortes seguem ciclos de vida usuais em
# crédito (início de carreira, formação de família, maturidade, aposentadoria).
AGE_BINS = (17, 29, 44, 59, 200)
AGE_LABELS = ("18-29", "30-44", "45-59", "60+")

# Faixa mais jovem: é a que o cenário de recessão atinge e a que o teste de
# proxy tenta reconhecer.
YOUNG_BAND = AGE_LABELS[0]

# Limiares de alerta. A razão de 0,8 é a regra dos quatro quintos; a diferença
# de 0,10 em aprovação de bons pagadores é escolha deste projeto — dez pontos
# percentuais de diferença na chance de um bom pagador ser aprovado.
DISPARATE_IMPACT_MIN = 0.8
EQUAL_OPPORTUNITY_MAX_GAP = 0.10

# AUC acima disso no teste de proxy indica que as outras features codificam a
# faixa etária bem o bastante para remover `age` ser inócuo.
PROXY_AUC_ALERT = 0.70


@dataclass(frozen=True)
class GroupMetrics:
    """Desempenho e decisão do modelo dentro de uma faixa etária."""

    group: str
    n_rows: int
    share: float
    default_rate: float
    approval_rate: float
    good_payer_approval_rate: float
    bad_payer_approval_rate: float
    approved_bad_rate: float


@dataclass
class FairnessReport:
    """Auditoria de um lote: métricas por faixa e as disparidades resumidas."""

    batch: str
    model: str
    roc_auc: float
    groups: list[GroupMetrics]
    disparate_impact_ratio: float = 0.0
    equal_opportunity_gap: float = 0.0
    approved_bad_rate_gap: float = 0.0
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["groups"] = [asdict(g) for g in self.groups]
        return payload


def age_band(age: pd.Series) -> pd.Series:
    """Classifica cada idade na sua faixa da auditoria."""
    return pd.cut(age, bins=list(AGE_BINS), labels=list(AGE_LABELS)).astype(str)


def _rate(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


def group_metrics(
    y_true: np.ndarray, approved: np.ndarray, groups: pd.Series
) -> list[GroupMetrics]:
    """Calcula as métricas de cada faixa presente no lote."""
    y_true = np.asarray(y_true).astype(int)
    approved = np.asarray(approved).astype(bool)
    labels = groups.to_numpy()
    total = len(y_true)

    result: list[GroupMetrics] = []
    for label in AGE_LABELS:
        in_group = labels == label
        if not in_group.any():
            continue
        y_g, a_g = y_true[in_group], approved[in_group]
        result.append(
            GroupMetrics(
                group=label,
                n_rows=int(in_group.sum()),
                share=round(float(in_group.sum()) / total, 4),
                default_rate=round(_rate(y_g == 1), 4),
                approval_rate=round(_rate(a_g), 4),
                good_payer_approval_rate=round(_rate(a_g[y_g == 0]), 4),
                bad_payer_approval_rate=round(_rate(a_g[y_g == 1]), 4),
                approved_bad_rate=round(_rate(y_g[a_g] == 1), 4),
            )
        )
    return result


def _spread(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if not np.isnan(v)]
    if not clean:
        return float("nan"), float("nan")
    return min(clean), max(clean)


def audit(
    batch: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    batch_name: str,
    model_name: str,
) -> FairnessReport:
    """Audita um lote rotulado já pontuado pelo modelo."""
    y_true = batch[TARGET].to_numpy()
    approved = np.asarray(scores) < threshold
    groups = group_metrics(y_true, approved, age_band(batch[PROTECTED]))

    low, high = _spread([g.approval_rate for g in groups])
    eo_low, eo_high = _spread([g.good_payer_approval_rate for g in groups])
    bad_low, bad_high = _spread([g.approved_bad_rate for g in groups])

    report = FairnessReport(
        batch=batch_name,
        model=model_name,
        roc_auc=round(float(roc_auc_score(y_true, scores)), 4),
        groups=groups,
        disparate_impact_ratio=round(low / high, 4) if high else float("nan"),
        equal_opportunity_gap=round(eo_high - eo_low, 4),
        approved_bad_rate_gap=round(bad_high - bad_low, 4),
    )
    if report.disparate_impact_ratio < DISPARATE_IMPACT_MIN:
        report.alerts.append("impacto_desigual")
    if report.equal_opportunity_gap > EQUAL_OPPORTUNITY_MAX_GAP:
        report.alerts.append("oportunidade_desigual")
    return report


def retrain_without(
    bundle: dict[str, Any], reference: pd.DataFrame, drop: str
) -> tuple[Any, list[str]]:
    """Retreina o mesmo estimador do baseline sem a coluna `drop`.

    `clone` reaproveita exatamente os hiperparâmetros do vencedor: a única
    diferença entre os dois modelos é a coluna removida.
    """
    features = [f for f in bundle["features"] if f != drop]
    model = clone(bundle["model"])
    model.fit(reference[features], reference[TARGET])
    return model, features


def proxy_test(reference: pd.DataFrame, target_band: str = YOUNG_BAND, cv: int = 3) -> float:
    """AUC de um classificador que tenta prever a faixa a partir das outras features."""
    features = [f for f in FEATURES if f != PROTECTED]
    y = (age_band(reference[PROTECTED]) == target_band).astype(int)
    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=settings.random_state,
    )
    scores = cross_val_score(classifier, reference[features], y, cv=cv, scoring="roc_auc")
    return round(float(np.mean(scores)), 4)


def run_fairness_audit(
    batches: dict[str, pd.DataFrame] | None = None,
    reference: pd.DataFrame | None = None,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audita todos os lotes com o modelo em produção e com o modelo sem idade."""
    bundle = bundle or load_model()
    reference = reference if reference is not None else pd.read_csv(settings.reference_path)
    if batches is None:
        from src.drift.simulate import SCENARIOS

        batches = {
            s.name: pd.read_csv(settings.production_dir / f"{s.name}.csv") for s in SCENARIOS
        }

    threshold = float(bundle["threshold"])
    model_without_age, features_without_age = retrain_without(bundle, reference, PROTECTED)

    production: list[FairnessReport] = []
    without_age: list[FairnessReport] = []
    for name, batch in batches.items():
        production.append(audit(batch, score(batch, bundle), threshold, name, "producao"))
        scores_without = model_without_age.predict_proba(batch[features_without_age])[:, 1]
        without_age.append(audit(batch, scores_without, threshold, name, "sem_idade"))
        logger.info(
            "equidade auditada",
            extra=safe_extra(
                {
                    "batch": name,
                    "disparate_impact_ratio": production[-1].disparate_impact_ratio,
                    "equal_opportunity_gap": production[-1].equal_opportunity_gap,
                    "alerts": production[-1].alerts or ["nenhum"],
                }
            ),
        )

    proxy_auc = proxy_test(reference)
    return {
        "protected_attribute": PROTECTED,
        "age_bands": list(AGE_LABELS),
        "thresholds": {
            "disparate_impact_min": DISPARATE_IMPACT_MIN,
            "equal_opportunity_max_gap": EQUAL_OPPORTUNITY_MAX_GAP,
            "proxy_auc_alert": PROXY_AUC_ALERT,
        },
        "production_model": [r.to_dict() for r in production],
        "model_without_age": [r.to_dict() for r in without_age],
        "proxy_test": {
            "target": f"faixa {YOUNG_BAND}",
            "features": [f for f in FEATURES if f != PROTECTED],
            "roc_auc": proxy_auc,
            "proxy_detected": proxy_auc >= PROXY_AUC_ALERT,
        },
    }


def save_fairness(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Persiste a auditoria em JSON, ao lado dos demais laudos."""
    path = path or settings.reports_dir / "equidade_idade.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    from src.observability.logging_setup import setup_logging

    setup_logging()
    resultado = run_fairness_audit()
    destino = save_fairness(resultado)

    for com, sem in zip(
        resultado["production_model"], resultado["model_without_age"], strict=True
    ):
        logger.info(
            "%-22s AUC %.4f→%.4f | razão impacto %.3f→%.3f | gap bons pagadores %.3f→%.3f | %s",
            com["batch"],
            com["roc_auc"],
            sem["roc_auc"],
            com["disparate_impact_ratio"],
            sem["disparate_impact_ratio"],
            com["equal_opportunity_gap"],
            sem["equal_opportunity_gap"],
            ", ".join(com["alerts"]) or "sem alerta",
        )
    logger.info("teste de proxy: %s", json.dumps(resultado["proxy_test"], ensure_ascii=False))
    logger.info("laudo salvo em %s", destino)
