"""Dashboard HTML consolidado do monitoramento (entregável da Etapa 3).

Os relatórios do Evidently respondem muito bem "o que aconteceu neste lote?",
mas são um arquivo por lote por tipo de relatório — quinze arquivos no total.
Falta a visão de cima: *a saúde do pipeline ao longo do tempo*.

É o que esta página entrega, em um arquivo único, offline, com tema claro e
escuro:

1. **KPIs** — lotes monitorados, lotes com alerta, maior PSI e o prejuízo na
   carteira aprovada;
2. **Heatmap de PSI** — feature × lote, a resposta visual para "quais
   variáveis degradaram";
3. **ROC AUC por lote** contra a baseline de validação;
4. **Inadimplência da carteira aprovada** por lote — a métrica de negócio,
   em gráfico separado porque a escala é outra (eixo único por gráfico);
5. **Alertas por lote**, com ícone e rótulo (a cor nunca carrega sozinha);
6. **Status das tarefas do pipeline**, lote por lote;
7. **Violações do contrato** no lote deliberadamente corrompido;
8. **Tabela completa** — toda a informação dos gráficos também em texto.
"""

from __future__ import annotations

from datetime import UTC, datetime
import html
import json
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from src.config.settings import FEATURES, settings
from src.pipeline.charts import (
    SEQUENTIAL_BLUE,
    Point,
    horizontal_bars,
    line_chart,
    psi_heatmap,
)

if TYPE_CHECKING:  # pragma: no cover
    from src.pipeline.monitor import PipelineRun

logger = logging.getLogger(__name__)

# Ícone + rótulo acompanham toda cor de status: a cor nunca é o único canal.
ALERT_META: dict[str, tuple[str, str, str]] = {
    "contrato_violado": ("critical", "⛔", "Contrato violado"),
    "degradacao_silenciosa": ("critical", "🔇", "Degradação silenciosa"),
    "queda_de_desempenho": ("serious", "📉", "Queda de desempenho"),
    "drift_de_dataset": ("warning", "🌊", "Drift de dataset"),
    "drift_de_feature": ("warning", "📊", "Drift de feature"),
}

STATUS_META: dict[str, tuple[str, str]] = {
    "sucesso": ("good", "✅"),
    "falha": ("critical", "❌"),
    "bloqueada": ("serious", "🚫"),
}


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


# Tradução das regras do Pandera para a linguagem do negócio. O rótulo do
# gráfico precisa dizer *qual coluna* e *qual limite*, senão três regras
# diferentes aparecem como "not_nullable" e ninguém sabe qual é qual.
_RULE_PATTERNS: list[tuple[str, str]] = [
    (r"^not_nullable$", "não pode ser nulo"),
    (r"^greater_than_or_equal_to\((.+)\)$", "≥ {0}"),
    (r"^less_than_or_equal_to\((.+)\)$", "≤ {0}"),
    (r"^isin\((.+)\)$", "só aceita {0}"),
    (r"^column_in_schema$", "coluna fora do contrato"),
    (r"^coerce_dtype\((.+)\)$", "não foi possível converter para {0}"),
    (r"^dtype\((.+)\)$", "tipo deve ser {0}"),
]


def _humanize_rule(rule: str) -> str:
    """Converte `Coluna:check(args)` do Pandera em texto legível."""
    column, _, check = rule.partition(":")
    if not check:
        column, check = "", rule

    readable = check
    for pattern, template in _RULE_PATTERNS:
        match = re.match(pattern, check)
        if match:
            readable = template.format(*match.groups())
            break

    if column in ("", "CreditReferenceContract", "CreditInferenceContract"):
        return readable
    return f"{column} · {readable}"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%".replace(".", ",")


def _num(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _signed(value: float, decimals: int = 4) -> str:
    """Delta sempre com sinal: `+0,0024` lê diferente de `0,0024`."""
    return f"{value:+.{decimals}f}".replace(".", ",")


def _collect(runs: list[PipelineRun]) -> dict[str, Any]:
    """Extrai do histórico de execuções tudo que a página precisa."""
    monitored = [r for r in runs if r.drift is not None]

    # Mapa de PSI por lote, calculado uma vez: o heatmap pede a transposta
    # (uma linha por feature), e refazer `to_dict()` por célula seria O(n²).
    psi_by_batch: list[dict[str, float]] = []
    for run in monitored:
        drift = run.drift
        assert drift is not None  # garantido pelo filtro acima
        psi_by_batch.append(drift.to_dict()["psi_by_feature"])

    psi_matrix: list[list[float]] = [
        [float(psi.get(feature, 0.0)) for psi in psi_by_batch] for feature in FEATURES
    ]

    return {
        "runs": runs,
        "monitored": monitored,
        "batches": [r.batch for r in monitored],
        "psi_matrix": psi_matrix,
    }


def _kpis(data: dict[str, Any], baseline: dict[str, float]) -> list[dict[str, str]]:
    monitored = data["monitored"]
    runs = data["runs"]

    max_psi = max((r.drift.max_psi for r in monitored), default=0.0)
    worst_auc_delta = min(
        (r.drift.performance_delta.get("roc_auc", 0.0) for r in monitored), default=0.0
    )
    worst_bad_rate = max(
        (r.drift.performance.get("approved_bad_rate", 0.0) for r in monitored), default=0.0
    )
    baseline_bad_rate = baseline.get("approved_bad_rate") or 0.0
    multiplier = worst_bad_rate / baseline_bad_rate if baseline_bad_rate else 0.0
    with_alerts = sum(1 for r in runs if r.alerts)

    return [
        {
            "label": "Lotes monitorados",
            "value": str(len(monitored)),
            "detail": f"{len(runs)} execuções do pipeline",
            "status": "",
        },
        {
            "label": "Lotes com alerta",
            "value": str(with_alerts),
            "detail": f"de {len(runs)} lotes processados",
            "status": "critical" if with_alerts else "good",
        },
        {
            "label": "Maior PSI observado",
            "value": _num(max_psi),
            "detail": f"limiar de alerta: {_num(settings.psi_alert_threshold, 2)}",
            "status": "warning" if max_psi >= settings.psi_alert_threshold else "good",
        },
        {
            "label": "Pior queda de ROC AUC",
            "value": _signed(worst_auc_delta),
            "detail": f"limiar: −{_num(settings.auc_degradation_threshold, 2)}",
            "status": (
                "serious" if worst_auc_delta <= -settings.auc_degradation_threshold else "good"
            ),
        },
        {
            "label": "Inadimplência na carteira aprovada",
            "value": _pct(worst_bad_rate),
            "detail": f"{_num(multiplier, 2)}× a baseline ({_pct(baseline_bad_rate)})",
            "status": "critical" if multiplier >= 1.5 else "good",
        },
    ]


def _alert_badges(alerts: list[str]) -> str:
    if not alerts:
        return (
            '<span class="badge badge--good"><span aria-hidden="true">✅</span> Sem alertas</span>'
        )
    chips = []
    for alert in alerts:
        status, icon, label = ALERT_META.get(alert, ("warning", "⚠️", alert))
        chips.append(
            f'<span class="badge badge--{status}"><span aria-hidden="true">{icon}</span> '
            f"{_esc(label)}</span>"
        )
    return "".join(chips)


def _tasks_table(runs: list[PipelineRun]) -> str:
    task_names: list[str] = []
    for run in runs:
        for task in run.tasks:
            if task.name not in task_names:
                task_names.append(task.name)

    head = "".join(f"<th>{_esc(n)}</th>" for n in task_names)
    rows = []
    for run in runs:
        by_name = {t.name: t for t in run.tasks}
        cells = []
        for name in task_names:
            task = by_name.get(name)
            if task is None:
                cells.append('<td class="cell-empty">—</td>')
                continue
            status, icon = STATUS_META.get(task.status, ("warning", "⚠️"))
            cells.append(
                f'<td class="status status--{status}" title="{_esc(task.detail or task.status)}">'
                f'<span aria-hidden="true">{icon}</span> {_esc(task.status)}'
                f'<span class="duration">{_num(task.duration_seconds, 2)}s</span></td>'
            )
        rows.append(f'<tr><th scope="row">{_esc(run.batch)}</th>{"".join(cells)}</tr>')

    return (
        f'<table class="data-table"><thead><tr><th scope="col">Lote</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _summary_table(data: dict[str, Any], baseline: dict[str, float]) -> str:
    rows = []
    for run in data["runs"]:
        drift = run.drift
        if drift is None:
            rows.append(
                f'<tr><th scope="row">{_esc(run.batch)}</th>'
                f"<td>{_esc(run.status)}</td>"
                f'<td class="num">{_num(run.n_rows, 0)}</td>'
                f'<td colspan="5" class="cell-empty">bloqueado no portão de dados</td>'
                f"<td>{_alert_badges(run.alerts)}</td></tr>"
            )
            continue
        performance = drift.performance
        delta = drift.performance_delta
        rows.append(
            f'<tr><th scope="row">{_esc(run.batch)}</th>'
            f"<td>{_esc(run.status)}</td>"
            f'<td class="num">{_num(run.n_rows, 0)}</td>'
            f'<td class="num">{_num(drift.max_psi)}</td>'
            f'<td class="num">{drift.summary.get("n_drifted", 0)}</td>'
            f'<td class="num">{_num(performance.get("roc_auc", float("nan")), 4)}</td>'
            f'<td class="num">{_signed(delta.get("roc_auc", 0.0))}</td>'
            f'<td class="num">{_pct(performance.get("approved_bad_rate", 0.0))}</td>'
            f"<td>{_alert_badges(run.alerts)}</td></tr>"
        )

    return (
        '<table class="data-table"><thead><tr>'
        '<th scope="col">Lote</th><th scope="col">Status</th><th scope="col">Linhas</th>'
        '<th scope="col">PSI máx.</th><th scope="col">Features em drift</th>'
        '<th scope="col">ROC AUC</th><th scope="col">Δ AUC</th>'
        '<th scope="col">Inadimpl. aprovada</th><th scope="col">Alertas</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _contract_bars(reports_dir: Path) -> str:
    path = reports_dir / "lote_corrompido__validation.json"
    if not path.exists():
        return '<p class="empty">Rode o pipeline para gerar o laudo do lote corrompido.</p>'

    laudo = json.loads(path.read_text(encoding="utf-8"))
    by_rule = laudo.get("failures_by_rule") or {}
    points = [
        Point(
            label=_humanize_rule(rule),
            value=float(count),
            tooltip=f"{rule}: {int(count)} ocorrência(s) reprovada(s)",
        )
        for rule, count in sorted(by_rule.items(), key=lambda kv: -kv[1])
    ]
    return horizontal_bars(points, unit_singular=" ocorrência", unit_plural=" ocorrências")


def _ramp_css() -> str:
    """CSS da rampa sequencial do heatmap, com passos próprios por tema.

    Tema **claro**: a rampa vai do azul mais claro ao mais escuro — mais
    escuro significa mais drift — e o texto vira branco na metade escura.

    Tema **escuro**: a rampa é **invertida**. Sobre superfície escura, um
    valor alto pintado de azul-marinho sumiria no fundo; quem precisa saltar
    aos olhos é o drift, então ele recebe os passos claros. Não é um flip
    automático: é a mesma rampa documentada, com os passos escolhidos para a
    superfície em que de fato é renderizada.

    Em ambos os temas o valor numérico vai impresso na célula, então a
    informação nunca depende de distinguir tons.
    """
    ramp = SEQUENTIAL_BLUE
    last = len(ramp) - 1
    dark_from = 7  # a partir deste passo a rampa é escura e pede tinta clara

    def rules(dark_mode: bool, prefix: str = "") -> str:
        out: list[str] = []
        for i in range(len(ramp)):
            if dark_mode:
                color = ramp[last - i]
                ink = "#ffffff" if i < (last - dark_from) else "#0b0b0b"
            else:
                color = ramp[i]
                ink = "#0b0b0b" if i < dark_from else "#ffffff"
            out.append(f"{prefix}.psi-{i} rect {{ fill: {color}; }}")
            out.append(f"{prefix}.psi-{i} .cell-value {{ fill: {ink}; }}")
        return "\n".join(out)

    # Mesma disciplina de escopo do resto da folha: a media query cobre a
    # preferência do sistema, e o seletor data-theme vence nos dois sentidos.
    media_scope = ':root:not([data-theme="light"]) '
    toggle_scope = ':root[data-theme="dark"] '

    return (
        rules(dark_mode=False)
        + "\n@media (prefers-color-scheme: dark) {\n"
        + rules(dark_mode=True, prefix=media_scope)
        + "\n}\n"
        + rules(dark_mode=True, prefix=toggle_scope)
        + "\n"
    )


_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6;
  --good: #0ca30c;
  --warning: #fab219;
  --serious: #ec835a;
  --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d;
  --surface-1: #1a1a19;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #898781;
  --grid: #2c2c2a;
  --axis: #383835;
  --border: rgba(255, 255, 255, 0.10);
  --series-1: #3987e5;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0 16px 64px;
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.5;
}
.wrap { max-width: 1120px; margin: 0 auto; }
header { padding: 40px 0 8px; }
h1 { font-size: clamp(1.5rem, 4vw, 2rem); margin: 0 0 8px; letter-spacing: -0.02em; }
.subtitle { color: var(--text-secondary); margin: 0 0 4px; max-width: 70ch; }
.meta { color: var(--text-muted); font-size: 0.8125rem; margin: 8px 0 0; }

section {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  overflow-x: auto;
}
h2 { font-size: 1.0625rem; margin: 0 0 4px; }
.hint { color: var(--text-secondary); font-size: 0.875rem; margin: 0 0 20px; max-width: 78ch; }

.kpis {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px;
  margin: 24px 0;
}
.kpi {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px;
}
.kpi-label { color: var(--text-secondary); font-size: 0.8125rem; margin: 0 0 6px; }
.kpi-value { font-size: 1.875rem; font-weight: 600; letter-spacing: -0.02em; }
.kpi-detail { color: var(--text-muted); font-size: 0.75rem; margin: 4px 0 0; }
.kpi[data-status="good"] { border-left: 3px solid var(--good); }
.kpi[data-status="warning"] { border-left: 3px solid var(--warning); }
.kpi[data-status="serious"] { border-left: 3px solid var(--serious); }
.kpi[data-status="critical"] { border-left: 3px solid var(--critical); }

.chart { display: block; min-width: 680px; }
.chart--bars { min-width: 820px; }
.axis-label { font-size: 11px; fill: var(--text-secondary); }
.axis-sub { font-size: 10px; fill: var(--text-muted); }
.axis-tick { font-size: 10px; fill: var(--text-muted); font-variant-numeric: tabular-nums; }
.cell-value { font-size: 11px; font-variant-numeric: tabular-nums; }
.bar-value { font-size: 11px; fill: var(--text-secondary); font-variant-numeric: tabular-nums; }
.point-value { font-size: 11px; fill: var(--text-primary); font-weight: 600; font-variant-numeric: tabular-nums; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.threshold { stroke: var(--critical); stroke-width: 1.5; stroke-dasharray: 5 4; }
.threshold-label { font-size: 10px; fill: var(--critical); }
.baseline-ref { stroke: var(--axis); stroke-width: 1.5; stroke-dasharray: 2 3; }
.baseline-label { font-size: 10px; fill: var(--text-muted); }
.series-line { fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.marker-halo { fill: var(--surface-1); }
.marker-dot { fill: var(--series-1); }
.cell, .marker, .bar-row { cursor: default; }
.cell:hover rect, .bar-row:hover rect { stroke: var(--text-primary); stroke-width: 2; }
.marker:hover .marker-dot { r: 7; }

.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0 0; font-size: 0.8125rem; color: var(--text-secondary); }
.legend-item { display: inline-flex; align-items: center; gap: 7px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.ramp { display: inline-flex; height: 12px; border-radius: 3px; overflow: hidden; }
.ramp-chip { width: 18px; height: 12px; display: block; }

.badge {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.75rem; padding: 3px 9px; border-radius: 999px;
  border: 1px solid var(--border); white-space: nowrap; margin: 2px 3px 2px 0;
}
.badge--good { background: color-mix(in srgb, var(--good) 14%, transparent); }
.badge--warning { background: color-mix(in srgb, var(--warning) 20%, transparent); }
.badge--serious { background: color-mix(in srgb, var(--serious) 20%, transparent); }
.badge--critical { background: color-mix(in srgb, var(--critical) 18%, transparent); }

.data-table { width: 100%; border-collapse: collapse; font-size: 0.8125rem; min-width: 760px; }
.data-table th, .data-table td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }
.data-table thead th { color: var(--text-secondary); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
.data-table tbody th { font-weight: 600; white-space: nowrap; }
.data-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.cell-empty { color: var(--text-muted); }
.status { white-space: nowrap; }
.status .duration { color: var(--text-muted); font-size: 0.6875rem; margin-left: 6px; font-variant-numeric: tabular-nums; }
.status--good { color: var(--good); }
.status--critical { color: var(--critical); }
.status--serious { color: var(--serious); }
.empty { color: var(--text-muted); font-size: 0.875rem; margin: 0; }
.links { display: flex; flex-wrap: wrap; gap: 10px; margin: 0; padding: 0; list-style: none; }
.links a { font-size: 0.8125rem; color: var(--series-1); text-decoration: none; border: 1px solid var(--border); border-radius: 8px; padding: 6px 11px; }
.links a:hover { text-decoration: underline; }
.callout { border-left: 3px solid var(--critical); padding: 4px 0 4px 16px; margin: 20px 0 0; color: var(--text-secondary); font-size: 0.875rem; }
.callout strong { color: var(--text-primary); }
@media print { section { break-inside: avoid; } }
"""


def render_dashboard(
    runs: list[PipelineRun],
    output_path: Path | None = None,
    baseline: dict[str, float] | None = None,
    candidate: str | None = None,
) -> Path:
    """Renderiza o dashboard consolidado e devolve o caminho do HTML."""
    output_path = output_path or settings.reports_dir / "dashboard.html"

    if baseline is None or candidate is None:
        from src.credit_scoring.train import load_model

        try:
            bundle = load_model()
            baseline = baseline if baseline is not None else bundle.get("validation_metrics", {})
            candidate = candidate or str(bundle.get("candidate", "baseline"))
        except FileNotFoundError:
            baseline = baseline or {}
            candidate = candidate or "indisponível"

    data = _collect(runs)
    monitored = data["monitored"]

    kpi_html = "".join(
        f'<div class="kpi" data-status="{_esc(k["status"])}">'
        f'<p class="kpi-label">{_esc(k["label"])}</p>'
        f'<div class="kpi-value">{_esc(k["value"])}</div>'
        f'<p class="kpi-detail">{_esc(k["detail"])}</p></div>'
        for k in _kpis(data, baseline)
    )

    heatmap = (
        psi_heatmap(
            FEATURES,
            data["batches"],
            data["psi_matrix"],
            settings.psi_alert_threshold,
            settings.psi_warn_threshold,
        )
        if monitored
        else '<p class="empty">Nenhum lote monitorado.</p>'
    )

    auc_points = [
        Point(
            label=r.batch,
            value=float(r.drift.performance.get("roc_auc", 0.0)),
            tooltip=(
                f"{r.batch}: ROC AUC {_num(r.drift.performance.get('roc_auc', 0.0), 4)} "
                f"(Δ {_signed(r.drift.performance_delta.get('roc_auc', 0.0))})"
            ),
        )
        for r in monitored
    ]
    baseline_auc = float(baseline.get("roc_auc", 0.0))
    auc_chart = line_chart(
        auc_points,
        baseline=baseline_auc,
        baseline_label="baseline de validação",
        y_label="ROC AUC",
        alert_below=baseline_auc - settings.auc_degradation_threshold,
        value_format="{:.3f}",
    )

    bad_rate_points = [
        Point(
            label=r.batch,
            value=float(r.drift.performance.get("approved_bad_rate", 0.0)),
            tooltip=(
                f"{r.batch}: {_pct(r.drift.performance.get('approved_bad_rate', 0.0))} "
                f"de inadimplência entre os aprovados"
            ),
        )
        for r in monitored
    ]
    bad_rate_chart = line_chart(
        bad_rate_points,
        baseline=float(baseline.get("approved_bad_rate", 0.0)),
        baseline_label="baseline de validação",
        y_label="Inadimplência na carteira aprovada",
        value_format="{:.3f}",
    )

    # A legenda reusa as classes da rampa para acompanhar o tema junto com
    # o heatmap, em vez de congelar as cores do tema claro.
    ramp_html = "".join(
        f'<svg class="ramp-chip psi-{step}" viewBox="0 0 1 1" aria-hidden="true">'
        f'<rect width="1" height="1"/></svg>'
        for step in (0, 2, 4, 6, 8, 10, 12)
    )

    def _drifted_cell(run: PipelineRun) -> str:
        if run.drift is None:
            return '<td class="cell-empty">— (lote bloqueado)</td>'
        features = run.drift.drifted_features
        if not features:
            return '<td class="cell-empty">nenhuma</td>'
        return f"<td>{_esc(', '.join(features))}</td>"

    alert_rows = "".join(
        f'<tr><th scope="row">{_esc(r.batch)}</th>'
        f"{_drifted_cell(r)}"
        f"<td>{_alert_badges(r.alerts)}</td></tr>"
        for r in data["runs"]
    )

    silent = [r for r in monitored if r.drift.silent_degradation]
    callout = (
        f'<p class="callout"><strong>Degradação silenciosa detectada em '
        f"{_esc(', '.join(r.batch for r in silent))}.</strong> O desempenho caiu além do "
        f"limiar e <em>nenhuma</em> feature acusou drift — assinatura de concept drift. "
        f"Monitoramento só de distribuição de dados não teria visto nada.</p>"
        if silent
        else ""
    )

    evidently_links = "".join(
        f'<li><a href="{_esc(path.name)}">{_esc(r.batch)} — {_esc(kind.replace("_", " "))}</a></li>'
        for r in monitored
        for kind, path in r.drift.html_reports.items()
    )

    generated_at = datetime.now(UTC).strftime("%d/%m/%Y %H:%M UTC")

    page = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Observabilidade de Credit Scoring</title>
<style>{_CSS}\n{_ramp_css()}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Observabilidade de Credit Scoring</h1>
  <p class="subtitle">Saúde do contrato de dados, do pipeline e do modelo ao longo de
  cinco lotes mensais de produção — do mês estável à recessão.</p>
  <p class="meta">Gerado em {generated_at} · modelo <code>{_esc(candidate)}</code>
  · baseline ROC AUC {_num(baseline_auc, 4)} · limiar de PSI {_num(settings.psi_alert_threshold, 2)}
  · limiar de queda de AUC {_num(settings.auc_degradation_threshold, 2)}</p>
</header>

<div class="kpis">{kpi_html}</div>

{callout}

<section>
  <h2>Drift por feature e por lote (PSI)</h2>
  <p class="hint">Quanto mais a célula se destaca do fundo, maior o drift. O valor vai
  impresso em cada uma, então a leitura nunca depende de distinguir tons. Compare a última
  coluna com as duas anteriores: na recessão o PSI praticamente não se move — e é
  justamente o lote mais caro.</p>
  {heatmap}
  <div class="legend">
    <span class="legend-item">PSI menor <span class="ramp">{ramp_html}</span> maior</span>
    <span class="legend-item">estável &lt; {_num(settings.psi_warn_threshold, 2)}</span>
    <span class="legend-item">atenção {_num(settings.psi_warn_threshold, 2)}–{_num(settings.psi_alert_threshold, 2)}</span>
    <span class="legend-item">drift &gt; {_num(settings.psi_alert_threshold, 2)}</span>
  </div>
</section>

<section>
  <h2>Poder discriminante do modelo (ROC AUC)</h2>
  <p class="hint">Uma série — o AUC medido em cada lote de produção. A linha pontilhada
  cinza é a baseline de validação; a vermelha é o limiar de alerta
  (−{_num(settings.auc_degradation_threshold, 2)}).</p>
  {auc_chart}
</section>

<section>
  <h2>Inadimplência na carteira aprovada</h2>
  <p class="hint">A métrica de negócio: entre os clientes que o modelo <em>aprovou</em>,
  quantos deixaram de pagar. Gráfico separado do AUC de propósito — escalas diferentes
  nunca compartilham eixo.</p>
  {bad_rate_chart}
</section>

<section>
  <h2>Alertas por lote</h2>
  <p class="hint">Cada alerta traz ícone e rótulo; a cor é reforço, nunca o único canal
  de informação.</p>
  <table class="data-table">
    <thead><tr><th scope="col">Lote</th><th scope="col">Features em drift</th>
    <th scope="col">Alertas</th></tr></thead>
    <tbody>{alert_rows}</tbody>
  </table>
</section>

<section>
  <h2>Status das tarefas do pipeline</h2>
  <p class="hint">Status e duração de cada tarefa, lote por lote. Tarefa
  <em>bloqueada</em> significa que o portão de dados barrou o lote antes de o modelo vê-lo.</p>
  {_tasks_table(data["runs"])}
</section>

<section>
  <h2>Contrato de dados bloqueando o lote corrompido</h2>
  <p class="hint">Ocorrências reprovadas por regra do contrato, no lote em que injetamos
  11 defeitos de propósito. Nenhuma linha deste lote chegou ao modelo: a tarefa de
  inferência foi bloqueada no portão. Barras horizontais porque os nomes das regras
  são longos.</p>
  {_contract_bars(settings.reports_dir)}
</section>

<section>
  <h2>Resumo completo</h2>
  <p class="hint">Toda a informação dos gráficos acima, também em texto — para leitores
  de tela, impressão e conferência numérica.</p>
  {_summary_table(data, baseline)}
</section>

<section>
  <h2>Relatórios detalhados do Evidently</h2>
  <p class="hint">Um relatório por lote por tipo de análise, com os testes estatísticos
  completos e a distribuição de cada feature.</p>
  <ul class="links">{evidently_links or '<li class="empty">Nenhum relatório gerado.</li>'}</ul>
</section>
</div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    logger.info("dashboard renderizado", extra={"path": str(output_path)})
    return output_path
