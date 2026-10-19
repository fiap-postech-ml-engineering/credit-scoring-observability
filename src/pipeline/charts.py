"""Geração dos gráficos SVG do dashboard, sem dependência de JavaScript externo.

Os SVGs são montados à mão em vez de via biblioteca de plot por três razões:
o relatório precisa abrir offline (um arquivo, zero CDN), precisa pesar pouco
o suficiente para versionar, e precisa herdar o tema claro/escuro da página via
CSS custom properties — o que uma imagem PNG não faz.

Convenções de desenho seguidas em todos os gráficos:

* marcas finas, extremidade do dado arredondada em 4px e ancorada na linha de base;
* grade e eixos recessivos (hairline), nunca competindo com o dado;
* rótulos diretos seletivos — nunca um número em cima de cada ponto;
* `<title>` em cada marca, que o navegador transforma em tooltip nativo
  acessível, somado à camada de hover em CSS;
* eixo único por gráfico: duas medidas de escalas diferentes viram dois
  gráficos, nunca dois eixos y.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
from typing import Sequence

# Rampa sequencial azul (100 -> 700) da paleta de referência, usada no heatmap
# de PSI. No tema CLARO, mais escuro = mais drift. No tema ESCURO a rampa é
# invertida (ver `_ramp_css` em dashboard.py): sobre fundo escuro, "mais" tem
# que ficar mais claro, senão o valor alto some na superfície. São passos
# próprios por modo, não um flip automático de cor.
SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]


@dataclass(frozen=True)
class Point:
    """Um ponto de série com rótulo e texto de tooltip."""

    label: str
    value: float
    tooltip: str


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _br(value: float) -> str:
    """Formata inteiro no padrão brasileiro (ponto como separador de milhar)."""
    return f"{int(value):,}".replace(",", ".")


def _dec(value: float, fmt: str = "{:.3f}") -> str:
    """Formata decimal com vírgula, como manda o português."""
    return fmt.format(value).replace(".", ",")


def sequential_step(value: float, vmax: float) -> int:
    """Índice do passo da rampa para `value`; `vmax` fixa a escala do gráfico."""
    if vmax <= 0:
        return 0
    ratio = max(0.0, min(1.0, value / vmax))
    return int(round(ratio * (len(SEQUENTIAL_BLUE) - 1)))


def sequential_color(value: float, vmax: float) -> str:
    """Cor da rampa no tema claro. Para marcas que mudam com o tema, prefira
    `sequential_step` + classe CSS, que permite passos próprios por modo."""
    return SEQUENTIAL_BLUE[sequential_step(value, vmax)]


def psi_heatmap(
    features: Sequence[str],
    batches: Sequence[str],
    matrix: Sequence[Sequence[float]],
    alert_threshold: float,
    warn_threshold: float,
) -> str:
    """Heatmap PSI: uma linha por feature, uma coluna por lote.

    Forma escolhida porque o dado é uma grade de magnitudes — comparar
    "quanto" ao longo de duas dimensões. Cor sequencial de um só matiz, mais
    escuro = mais drift. O valor vai impresso na célula, então a leitura não
    depende de distinguir tons de azul.
    """
    cell_w, cell_h, gap = 112, 34, 2
    label_w, header_h = 296, 46
    width = label_w + len(batches) * (cell_w + gap)
    height = header_h + len(features) * (cell_h + gap) + 8

    vmax = max((max(row) for row in matrix if row), default=0.0)
    vmax = max(vmax, alert_threshold)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Heatmap do Population Stability Index por feature e por lote" '
        f'class="chart chart--heatmap">'
    ]

    for col, batch in enumerate(batches):
        x = label_w + col * (cell_w + gap) + cell_w / 2
        short = batch.split("_", 1)[0]
        suffix = batch.split("_", 1)[1].replace("_", " ") if "_" in batch else ""
        parts.append(
            f'<text x="{x:.1f}" y="18" class="axis-label" text-anchor="middle">{_esc(short)}</text>'
            f'<text x="{x:.1f}" y="32" class="axis-sub" text-anchor="middle">{_esc(suffix)}</text>'
        )

    for row, feature in enumerate(features):
        y = header_h + row * (cell_h + gap)
        parts.append(
            f'<text x="{label_w - 12}" y="{y + cell_h / 2 + 4:.1f}" class="axis-label" '
            f'text-anchor="end">{_esc(feature)}</text>'
        )
        for col, batch in enumerate(batches):
            value = matrix[row][col]
            x = label_w + col * (cell_w + gap)
            step = sequential_step(value, vmax)
            severity = (
                "drift"
                if value >= alert_threshold
                else ("atencao" if value >= warn_threshold else "estavel")
            )
            # A cor do preenchimento e a do texto vêm do CSS (`.psi-N`), não de
            # atributos inline: é o que permite ao heatmap ter uma rampa própria
            # no tema escuro em vez de reaproveitar a clara.
            parts.append(
                f'<g class="cell psi-{step}" data-severity="{severity}">'
                f"<title>{_esc(feature)} — {_esc(batch)}: PSI {_dec(value)} ({severity})</title>"
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" rx="4"/>'
                f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 4:.1f}" '
                f'text-anchor="middle" class="cell-value">{_dec(value)}</text>'
                f"</g>"
            )

    parts.append("</svg>")
    return "".join(parts)


def line_chart(
    points: Sequence[Point],
    baseline: float,
    baseline_label: str,
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
    alert_below: float | None = None,
    value_format: str = "{:.3f}",
) -> str:
    """Linha de uma série contra uma referência horizontal.

    Forma de ênfase: a série é o assunto, a baseline é contexto em cinza
    hairline. Eixo único — nunca se mistura outra medida de escala diferente
    neste mesmo gráfico.
    """
    if not points:
        return '<p class="empty">Sem dados para plotar.</p>'

    width, height = 760, 300
    pad_l, pad_r, pad_t, pad_b = 64, 24, 24, 62
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    values = [p.value for p in points] + [baseline]
    if alert_below is not None:
        values.append(alert_below)
    lo = y_min if y_min is not None else min(values)
    hi = y_max if y_max is not None else max(values)
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.12, hi + span * 0.12
    span = hi - lo

    def x_at(i: int) -> float:
        if len(points) == 1:
            return pad_l + plot_w / 2
        return pad_l + i * plot_w / (len(points) - 1)

    def y_at(v: float) -> float:
        return pad_t + plot_h - (v - lo) / span * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{_esc(y_label)} por lote de produção, comparado com {_esc(baseline_label)}" '
        f'class="chart chart--line">'
    ]

    # Grade recessiva: 4 divisões horizontais com o valor no eixo.
    for step in range(5):
        value = lo + span * step / 4
        y = y_at(value)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{pad_l - 10}" y="{y + 4:.1f}" class="axis-tick" text-anchor="end">'
            f"{_dec(value, value_format)}</text>"
        )

    if alert_below is not None:
        y = y_at(alert_below)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" class="threshold"/>'
            f'<text x="{width - pad_r}" y="{y - 8:.1f}" class="threshold-label" '
            f'text-anchor="end">limiar de alerta</text>'
        )

    y_base = y_at(baseline)
    parts.append(
        f'<line x1="{pad_l}" y1="{y_base:.1f}" x2="{width - pad_r}" y2="{y_base:.1f}" '
        f'class="baseline-ref"/>'
        f'<text x="{width - pad_r}" y="{y_base + 15:.1f}" class="baseline-label" '
        f'text-anchor="end">{_esc(baseline_label)} '
        f"({_dec(baseline, value_format)})</text>"
    )

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x_at(i):.1f},{y_at(p.value):.1f}" for i, p in enumerate(points)
    )
    parts.append(f'<path d="{path}" class="series-line"/>')

    worst_index = min(range(len(points)), key=lambda j: points[j].value)

    for i, point in enumerate(points):
        x, y = x_at(i), y_at(point.value)
        # Anel de 2px na cor da superfície separa marcas que se sobrepõem.
        parts.append(
            f'<g class="marker">'
            f"<title>{_esc(point.tooltip)}</title>"
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" class="marker-halo"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="marker-dot"/>'
            f"</g>"
        )
        label = point.label.split("_", 1)[0]
        parts.append(
            f'<text x="{x:.1f}" y="{height - pad_b + 22:.1f}" class="axis-label" '
            f'text-anchor="middle">{_esc(label)}</text>'
        )
        # Rótulo direto apenas nas extremidades: primeiro, último e o pior ponto.
        # O primeiro ancora à direita do ponto e o último à esquerda, para não
        # esbarrarem nos ticks do eixo y nem na borda do gráfico.
        if i in (0, len(points) - 1, worst_index):
            if i == 0:
                anchor, label_x = "start", x + 11
            elif i == len(points) - 1:
                anchor, label_x = "end", x - 11
            else:
                anchor, label_x = "middle", x
            parts.append(
                f'<text x="{label_x:.1f}" y="{y - 15:.1f}" class="point-value" '
                f'text-anchor="{anchor}">{_dec(point.value, value_format)}</text>'
            )

    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" '
        f'y2="{pad_t + plot_h}" class="axis"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


def horizontal_bars(
    points: Sequence[Point], unit_singular: str = "", unit_plural: str = ""
) -> str:
    """Barras horizontais, cor sequencial pela própria magnitude.

    Horizontal porque os rótulos (regras de contrato) são longos — na vertical
    virariam texto rotacionado, que ninguém lê.
    """
    if not points:
        return '<p class="empty">Nenhuma violação registrada.</p>'

    row_h, gap, label_w = 30, 2, 430
    width = 960
    height = len(points) * (row_h + gap) + 12
    bar_area = width - label_w - 110
    vmax = max(p.value for p in points) or 1.0

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Violações de contrato por regra" class="chart chart--bars">'
    ]

    for i, point in enumerate(points):
        y = i * (row_h + gap)
        bar_w = max(3.0, point.value / vmax * bar_area)
        parts.append(
            f'<g class="bar-row">'
            f"<title>{_esc(point.tooltip)}</title>"
            f'<text x="{label_w - 12}" y="{y + row_h / 2 + 4:.1f}" class="axis-label" '
            f'text-anchor="end">{_esc(point.label)}</text>'
            f'<rect x="{label_w}" y="{y + 6}" width="{bar_w:.1f}" height="{row_h - 12}" '
            f'rx="4" fill="{sequential_color(point.value, vmax)}"/>'
            f'<text x="{label_w + bar_w + 10:.1f}" y="{y + row_h / 2 + 4:.1f}" '
            f'class="bar-value">{_br(point.value)}'
            f"{_esc(unit_singular if point.value == 1 else unit_plural)}</text>"
            f"</g>"
        )

    parts.append(f'<line x1="{label_w}" y1="0" x2="{label_w}" y2="{height - 12}" class="axis"/>')
    parts.append("</svg>")
    return "".join(parts)
