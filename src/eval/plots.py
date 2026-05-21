"""WP6 — Presentation-ready chart builders.

Each `build_*` function takes a `mode -> aggregate-dict` mapping (the
shape produced by `BenchmarkRunner._aggregate()`, i.e. what
`evals/results/aggregate_<mode>_<ts>.json` deserialises into) and returns
a Plotly `Figure`. No file I/O happens here — that lives in
`visualize.py`.

The style is tuned for academic slides: white background, large fonts,
no busy legends, and a subtitle baked into the title block calling out
that `claude_only_stub` is a *simulated upper-bound baseline* rather
than a live system. Titles are multi-line because paper-coordinate
annotations get clipped on certain layouts (notably polar / log-scale).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

MODE_DISPLAY: Dict[str, str] = {
    "clip_only": "CLIP-only",
    "dual_agent": "Dual-Agent",
    "claude_only_stub": "Claude-only (simulated)",
}

# Stable order for axes / legends — produces consistent ordering even when
# some modes are missing.
MODE_ORDER: Tuple[str, ...] = ("clip_only", "dual_agent", "claude_only_stub")

MODE_COLOR: Dict[str, str] = {
    "clip_only": "#3B82F6",        # blue   — retriever-only baseline
    "dual_agent": "#10B981",       # green  — production cascade
    "claude_only_stub": "#9CA3AF", # grey   — simulated upper bound
}

# Distinct hues for grouped-bar series so the chart reads at a glance.
SERIES_COLORS: Tuple[str, ...] = ("#1F2937", "#6366F1", "#F59E0B", "#EF4444")

UPPER_BOUND_DISCLAIMER = (
    "claude_only_stub is a simulated upper-bound baseline — not a live system."
)


def _title_block(title: str, subtitle: str, *, extra: str | None = None) -> str:
    """Build a multi-line title (HTML) with subtitle + disclaimer baked in.

    Baking the subtitle into the title is more robust than free-floating
    paper-coordinate annotations, which Plotly sometimes clips on polar
    layouts and log-scale axes.
    """
    parts = [
        f"<span style='font-size:22px;color:#111827'>{title}</span>",
        f"<span style='font-size:13px;color:#374151'>{subtitle}</span>",
    ]
    if extra:
        parts.append(f"<span style='font-size:12px;color:#374151'>{extra}</span>")
    parts.append(
        f"<span style='font-size:11px;color:#6B7280;font-style:italic'>"
        f"{UPPER_BOUND_DISCLAIMER}</span>"
    )
    return "<br>".join(parts)


def _base_layout(*, top_margin: int = 160) -> dict:
    """Layout common to every chart, parameterised on top margin so the
    radar chart can leave more vertical room for its multi-line title."""
    return dict(
        template="simple_white",
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=15, color="#111827"),
        title=dict(
            font=dict(size=22, color="#111827"),
            x=0.02, xanchor="left", y=0.97, yanchor="top",
        ),
        margin=dict(l=80, r=40, t=top_margin, b=120),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.18, x=0.0, font=dict(size=13)),
        width=1280,
        height=720,
    )


def _ordered_modes(results: Mapping[str, dict]) -> List[str]:
    """Return modes present in `results`, in canonical display order."""
    return [m for m in MODE_ORDER if m in results]


# ---------------------------------------------------------------------------
# Bar helpers
# ---------------------------------------------------------------------------

def _grouped_bar(
    modes: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float]]],
    *,
    title: str,
    subtitle: str,
    yaxis_title: str,
    yaxis_range: Tuple[float, float] | None = None,
    yaxis_tickformat: str | None = None,
    value_format: str = ".3f",
    extra_subtitle: str | None = None,
) -> go.Figure:
    """Build a grouped-bar chart: one cluster per mode, one bar per series."""
    fig = go.Figure()
    display_x = [MODE_DISPLAY[m] for m in modes]
    for i, (series_name, values) in enumerate(series):
        fig.add_bar(
            name=series_name,
            x=display_x,
            y=list(values),
            marker_color=SERIES_COLORS[i % len(SERIES_COLORS)],
            text=[format(v, value_format) for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    fig.update_layout(
        **_base_layout(),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        title_text=_title_block(title, subtitle, extra=extra_subtitle),
        yaxis_title=yaxis_title,
    )
    if yaxis_range is not None:
        fig.update_yaxes(range=list(yaxis_range))
    if yaxis_tickformat is not None:
        fig.update_yaxes(tickformat=yaxis_tickformat)
    fig.update_xaxes(title_text="System mode")
    return fig


def _cost_per_query(aggregate: dict) -> float:
    total = float(aggregate["cost_metrics"]["computed_cost_usd"])
    n = max(int(aggregate["metadata"]["total_queries"]), 1)
    return total / n


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def build_quality_chart(results: Mapping[str, dict]) -> go.Figure:
    """Recall@5 / F1@5 / MRR / nDCG@5 grouped by mode. Y ∈ [0, 1]."""
    modes = _ordered_modes(results)
    metric_keys = [
        ("Recall@5", "recall_at_5"),
        ("F1@5", "f1_at_5"),
        ("MRR", "mrr"),
        ("nDCG@5", "ndcg_at_5"),
    ]
    series: List[Tuple[str, List[float]]] = []
    for label, key in metric_keys:
        series.append(
            (label, [float(results[m]["retrieval_metrics"][key]) for m in modes])
        )
    return _grouped_bar(
        modes,
        series,
        title="Retrieval Quality by Mode",
        subtitle="Ranking / retrieval metrics — how well the system orders candidate frames, not final-answer correctness.",
        yaxis_title="Score (0–1, higher is better)",
        yaxis_range=(0.0, 1.1),
    )


def build_privacy_chart(results: Mapping[str, dict]) -> go.Figure:
    """Fraction of queries and unique frames kept on-premise, per mode."""
    modes = _ordered_modes(results)
    queries = [float(results[m]["privacy_metrics"]["fraction_queries_on_prem"]) for m in modes]
    frames = [float(results[m]["privacy_metrics"]["fraction_frames_on_prem"]) for m in modes]
    return _grouped_bar(
        modes,
        [("Queries on-prem", queries), ("Unique frames on-prem", frames)],
        title="Privacy Preservation by Mode",
        subtitle="Higher is better. Frames-on-prem counts unique corpus frames never sent to the cloud across the benchmark.",
        yaxis_title="Fraction on-prem",
        yaxis_range=(0.0, 1.1),
        yaxis_tickformat=".0%",
        # 4 decimals so dual_agent's 0.9999 doesn't round up to 1.0000 and
        # hide the headline that only a handful of frames left the enterprise.
        value_format=".4f",
    )


def build_cost_chart(results: Mapping[str, dict]) -> go.Figure:
    """Cost per query in USD. Linear y-axis with text labels does a better
    job at a-glance comparison than a log axis (which makes $0 ambiguous).
    The order-of-magnitude difference is communicated in the labels."""
    modes = _ordered_modes(results)
    values = [_cost_per_query(results[m]) for m in modes]
    headline = max(values) if values else 0.0
    fig = go.Figure()
    fig.add_bar(
        x=[MODE_DISPLAY[m] for m in modes],
        y=values,
        marker_color=[MODE_COLOR[m] for m in modes],
        text=[f"${v:.4f}" for v in values],
        textposition="outside",
        cliponaxis=False,
        showlegend=False,
    )
    # Annotate the cascade's order-of-magnitude advantage in the title row.
    cascade_cost = next(
        (v for m, v in zip(modes, values) if m == "dual_agent"),
        None,
    )
    advantage_line = None
    if cascade_cost is not None and cascade_cost > 0 and headline > cascade_cost:
        ratio = headline / cascade_cost
        advantage_line = (
            f"Dual-Agent costs ~{ratio:,.0f}× less per query than the simulated upper bound."
        )
    fig.update_layout(
        **_base_layout(),
        title_text=_title_block(
            "Estimated Cost per Query",
            "USD per query — Claude Haiku 4.5 list pricing: 1.00 / 1M input, 5.00 / 1M output tokens.",
            extra=advantage_line,
        ),
        yaxis_title="USD per query",
    )
    fig.update_yaxes(
        tickprefix="$",
        tickformat=".4f",
        range=[0, headline * 1.18 if headline > 0 else 1.0],
    )
    fig.update_xaxes(title_text="System mode")
    return fig


def build_latency_chart(results: Mapping[str, dict]) -> go.Figure:
    """End-to-end mean vs. reasoner mean latency, by mode."""
    modes = _ordered_modes(results)
    total = [float(results[m]["latency_metrics"]["total"]["mean_ms"]) for m in modes]
    reasoner = [float(results[m]["latency_metrics"]["reasoner"]["mean_ms"]) for m in modes]
    return _grouped_bar(
        modes,
        [("End-to-end mean (ms)", total), ("Reasoner mean (ms)", reasoner)],
        title="Latency Breakdown by Mode",
        subtitle="Lower is better. End-to-end mean is the wall-clock time per query; reasoner mean is the Claude call alone.",
        yaxis_title="Latency (ms)",
        value_format=".0f",
    )


def build_scorecard_chart(results: Mapping[str, dict]) -> go.Figure:
    """Normalized 4-axis radar: Quality, Privacy, Cost efficiency, Speed.

    Each axis is normalised to [0, 1] across the included modes so the
    chart is a *relative* comparison, not an absolute ranking. Quality
    uses nDCG@5; privacy uses fraction-of-frames-on-prem; cost and speed
    are inverted (1 − x/max) so larger is always better.

    Traces are drawn with a strong outline and a low-alpha fill so
    overlapping modes remain distinguishable (e.g. when CLIP-only and
    Dual-Agent share retrieval quality but diverge on cost/latency).
    """
    modes = _ordered_modes(results)
    quality = [float(results[m]["retrieval_metrics"]["ndcg_at_5"]) for m in modes]
    privacy = [float(results[m]["privacy_metrics"]["fraction_frames_on_prem"]) for m in modes]
    cost_pq = [_cost_per_query(results[m]) for m in modes]
    latency = [float(results[m]["latency_metrics"]["total"]["mean_ms"]) for m in modes]

    def _invert_normalize(values: Sequence[float]) -> List[float]:
        peak = max(values) if values else 0.0
        if peak <= 0:
            return [1.0 for _ in values]
        return [1.0 - (v / peak) for v in values]

    cost_eff = _invert_normalize(cost_pq)
    speed = _invert_normalize(latency)
    axes = ["Quality (nDCG@5)", "Privacy (frames on-prem)", "Cost efficiency", "Speed"]

    def _rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha:.2f})"

    fig = go.Figure()
    for i, m in enumerate(modes):
        vals = [quality[i], privacy[i], cost_eff[i], speed[i]]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=axes + [axes[0]],
            name=MODE_DISPLAY[m],
            line=dict(color=MODE_COLOR[m], width=3),
            fill="toself",
            fillcolor=_rgba(MODE_COLOR[m], 0.22),
        ))

    layout = _base_layout(top_margin=200)
    fig.update_layout(
        **layout,
        title_text=_title_block(
            "Simulated Upper Bound vs Practical Cascade",
            "Each axis normalised relative to the modes shown; cost / latency inverted so larger is better.",
            extra="A larger filled area = stronger overall trade-off across quality, privacy, cost, and speed.",
        ),
        polar=dict(
            bgcolor="white",
            domain=dict(x=[0.15, 0.85], y=[0.05, 0.78]),
            radialaxis=dict(range=[0, 1], tickformat=".1f", gridcolor="#E5E7EB", showline=False),
            angularaxis=dict(gridcolor="#E5E7EB"),
        ),
        showlegend=True,
    )
    return fig
