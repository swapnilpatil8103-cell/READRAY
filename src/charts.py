"""Plotly charts used where they genuinely help. Every data chart is labelled with its provenance."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .pdf_processor import PaperDocument, SearchHit

PROVENANCE = {
    "paper": ("Data extracted from the paper", "#2f6f4e"),
    "user": ("Data supplied by you", "#3b5b92"),
    "illustrative": ("Illustrative / simulated data — NOT from the paper", "#a0522d"),
}

PALETTE = ["#3b5b92", "#c8803a", "#2f6f4e", "#8c4a7d", "#5f8fa8", "#b5574b", "#7a7a3a", "#4d4d4d"]
BASE_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=40, r=16, t=56, b=40),
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    hoverlabel=dict(font_size=12),
)


def paper_map(paper: PaperDocument) -> go.Figure:
    """Text volume per page with figure/table/equation markers. Click a bar to jump to that page."""
    pages = [p.number for p in paper.pages]
    chars = [len(p.text) for p in paper.pages]
    sections = []
    for p in paper.pages:
        sections.append(paper.section_for(p.number) or "—")
    fig = go.Figure()
    fig.add_bar(
        x=pages, y=chars, name="Extracted text (characters)",
        marker_color=["#d9a35b" if p.image_only else "#9fb3d1" for p in paper.pages],
        customdata=sections, hovertemplate="Page %{x}<br>%{y:,} characters<br>Section: %{customdata}<extra></extra>",
    )
    for kind, symbol, color in (("figure", "square", "#2f6f4e"), ("table", "diamond", "#8c4a7d")):
        xs = [c.page for c in paper.captions if c.kind == kind]
        if xs:
            fig.add_scatter(x=xs, y=[max(chars or [1]) * 1.05] * len(xs), mode="markers", name=f"{kind.title()} caption",
                            marker=dict(symbol=symbol, size=10, color=color), hovertemplate=f"{kind.title()} on page %{{x}}<extra></extra>")
    eq_pages = [e.page for e in paper.equations]
    if eq_pages:
        fig.add_scatter(x=eq_pages, y=[max(chars or [1]) * 1.12] * len(eq_pages), mode="markers", name="Equation (detected)",
                        marker=dict(symbol="circle-open", size=9, color="#b5574b"), hovertemplate="Equation on page %{x}<extra></extra>")
    fig.update_layout(
        **BASE_LAYOUT, height=260, title=dict(text="Paper map — click a bar to open that page", font_size=14),
        xaxis=dict(title="Page", dtick=1 if paper.page_count <= 30 else None), yaxis=dict(title="Characters"),
        legend=dict(orientation="h", y=-0.3), bargap=0.15, clickmode="event+select",
    )
    return fig


def search_hits_chart(hits: list[SearchHit], page_count: int, query: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[h.page for h in hits], y=[h.count for h in hits], marker_color="#c8803a",
        hovertemplate="Page %{x}: %{y} match(es)<extra></extra>",
    ))
    fig.update_layout(**BASE_LAYOUT, height=200, title=dict(text=f"Where “{query}” appears", font_size=13),
                      xaxis=dict(title="Page", range=[0.5, page_count + 0.5]), yaxis=dict(title="Matches"))
    return fig


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns that look numeric (e.g. '1,234', '12.5%', '(0.03)', '0.21***') to numbers."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        cleaned = (
            out[col].astype(str).str.strip()
            .str.replace(r"[,%$€£\s]", "", regex=True)
            .str.replace(r"\*+$", "", regex=True)
            .str.replace(r"^\((-?[\d.]+)\)$", r"\1", regex=True)
            .str.replace("−", "-", regex=False)
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().sum() >= max(1, int(0.6 * out[col].replace("", pd.NA).notna().sum())):
            out[col] = numeric
    return out


def data_chart(df: pd.DataFrame, x: str, ys: list[str], kind: str, provenance: str, source_note: str = "") -> go.Figure:
    """Bar / line / scatter chart with a mandatory provenance banner in the title."""
    if provenance not in PROVENANCE:
        raise ValueError("provenance must be 'paper', 'user' or 'illustrative'")
    if not ys:
        raise ValueError("Choose at least one numeric column to plot.")
    label, color = PROVENANCE[provenance]
    fig = go.Figure()
    for i, y in enumerate(ys):
        style = dict(name=str(y), marker_color=PALETTE[i % len(PALETTE)])
        if kind == "line":
            fig.add_scatter(x=df[x], y=df[y], mode="lines+markers", **style)
        elif kind == "scatter":
            fig.add_scatter(x=df[x], y=df[y], mode="markers", **style)
        else:
            fig.add_bar(x=df[x], y=df[y], **style)
    title = f"<span style='color:{color}'><b>{label}</b></span>"
    if source_note:
        title += f"<br><span style='font-size:11px;color:#666'>{source_note}</span>"
    fig.update_layout(**BASE_LAYOUT, height=380, title=dict(text=title, font_size=13),
                      xaxis_title=str(x), yaxis_title=", ".join(map(str, ys)) if len(ys) <= 2 else "Value",
                      barmode="group", legend=dict(orientation="h", y=-0.25))
    if provenance == "illustrative":
        fig.add_annotation(text="ILLUSTRATIVE", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=46, color="rgba(160,82,45,0.12)"), textangle=-20)
    return fig


def matrix_coverage_chart(topic: dict, fields: list[tuple[str, str]]) -> go.Figure:
    """Heatmap of which literature-review cells are empty, AI-drafted, auto-detected or manual."""
    code = {"": 0, "metadata": 1, "ai": 2, "manual": 3}
    titles, z, text = [], [], []
    for pid in topic["paper_ids"]:
        row = topic["rows"].get(pid, {})
        title = row.get("title", {}).get("value") or pid
        titles.append(title[:40] + ("…" if len(title) > 40 else ""))
        z.append([code.get(row.get(k, {}).get("source", ""), 0) if row.get(k, {}).get("value") else 0 for k, _ in fields])
        text.append([{0: "empty", 1: "auto-detected", 2: "AI draft", 3: "manual"}[v] for v in z[-1]])
    colorscale = [[0, "#f1f1f1"], [0.25, "#f1f1f1"], [0.25, "#cfd8e6"], [0.5, "#cfd8e6"],
                  [0.5, "#f0c987"], [0.75, "#f0c987"], [0.75, "#7fb08f"], [1, "#7fb08f"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=[label for _, label in fields], y=titles, text=text, zmin=0, zmax=3, colorscale=colorscale,
        showscale=False, xgap=2, ygap=2, hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(**BASE_LAYOUT, height=120 + 34 * max(1, len(titles)),
                      title=dict(text="Matrix coverage — grey: empty · blue: auto-detected · amber: AI draft · green: manual", font_size=12))
    fig.update_yaxes(autorange="reversed")
    return fig
