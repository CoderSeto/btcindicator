"""
Plotly chart construction for the BTC indicator dashboard.
Kept separate from the Streamlit app (app.py) so it can be exercised
with synthetic data and no live Streamlit session.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_figure(df, cfg):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.04,
        subplot_titles=("Price", "RSI", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["time"], y=df[f"sma_{cfg.SMA_FAST}"], name=f"SMA {cfg.SMA_FAST}",
        line=dict(width=1, color="#2ca02c"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["time"], y=df[f"sma_{cfg.SMA_SLOW}"], name=f"SMA {cfg.SMA_SLOW}",
        line=dict(width=1, color="#d62728"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["bb_upper"], name="BB Upper",
        line=dict(width=1, dash="dot", color="gray"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["bb_lower"], name="BB Lower",
        line=dict(width=1, dash="dot", color="gray"),
        fill="tonexty", fillcolor="rgba(128,128,128,0.08)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["time"], y=df["rsi"], name="RSI", line=dict(color="#9467bd"),
    ), row=2, col=1)
    fig.add_hline(y=cfg.RSI_OVERBOUGHT, line_dash="dash", line_color="#d62728", row=2, col=1)
    fig.add_hline(y=cfg.RSI_OVERSOLD, line_dash="dash", line_color="#2ca02c", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    hist_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(
        x=df["time"], y=df["macd_hist"], name="MACD Hist", marker_color=hist_colors,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["macd"], name="MACD", line=dict(color="#1f77b4")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["macd_signal"], name="Signal", line=dict(color="#ff7f0e")), row=3, col=1)

    fig.update_layout(
        height=750,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=20),
    )
    return fig
