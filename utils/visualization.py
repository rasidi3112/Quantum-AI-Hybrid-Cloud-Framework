"""Visualization helpers for metrics."""

from __future__ import annotations

from typing import Iterable, Mapping

import plotly.graph_objects as go # type: ignore


def create_loss_accuracy_figure(history: Iterable[Mapping[str, float]]) -> go.Figure:
    """Create a Plotly figure showing loss and accuracy curves."""
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_acc = [item["train_accuracy"] for item in history]
    val_acc = [item["val_accuracy"] for item in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers", name="Train Loss"))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="Val Loss"))
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode="lines+markers", name="Train Accuracy", yaxis="y2"))
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines+markers", name="Val Accuracy", yaxis="y2"))

    fig.update_layout(
        title="Training and Validation Metrics",
        xaxis_title="Epoch",
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
    )
    return fig