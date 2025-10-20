"""Streamlit dashboard for monitoring hybrid model training and quantum layers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from deployment.inference import HybridInferenceService
from utils.logging_utils import load_metrics_file


def load_metrics(metrics_path: Path) -> List[Dict[str, float]]:
    """Load metrics from JSON file."""
    if not metrics_path.exists():
        st.warning(f"Metrics file {metrics_path} not found.")
        return []
    return load_metrics_file(metrics_path)


def bloch_sphere_plot(theta: float, phi: float) -> go.Figure:
    """Create a Bloch sphere plot for given angles."""
    sphere = go.Figure(
        data=[
            go.Surface(
                x=np.outer(np.cos(np.linspace(0, 2 * np.pi, 50)), np.sin(np.linspace(0, np.pi, 50))),
                y=np.outer(np.sin(np.linspace(0, 2 * np.pi, 50)), np.sin(np.linspace(0, np.pi, 50))),
                z=np.outer(np.ones(50), np.cos(np.linspace(0, np.pi, 50))),
                colorscale="Blues",
                opacity=0.3,
                showscale=False,
            )
        ]
    )
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    sphere.add_trace(
        go.Cone(
            x=[0], y=[0], z=[0],
            u=[x], v=[y], w=[z],
            sizemode="absolute",
            sizeref=0.3,
        )
    )
    sphere.update_layout(
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return sphere


def main() -> None:
    """Run the Streamlit dashboard."""
    st.title("Quantum-AI Hybrid Dashboard")

   
    metrics_path = Path(
        st.sidebar.text_input("Metrics JSON", "runs/iris-experiment/metrics.json")
    )
    checkpoint_path = Path(
        st.sidebar.text_input("Checkpoint Path", "runs/iris-experiment/checkpoints/best.pt")
    )

   
    available_devices = ["cpu"]
    if torch.has_cuda:
        available_devices.append("cuda")
    if torch.backends.mps.is_available():
        available_devices.append("mps")

    device = st.sidebar.selectbox("Device", options=available_devices, index=0)

   
    metrics = load_metrics(metrics_path)
    if metrics:
        st.subheader("Training Metrics")
        epochs = [m["epoch"] for m in metrics]
        train_loss = [m["train_loss"] for m in metrics]
        val_loss = [m["val_loss"] for m in metrics]
        train_acc = [m["train_accuracy"] for m in metrics]
        val_acc = [m["val_accuracy"] for m in metrics]

        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers", name="Train Loss"))
        loss_fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="Val Loss"))
        loss_fig.update_layout(title="Loss over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(loss_fig, use_container_width=True)

        acc_fig = go.Figure()
        acc_fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode="lines+markers", name="Train Acc"))
        acc_fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines+markers", name="Val Acc"))
        acc_fig.update_layout(title="Accuracy over Epochs", xaxis_title="Epoch", yaxis_title="Accuracy")
        st.plotly_chart(acc_fig, use_container_width=True)

   
    if checkpoint_path.exists():
        try:
            service = HybridInferenceService(model_path=checkpoint_path, device=device)
            st.sidebar.success(f"Loaded model with backend: {service.backend_name}")

            st.subheader("Inference Playground")
            sample = st.text_area(
                "Input sample (comma-separated features)",
                "5.1,3.5,1.4,0.2",
            )

            if st.button("Predict"):
                try:
                    features = torch.tensor(
                        [[float(x.strip()) for x in sample.split(",")]],
                        dtype=torch.float32
                    )
                    preds, probs = service.predict(features)
                    st.write("Prediction:", preds.tolist())
                    st.write("Probabilities:", np.round(probs.numpy(), 3).tolist())
                except Exception as e:
                    st.error(f"Inference failed: {e}")
        except Exception as e:
            st.error(f"Model loading failed: {e}")

    
    st.subheader("Bloch Sphere Visualizer")
    theta = st.slider("Theta (rad)", 0.0, np.pi, np.pi / 4, step=0.01)
    phi = st.slider("Phi (rad)", 0.0, 2 * np.pi, np.pi / 2, step=0.01)
    st.plotly_chart(bloch_sphere_plot(theta, phi), use_container_width=True)


if __name__ == "__main__":
    main()
