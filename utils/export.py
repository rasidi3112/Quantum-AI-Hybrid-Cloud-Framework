"""Model export utilities for TorchScript and ONNX formats."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch # type: ignore

from models import (
    ClassicalModelConfig,
    HybridClassifier,
    HybridModelConfig,
    QuantumLayerConfig,
)


def export_to_torchscript(
    model: HybridClassifier,
    input_dim: int,
    path: Path,
    device: torch.device,
) -> None:
    """Export model to TorchScript format."""
    model.eval()
    dummy_input = torch.randn(1, input_dim, device=device)
    traced = torch.jit.trace(model, dummy_input)
    traced.save(str(path))


def export_to_onnx(
    model: HybridClassifier,
    input_dim: int,
    path: Path,
    device: torch.device,
) -> None:
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, input_dim, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def build_model_from_signature(signature: dict) -> HybridClassifier:
    """Reconstruct a model from a signature dictionary."""
    classical_config = ClassicalModelConfig(
        input_dim=signature["input_dim"],
        hidden_dims=tuple(signature["classical_hidden"]),
        activation=signature["classical_activation"],
        dropout=signature["classical_dropout"],
    )
    quantum_config = QuantumLayerConfig(
        n_qubits=signature["quantum_n_qubits"],
        n_layers=signature["quantum_layers"],
        backend=signature.get("backend"),
        shots=signature.get("shots"),
    )
    hybrid_config = HybridModelConfig(
        classical=classical_config,
        quantum=quantum_config,
        n_classes=signature["n_classes"],
        post_quantum_dim=signature.get("post_quantum_dim"),
        use_skip_connection=signature.get("use_skip_connection", True),
    )
    return HybridClassifier(hybrid_config)


def cli() -> None:
    """CLI for exporting models from checkpoint files."""
    parser = argparse.ArgumentParser(description="Export hybrid model to TorchScript and ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint (best.pt).")
    parser.add_argument("--onnx", type=Path, default=None, help="ONNX output path.")
    parser.add_argument("--torchscript", type=Path, default=None, help="TorchScript output path.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    signature = checkpoint["signature"]
    model = build_model_from_signature(signature)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device)

    if args.torchscript:
        export_to_torchscript(model, signature["input_dim"], args.torchscript, device)
    if args.onnx:
        export_to_onnx(model, signature["input_dim"], args.onnx, device)


if __name__ == "__main__":
    cli()