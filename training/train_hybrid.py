"""Hybrid training loop orchestrating classical and quantum components."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np # type: ignore
import torch # type: ignore
from torch import nn, optim # type: ignore
from tqdm import tqdm # type: ignore

from models import (
    ClassicalModelConfig,
    HybridClassifier,
    HybridModelConfig,
    QuantumLayerConfig,
)
from training.data import DataConfig, load_tabular_dataset
from utils.checkpoint import CheckpointManager
from utils.export import export_to_onnx, export_to_torchscript
from utils.hardware import summarize_hardware
from utils.logging_utils import create_logger, dump_metrics
from utils.visualization import create_loss_accuracy_figure


@dataclass
class TrainingConfig:
    """Configuration for hybrid training."""
    dataset_path: Path
    output_dir: Path
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    backend: Optional[str] = None
    shots: Optional[int] = None
    seed: int = 42
    use_gpu: bool = True
    use_mps: bool = True
    log_interval: int = 10
    weight_decay: float = 1e-4
    classical_hidden: tuple[int, ...] = (64, 32)
    classical_activation: str = "relu"
    classical_dropout: float = 0.1
    quantum_layers: int = 2
    post_quantum_dim: Optional[int] = None
    export_onnx: bool = True
    export_torchscript: bool = True

    def ensure_output(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: TrainingConfig) -> torch.device:
    """Determine the best available torch device based on the config."""
    if config.use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if config.use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def serialize_config(config: TrainingConfig) -> dict:
    """Convert TrainingConfig to JSON-serializable dict (Path -> str)."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}


def train(config: TrainingConfig) -> Dict[str, float]:
    """Run the hybrid training loop."""
    config.ensure_output()
    set_random_seed(config.seed)
    logger = create_logger(config.output_dir / "training.log")

    logger.info(
        "Starting training with configuration: %s",
        json.dumps(serialize_config(config), indent=2)
    )

    device = resolve_device(config)
    logger.info("Using device: %s", device)
    logger.info("Hardware summary: %s", summarize_hardware())

    # =============================
    # Load dataset
    # =============================
    data_config = DataConfig(
        dataset_path=config.dataset_path,
        batch_size=config.batch_size,
        val_split=0.2,
        seed=config.seed,
    )
    train_loader, val_loader, n_classes = load_tabular_dataset(data_config)
    input_dim = next(iter(train_loader))[0].shape[-1]

    # =============================
    # Classical model
    # =============================
    classical_config = ClassicalModelConfig(
        input_dim=input_dim,
        hidden_dims=config.classical_hidden,
        activation=config.classical_activation,
        dropout=config.classical_dropout,
    )

    # Adjust classical output dim to match qubits
    max_qubits = 8
    if classical_config.output_dim != max_qubits:
        logger.info(f"[INFO] Adjusting classical output dim {classical_config.output_dim} -> {max_qubits}")
        projection_layer = nn.Linear(classical_config.output_dim, max_qubits).to(device)
        classical_config.output_dim = max_qubits
        classical_config.projection_layer = projection_layer

    quantum_config = QuantumLayerConfig(
        n_qubits=classical_config.output_dim,
        n_layers=config.quantum_layers,
        backend=config.backend,
        shots=config.shots,
    )

    hybrid_config = HybridModelConfig(
        classical=classical_config,
        quantum=quantum_config,
        n_classes=n_classes,
        post_quantum_dim=config.post_quantum_dim,
    )

    model = HybridClassifier(hybrid_config).to(device)

    # Apply projection if exists
    if hasattr(classical_config, "projection_layer"):
        model.classical_encoder.model.add_module("projection", classical_config.projection_layer)

    logger.info("Hybrid model instantiated: %s", model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    checkpoint_manager = CheckpointManager(config.output_dir / "checkpoints")
    best_val_accuracy = 0.0
    metrics_history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for step, (features, labels) in enumerate(progress, start=1):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += features.size(0)

            if step % config.log_interval == 0:
                progress.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / total_train
        train_accuracy = train_correct / total_train

        # =============================
        # Validation
        # =============================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * features.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += features.size(0)

        avg_val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total

        metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
        }
        metrics_history.append(metrics)
        logger.info("Epoch %d metrics: %s", epoch, metrics)

        # Save checkpoint
        checkpoint_manager.save(
            state={
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
                "config": serialize_config(config),
            },
            filename=f"epoch_{epoch}.pt",
        )

        # Save best checkpoint
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_manager.save_best(
                model_state=model.state_dict(),
                metrics=metrics,
                signature=f"epoch{epoch}_val{val_accuracy:.4f}"
            )

    # =============================
    # Metrics & plot
    # =============================
    metrics_path = config.output_dir / "metrics.json"
    dump_metrics(metrics_history, metrics_path)

    figure = create_loss_accuracy_figure(metrics_history)
    plot_path = config.output_dir / "plots" / "training_metrics.html"
    figure.write_html(plot_path)
    logger.info("Training metrics plotted at %s", plot_path)

    # Load best checkpoint
    best_checkpoint = checkpoint_manager.best_checkpoint_path
    if best_checkpoint:
        model.load_state_dict(torch.load(best_checkpoint, map_location=device)["model_state"])

    # Export model
    if config.export_torchscript:
        export_to_torchscript(model, input_dim, config.output_dir / "model.ts", device)
    if config.export_onnx:
        export_to_onnx(model, input_dim, config.output_dir / "model.onnx", device)

    return metrics_history[-1] if metrics_history else {}


def parse_args() -> TrainingConfig:
    """Parse CLI arguments into a TrainingConfig."""
    parser = argparse.ArgumentParser(description="Train a hybrid quantum-classical model.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to CSV dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--no-mps", action="store_true")
    parser.add_argument("--quantum-layers", type=int, default=2)
    parser.add_argument("--post-quantum-dim", type=int, default=None)
    parser.add_argument("--classical-hidden", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--classical-activation", type=str, default="relu")
    parser.add_argument("--classical-dropout", type=float, default=0.1)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--export-torchscript", action="store_true")
    args = parser.parse_args()

    return TrainingConfig(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        backend=args.backend,
        shots=args.shots,
        seed=args.seed,
        use_gpu=not args.no_gpu,
        use_mps=not args.no_mps,
        quantum_layers=args.quantum_layers,
        post_quantum_dim=args.post_quantum_dim,
        classical_hidden=tuple(args.classical_hidden),
        classical_activation=args.classical_activation,
        classical_dropout=args.classical_dropout,
        export_onnx=args.export_onnx,
        export_torchscript=args.export_torchscript,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
