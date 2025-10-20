"""Simple hyperparameter grid search for the hybrid training pipeline."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from training.train_hybrid import TrainingConfig, train
from utils.logging_utils import create_logger  # type: ignore


@dataclass
class HyperparameterSearchConfig:
    """Configuration for hyperparameter search."""
    dataset_path: Path
    output_dir: Path
    learning_rates: Iterable[float] = field(default_factory=lambda: (1e-3, 5e-4))
    quantum_layers: Iterable[int] = field(default_factory=lambda: (2, 3))
    backends: Iterable[str] = field(default_factory=lambda: ("default.qubit",))
    epochs: int = 15
    batch_size: int = 32

    def ensure_output(self) -> None:
        """Create root output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


def run_grid_search(config: HyperparameterSearchConfig) -> List[Dict[str, float]]:
    """Execute a simple grid search over hyperparameters."""
    config.ensure_output()
    logger = create_logger(config.output_dir / "hyperparameter_search.log")
    logger.info("Starting hyperparameter search with config: %s", config)

    results = []
    for lr, layers, backend in itertools.product(
        config.learning_rates, config.quantum_layers, config.backends
    ):
        run_name = f"lr{lr}_layers{layers}_backend{backend.replace('.', '_')}"
        run_dir = config.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)  # Pastikan folder run ada

        # Paksa CPU supaya aman di Mac M1/M2
        train_cfg = TrainingConfig(
            dataset_path=config.dataset_path,
            output_dir=run_dir,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=float(lr),
            quantum_layers=int(layers),
            backend=str(backend),
            use_gpu=False,
            use_mps=False,
        )

        metrics = train(train_cfg)
        metrics.update({
            "run": run_name,
            "learning_rate": float(lr),
            "quantum_layers": int(layers),
            "backend": str(backend)
        })
        results.append(metrics)
        logger.info("Run %s finished with metrics %s", run_name, metrics)

    return results


def parse_args() -> HyperparameterSearchConfig:
    """Parse CLI arguments for hyperparameter search."""
    parser = argparse.ArgumentParser(description="Hyperparameter search for hybrid model.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-3, 5e-4])
    parser.add_argument("--quantum-layers", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--backends", type=str, nargs="+", default=["default.qubit"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Pastikan backend valid dan tidak None
    backends = [b if b.lower() != "none" else "default.qubit" for b in (args.backends or ["default.qubit"])]

    return HyperparameterSearchConfig(
        dataset_path=args.dataset,
        output_dir=args.output,
        learning_rates=[float(lr) for lr in args.learning_rates],
        quantum_layers=[int(l) for l in args.quantum_layers],
        backends=[str(b) for b in backends],
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_grid_search(cfg)
