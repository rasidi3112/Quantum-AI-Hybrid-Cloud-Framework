"""Inference utilities wrapping hybrid model loading and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch # type: ignore

from models import (
    ClassicalModelConfig,
    HybridClassifier,
    HybridModelConfig,
    QuantumLayerConfig,
)
from utils.hardware import QuantumBackendInfo # type: ignore


@dataclass
class HybridInferenceService:
    """Service responsible for loading checkpoints and performing inference."""
    model_path: Path
    device: str = "cpu"  # PAKSA CPU untuk hindari MPS bug

    def __post_init__(self) -> None:
        self._device = torch.device(self.device)
        checkpoint = torch.load(self.model_path, map_location=self._device)
        
        # Ambil state_dict dari checkpoint
        state_dict = checkpoint["model_state"]

        # Tentukan dimensi input/output dari checkpoint
        # Jika ada classical_encoder
        classical_weight = state_dict.get("classical_encoder.model.0.weight")
        input_dim = classical_weight.shape[1] if classical_weight is not None else 32
        hidden_dims = [classical_weight.shape[0]] if classical_weight is not None else [32]

        # Quantum layer dimensi
        quantum_weight = state_dict.get("quantum_layer.layer.weights")
        n_qubits = quantum_weight.shape[1] if quantum_weight is not None else hidden_dims[-1]
        quantum_layers = quantum_weight.shape[0] if quantum_weight is not None else 2

        # Jumlah kelas output
        post_head_weight = state_dict.get("post_quantum_head.4.weight")
        n_classes = post_head_weight.shape[0] if post_head_weight is not None else 3

        # Ambil konfigurasi lain dari checkpoint jika ada
        config_dict = checkpoint.get("config") or {}
        classical_activation = config_dict.get("classical_activation") or "relu"
        classical_dropout = config_dict.get("classical_dropout") or 0.1
        backend_name = config_dict.get("backend")
        shots = config_dict.get("shots")

        classical_config = ClassicalModelConfig(
            input_dim=input_dim,
            hidden_dims=tuple(hidden_dims),
            activation=classical_activation,
            dropout=classical_dropout,
        )
        quantum_config = QuantumLayerConfig(
            n_qubits=n_qubits,
            n_layers=quantum_layers,
            backend=backend_name,
            shots=shots,
        )
        hybrid_config = HybridModelConfig(
            classical=classical_config,
            quantum=quantum_config,
            n_classes=n_classes,
        )

        # Buat model
        self.model = HybridClassifier(hybrid_config)

        # Load state_dict dengan strict=False untuk skip mismatch layer
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self._device)
        self.model.eval()

        # Simpan info backend
        self.backend_info: QuantumBackendInfo = getattr(
            self.model.quantum_layer, "backend_info", QuantumBackendInfo(name="cpu")
        )

    @property
    def backend_name(self) -> str:
        """Return the backend name used by the quantum layer."""
        return self.backend_info.name

    def predict(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform inference and return predictions and probabilities."""
        features = features.to(self._device)
        with torch.no_grad():
            probs = self.model.predict_proba(features)
            preds = probs.argmax(dim=1)
        return preds.cpu(), probs.cpu()
