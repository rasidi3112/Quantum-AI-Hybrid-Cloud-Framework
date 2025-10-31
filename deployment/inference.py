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
from utils.hardware import QuantumBackendInfo
import pennylane as qml # type: ignore


@dataclass
class HybridInferenceService:
    """Service for loading hybrid checkpoints and inference."""
    model_path: Path
    device: str = "cpu" 

    def __post_init__(self) -> None:
        self._device = torch.device(self.device)

       
        checkpoint = torch.load(self.model_path, map_location=self._device)
        state_dict = checkpoint["model_state"]

      
        classical_weight = state_dict.get("classical_encoder.model.0.weight")
        input_dim = classical_weight.shape[1] if classical_weight is not None else 4
        hidden_dims = [classical_weight.shape[0]] if classical_weight is not None else [32]

       
        quantum_weight = state_dict.get("quantum_layer.layer.weights")
        n_qubits = quantum_weight.shape[1] if quantum_weight is not None else hidden_dims[-1]
        quantum_layers = quantum_weight.shape[0] if quantum_weight is not None else 2

      
        post_head_weight = state_dict.get("post_quantum_head.4.weight")
        n_classes = post_head_weight.shape[0] if post_head_weight is not None else 3

       
        config_dict = checkpoint.get("config") or {}
        classical_activation = config_dict.get("classical_activation") or "relu"
        classical_dropout = config_dict.get("classical_dropout") or 0.1
        backend_name = config_dict.get("backend") or "default.qubit"
        shots = config_dict.get("shots") or 1024

     
        if classical_weight is not None and classical_weight.shape[0] != n_qubits:
            print(f"[INFO] Resizing classical encoder from {classical_weight.shape[0]} -> {n_qubits}")
            state_dict["classical_encoder.model.0.weight"] = classical_weight[:n_qubits, :]
            state_dict["classical_encoder.model.0.bias"] = state_dict["classical_encoder.model.0.bias"][:n_qubits]
            hidden_dims[-1] = n_qubits

       
        classical_config = ClassicalModelConfig(
            input_dim=input_dim,
            hidden_dims=tuple(hidden_dims),
            activation=classical_activation,
            dropout=classical_dropout,
        )
        quantum_config = QuantumLayerConfig(
            n_qubits=n_qubits,
            n_layers=quantum_layers,
            backend=backend_name if backend_name != "cpu" else "default.qubit",
            shots=shots,
        )
        hybrid_config = HybridModelConfig(
            classical=classical_config,
            quantum=quantum_config,
            n_classes=n_classes,
        )

        # --- Instantiate model ---
        self.model = HybridClassifier(hybrid_config)
        self.model.load_state_dict(state_dict, strict=False)  
        self.model.to(self._device)
        self.model.eval()

        # --- Backend info ---
        self.backend_info: QuantumBackendInfo = getattr(
            self.model.quantum_layer,
            "device_info",
            QuantumBackendInfo(name=quantum_config.backend, is_hardware=False, provider="local", shots=shots),
        )

    @property
    def backend_name(self) -> str:
        return self.backend_info.name

    def predict(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = features.to(self._device)
        with torch.no_grad():
            probs = self.model.predict_proba(features)
            preds = probs.argmax(dim=1)
        return preds.cpu(), probs.cpu()
