"""Hybrid quantum-classical model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch # type: ignore
from torch import nn # type: ignore

from .classical_model import ClassicalFeatureExtractor, ClassicalModelConfig
from .quantum_layers import QuantumLayer, QuantumLayerConfig


@dataclass(frozen=True)
class HybridModelConfig:
    classical: ClassicalModelConfig
    quantum: QuantumLayerConfig
    n_classes: int
    post_quantum_dim: Optional[int] = None
    use_skip_connection: bool = True


class HybridClassifier(nn.Module):
    """Quantum-classical hybrid classifier architecture."""

    def __init__(self, config: HybridModelConfig) -> None:
        super().__init__()
        self.config = config

        if config.classical.output_dim != config.quantum.n_qubits:
            raise ValueError(
                "Classical output dimension must match quantum layer qubits. "
                f"Got classical={config.classical.output_dim} vs quantum={config.quantum.n_qubits}"
            )

        self.classical_encoder = ClassicalFeatureExtractor(config.classical)
        self.quantum_layer = QuantumLayer(config.quantum)

        post_dim = config.post_quantum_dim or config.quantum.n_qubits
        layers = [
            nn.Linear(config.quantum.n_qubits, post_dim),
            nn.LayerNorm(post_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(post_dim, config.n_classes),
        ]
        self.post_quantum_head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        classical_features = self.classical_encoder(x)
        quantum_outputs = self.quantum_layer(classical_features)

        # Flatten jika perlu
        if quantum_outputs.ndim > 2:
            quantum_outputs = quantum_outputs.squeeze(-1)
        if quantum_outputs.shape[0] != x.shape[0]:
            quantum_outputs = quantum_outputs.T

        if self.config.use_skip_connection:
            if classical_features.shape[1] != quantum_outputs.shape[1]:
                projector = nn.Linear(classical_features.shape[1], quantum_outputs.shape[1]).to(x.device)
                skip = projector(classical_features)
            else:
                skip = classical_features

            min_batch = min(skip.shape[0], quantum_outputs.shape[0])
            fused = quantum_outputs[:min_batch] + skip[:min_batch]
        else:
            fused = quantum_outputs

        fused = fused.reshape(fused.shape[0], -1)
        logits = self.post_quantum_head(fused)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.predict_proba(x), dim=-1)
