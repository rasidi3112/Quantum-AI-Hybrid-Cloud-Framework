"""Quantum layer definitions leveraging PennyLane for hybrid models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import pennylane as qml  # type: ignore
import torch  # type: ignore
from torch import nn  # type: ignore

from utils.hardware import QuantumBackendInfo  # type: ignore


@dataclass(frozen=True)
class QuantumLayerConfig:
    """Configuration for a parametrized quantum circuit layer."""
    n_qubits: int
    n_layers: int = 2
    backend: str | None = None
    shots: int | None = None
    use_mps: bool = True  # tambahan untuk Mac M1/M2


class QuantumLayer(nn.Module):
    """Parametrized quantum circuit implemented as a Torch layer."""

    def __init__(self, config: QuantumLayerConfig) -> None:
        """Create a quantum layer with the specified configuration."""
        super().__init__()
        if config.n_qubits <= 0:
            raise ValueError("Number of qubits must be positive.")

        self.config = config

        # Pilih backend default.qubit jika tidak ditentukan
        backend_name = config.backend or "default.qubit"

        # Inisialisasi device PennyLane yang valid
        self.device = qml.device(backend_name, wires=config.n_qubits, shots=config.shots)

        # Inisialisasi QuantumBackendInfo sesuai signature terbaru
        self.device_info = QuantumBackendInfo(
            name=backend_name,
            is_hardware=False,   # simulator
            provider="local",    # provider default
            shots=config.shots
        )

        # Bentuk weight untuk StronglyEntanglingLayers: [layers, qubits, 3]
        self.weight_shapes: Dict[str, Tuple[int, int, int]] = {
            "weights": (config.n_layers, config.n_qubits, 3)
        }

        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            """Quantum circuit for TorchLayer."""
            qml.templates.AngleEmbedding(
                inputs, wires=range(config.n_qubits), rotation="Y"
            )
            qml.templates.StronglyEntanglingLayers(
                weights, wires=range(config.n_qubits)
            )
            return [qml.expval(qml.PauliZ(w)) for w in range(config.n_qubits)]


        self.qnode: Callable[..., torch.Tensor] = qml.QNode(
            circuit,
            device=self.device,
            interface="torch",
            diff_method="best",
        )

        # TorchLayer untuk integrasi dengan PyTorch
        self.layer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the quantum layer to inputs.

        Zero-pad atau truncate agar cocok dengan jumlah qubit.
        """
        # Pastikan input di CPU agar compatible dengan QNode
        inputs = inputs.to(torch.device("cpu"))

        batch_size, feature_dim = inputs.shape
        inputs_processed = inputs
        if feature_dim < self.config.n_qubits:
            pad_width = self.config.n_qubits - feature_dim
            inputs_processed = torch.nn.functional.pad(inputs, (0, pad_width))
        elif feature_dim > self.config.n_qubits:
            inputs_processed = inputs[:, : self.config.n_qubits]

        out = self.layer(inputs_processed)

        # Pastikan output 2D: [batch_size, n_qubits]
        if out.ndim == 3 and out.shape[2] == 1:
            out = out.squeeze(-1)
        if out.ndim == 1:
            out = out.unsqueeze(0)

        return out.float()

    def extra_repr(self) -> str:
        """Provide string representation for module summary."""
        backend = self.device_info.name
        return (
            f"n_qubits={self.config.n_qubits}, "
            f"n_layers={self.config.n_layers}, backend={backend}"
        )
