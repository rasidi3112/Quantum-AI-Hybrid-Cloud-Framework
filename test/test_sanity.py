"""Basic sanity checks for hybrid framework modules."""


from models.classical_model_stub import ClassicalFeatureExtractor
from models.classical_model import ClassicalModelConfig
from models.quantum_layers import QuantumLayer, QuantumLayerConfig
from models.hybrid_model import HybridClassifier, HybridModelConfig


from models import (
    ClassicalModelConfig,
    HybridClassifier,
    HybridModelConfig,
    QuantumLayerConfig,
)
import torch # type: ignore


def test_hybrid_forward_pass() -> None:
    """Ensure the hybrid model produces logits of expected shape."""
    classical_config = ClassicalModelConfig(input_dim=4, hidden_dims=(8, 4), activation="relu", dropout=0.0)
    quantum_config = QuantumLayerConfig(n_qubits=classical_config.output_dim, n_layers=1, backend="default.qubit")
    hybrid_config = HybridModelConfig(classical=classical_config, quantum=quantum_config, n_classes=3)

    model = HybridClassifier(hybrid_config)
    inputs = torch.randn(2, 4)
    logits = model(inputs)
    assert logits.shape == (2, 3)

    