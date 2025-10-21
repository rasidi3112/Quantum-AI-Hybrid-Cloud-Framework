## Models Module  
Quantum-AI Hybrid Cloud Framework
Modular, Reproducible, and Hardware-Aware Hybrid Quantum-Classical Architectures  


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)


---

## Overview  
The models/ module defines the core neural architectures of the Quantum-AI Hybrid Cloud Framework. It provides:  

- Classical feature extractors (fully configurable MLPs)  
- Parametrized quantum layers (PennyLane-based variational circuits)  
- End-to-end hybrid classifiers with skip connections and post-processing heads  


All components are:  

- Type-safe (using dataclass and type hints)  
- Hardware-aware (auto-adapts to CPU, MPS, CUDA, or QPU backends)  
- Reproducible (config-driven, seed-controlled)  
- Interoperable (compatible with PyTorch ecosystem, ONNX, TorchScript)

This module is the foundation for training, exporting, and deploying hybrid quantum-classical models in research and enterprise settings.  
## Module Structure  
```bash
models/
├── __init__.py                  # Public API exports
├── classical_model.py           # Classical MLP encoder
├── quantum_layers.py            # PennyLane-based quantum circuit layer
├── hybrid_model.py              # End-to-end hybrid classifier
└── classical_model_stub.py      # Pytest-compatible stub (internal use)
```
*💡 Note: classical_model_stub.py is only used for unit testing and is not part of the public API.*

## Core Components 
1. ClassicalFeatureExtractor (classical_model.py)  
   A. configurable multi-layer perceptron (MLP) for classical feature encoding.  
       Configuration (ClassicalModelConfig)
   
```bash
 @dataclass(frozen=True)
class ClassicalModelConfig:
    input_dim: int
    hidden_dims: Tuple[int, ...]  # e.g., (64, 32, 16)
    activation: str = "relu"      # relu, gelu, tanh
    dropout: float = 0.1
```

Features

- Supports arbitrary depth and width
- LayerNorm + Dropout for regularization
- Fully compatible with torch.nn.Module

2. QuantumLayer (quantum_layers.py)  
   A. differentiable quantum circuit implemented as a native PyTorch layer using PennyLane.
   Configuration (QuantumLayerConfig)
```bash
   @dataclass(frozen=True)
class QuantumLayerConfig:
    n_qubits: int          # Number of qubits (must match classical output)
    n_layers: int = 2      # Depth of StronglyEntanglingLayers
    backend: str = "default.qubit"  # e.g., "ibmq_qasm_simulator"
    shots: int | None = 1024        # Sampling shots (None = analytic)
    use_mps: bool = True   # Optimized for Apple Silicon 
   ```
Circuit Design  

- Feature Encoding: AngleEmbedding (Y-rotation)  
- Ansatz: StronglyEntanglingLayers (hardware-efficient)  
- Measurement: Expectation values of PauliZ on all qubits  
- Interface: torch (enables end-to-end gradient flow)

---

  Hardware Support
  

| Backend Type | Example | Execution Mode |
|---------------|----------|----------------|
| Local Simulator | `default.qubit` | Deterministic / Stochastic |
| Cloud QPU (IBM) | `ibmq_manila` | Stochastic (shots) |
| Cloud QPU (Rigetti) | `rigetti_aspen-m-3` | Stochastic |
| Cloud QPU (D-Wave) | via hybrid solvers | Specialized |  

*Auto-fallback: If cloud credentials are missing, defaults to local simulator.*

---   
3. HybridClassifier (hybrid_model.py)  
   An end-to-end hybrid model that fuses classical and quantum processing.
   
Architecture  :  
```bash  
    Input → [Classical MLP] → [Quantum Circuit] → [Skip Connection] → [Post-Quantum Head] → Logits
```   
   Configuration (HybridModelConfig)  
```bash   @dataclass(frozen=True)
class HybridModelConfig:
    classical: ClassicalModelConfig
    quantum: QuantumLayerConfig
    n_classes: int
    post_quantum_dim: Optional[int] = None  # e.g., 32
    use_skip_connection: bool = True        # Residual fusion
```
Key Features  

- Dimensional Consistency Check: Ensures classical.output_dim == quantum.n_qubits  
- Skip Connection: Fuses classical and quantum features for richer representations  
- Post-Quantum Head: MLP with LayerNorm, GELU, and Dropout for robust classification  
- Inference Methods:
     - forward(x) → logits  
     - predict_proba(x) → softmax probabilities  
     - predict(x) → class labels  


  
   

   
   
   





     




