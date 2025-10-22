## Training Module

Quantum-AI Hybrid Cloud Framework
Reproducible, Hardware-Aware, and Configurable Training Pipelines for Hybrid Quantum-Classical Models



[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-yellow.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/Quantum-PennyLane-purple.svg)](https://pennylane.ai/)

---

## Overview

The training/ module provides end-to-end, production-grade training pipelines for hybrid quantum-classical models. It supports:

- Reproducible experiments with full random seeding and config logging
- Cross-platform hardware acceleration (CPU, MPS, CUDA)
- Automatic checkpointing, metric logging, and visualization
- Hyperparameter grid search for model tuning
- Model export to TorchScript and ONNX for deployment

All training workflows are config-driven, CLI-accessible, and designed for international research collaboration and enterprise MLOps integration.

---

## Module Structure
```bash
training/
├── __init__.py                  # Public API exports
├── data.py                      # Dataset loading & preprocessing
├── train_hybrid.py              # Main training loop & CLI
└── hyperparameter_search.py     # Grid search over quantum/classical hyperparameters
```

## Core Components  
1. train_hybrid.py — Main Training Pipeline
   
   Features :  
- Hardware auto-detection: Uses MPS (Apple Silicon), CUDA (NVIDIA), or CPU based on availability
- Full reproducibility: Seeds for Python, NumPy, and PyTorch
- Modular config: All parameters controlled via TrainingConfig dataclass
- Validation loop: Tracks train/val loss & accuracy per epoch
- Checkpointing: Saves best model by validation accuracy
- Visualization: Generates interactive training_metrics.html
- Export: Optional TorchScript/ONNX export for deployment

---

## input  
- Tabular CSV dataset (e.g., examples/iris.csv) with label column
- Configurable via CLI or programmatic TrainingConfig

## Output (saved to runs/<experiment>/)
```bash
checkpoints/
  ├── best.pt          # Best model by val accuracy
  └── epoch_*.pt       # Per-epoch checkpoints
metrics.json           # Full training history
plots/
  └── training_metrics.html  # Interactive Plotly dashboard
model.ts               # TorchScript (optional)
model.onnx             # ONNX (optional)
training.log           # Structured logs
```
---
## CLI Usage  
```bash
python -m training.train_hybrid \
  --dataset examples/iris.csv \
  --output runs/iris-hybrid \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --quantum-layers 3 \
  --backend default.qubit \
  --device mps  # or --no-gpu --no-mps for CPU
```

2. hyperparameter_search.py — Automated Tuning  
Purpose  
Perform grid search over key hyperparameters:
- Learning rate (--learning-rates)
- Quantum circuit depth (--quantum-layers)
- Quantum backend (--backends)

  

  ## Output Structure
  ```bash
  output_dir/
  ├── lr0.001_layers2_backenddefault_qubit/
  │   ├── metrics.json
  │   └── checkpoints/best.pt
  └── lr0.0005_layers3_backendibmq_qasm_simulator/
      ├── metrics.json
      └── checkpoints/best.pt
  ```

  
  ## CLI Usage
  ```bash
  python -m training.hyperparameter_search \
  --dataset examples/iris.csv \
  --output hp_search/iris \
  --learning-rates 1e-3 5e-4 \
  --quantum-layers 2 3 \
  --backends default.qubit ibmq_qasm_simulator
  ```
*⚠️ Note: For cloud QPUs (IBM, Rigetti), set environment variables (e.g., QAI_IBM_TOKEN) before running.*

  ---

  3. data.py — Dataset Utilities  
     Supported Format  
     - CSV with columns: feature_1, feature_2, ..., label  
     - Stratified split: Ensures balanced train/validation sets  
     - Auto-inference: Detects input_dim and n_classes from data
    
Configuration (DataConfig)  
```bash
@dataclass(frozen=True)
class DataConfig:
    dataset_path: Path
    batch_size: int = 32
    val_split: float = 0.2
    shuffle: bool = True
    seed: int = 42
```
Hardware & Reproducibility  
| Feature              | Implementation                                   |
|----------------------|--------------------------------------------------|
| Device Selection     | Auto: MPS > CUDA > CPU                           |
| Random Seeding       | Full stack (Python, NumPy, PyTorch)              |
| Cloud QPU Fallback   | Local simulator if credentials missing           |
| Apple Silicon (M1/M2)| Optimized via --use_mps (default: enabled)       |

*Best Practice: Always specify --seed for reproducible results.*

---

 Integration with Full Framework  
 The training/ module is the core engine of the framework and integrates seamlessly with:  
 | Component   | Integration Point                                  |
|--------------|----------------------------------------------------|
| models/      | Instantiates HybridClassifier from config          |
| utils/       | Logging, checkpointing, hardware, export           |
| deployment/  | Trained .pt files served via FastAPI/Streamlit     |
| tests/       | Validated via test_sanity.py                       |
| examples/    | Uses iris.csv as default dataset                   |

---
## License
Distributed under the MIT License. See LICENSE for details.  


## Acknowledgements  
This module leverages:

- PyTorch for classical training loops and optimizers  
- PennyLane for differentiable quantum circuits  
- scikit-learn for stratified data splitting  
- Plotly for interactive training visualizations  

Designed for rigorous, reproducible quantum machine learning research at scale.  

