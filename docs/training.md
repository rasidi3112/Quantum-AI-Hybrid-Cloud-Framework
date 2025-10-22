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

