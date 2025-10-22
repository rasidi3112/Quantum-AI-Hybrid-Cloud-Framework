## Utils Module

Quantum-AI Hybrid Cloud Framework
Cross-Cutting Concerns for Reproducible, Portable, and Production-Ready Quantum AI Workflows



[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-v0.27-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)


---

## Overview

The utils/ module provides essential cross-cutting utilities that power the entire Quantum-AI Hybrid Cloud Framework. These tools ensure:

- Reproducibility through structured logging and checkpointing
- Portability across CPU, GPU, MPS, and cloud QPUs
- Interoperability via standardized model export (TorchScript, ONNX)
- Transparency with hardware introspection and metric visualization

All utilities are modular, dependency-minimal, and designed for zero-config integration into training, evaluation, and deployment pipelines.

---

## Module Structure  
```bash
utils/
├── checkpoint.py        # Model checkpoint management
├── export.py            # TorchScript & ONNX model export
├── hardware.py          # Device detection & quantum backend resolution
├── logging_utils.py     # Structured logging & metric persistence
└── visualization.py     # Interactive training metric plots
```

## Core Utilities
1. checkpoint.py — Reproducible Model Persistence  
   Features:
- Automatic directory creation for experiment runs  
- Best-model tracking based on validation metrics  
- Full state saving: model weights, optimizer state, config, and metrics  

Output Format (best.pt) 
```bash
{
  "model_state": {...},      # HybridClassifier state_dict
  "metrics": {...},          # Final val accuracy, loss, etc.
  "signature": {...}         # Model architecture blueprint
}
```
*"Enables exact model reconstruction without original config files."*

---

2. export.py — Deployment-Ready Model Serialization  
   Supported Formats
   

| Format      | Use Case                                    | CLI Command                |
|------------|--------------------------------------------|---------------------------|
| TorchScript | PyTorch production inference               | `--torchscript model.ts`  |
| ONNX        | Cross-framework deployment (TensorRT, ONNX Runtime) | `--onnx model.onnx`       |
