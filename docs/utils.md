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

Key Capabilities  

- Dynamic batch support via dynamic_axes  
- Signature-based reconstruction: rebuild model from checkpoint metadata  
- Device-agnostic tracing (CPU/MPS/CUDA)

  
  CLI Usage
  ```bash
  python -m utils.export \
  --checkpoint runs/iris/checkpoints/best.pt \
  --torchscript exports/model.ts \
  --onnx exports/model.onnx \
  --device cpu
  ```

  ---

3.hardware.py — Unified Hardware Abstraction  
   Quantum Backend Resolution  
   

| Backend Type      | Environment Variable       | Fallback Behavior                 |
|------------------|---------------------------|----------------------------------|
| Local Simulator  | `QAI_DEFAULT_BACKEND`     | `default.qubit`                  |
| IBM QPU          | `QAI_IBM_TOKEN`           | Local simulator if missing       |
| Rigetti QVM      | (auto-detected)           | Local simulator if unavailable   |
| D-Wave           | `QAI_DWAVE_KEY`           | Planned                          |


   Hardware Summary  
```bash
=== Hardware Summary ===
Platform: macOS 14.5
Python: 3.11.9
CPU cores: 8
PyTorch threads: 8
No CUDA GPU detected
PennyLane version: 0.35.1
========================
```
*"Auto-detection: Framework silently falls back to local simulator if cloud credentials are missing — ideal for CI/CD."*  




   
