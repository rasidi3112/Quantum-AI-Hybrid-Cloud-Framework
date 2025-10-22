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

4. logging_utils.py — Structured Experiment Tracking
   Features:
   - Dual-channel logging: console + file (training.log)  
   - ISO 8601 timestamps for auditability  
   - JSON metric persistence (metrics.json) for programmatic analysis

   Log Format:
```bash
2025-10-15 14:30:22 | INFO | Starting training with configuration: {...}
2025-10-15 14:31:05 | INFO | Epoch 1 metrics: {"train_loss": 0.62, "val_accuracy": 0.78}
2025-10-15 14:32:10 | INFO | Epoch 2 metrics: {"train_loss": 0.58, "val_accuracy": 0.81}
2025-10-15 14:33:15 | INFO | Epoch 3 metrics: {"train_loss": 0.55, "val_accuracy": 0.83}
```
*"Enables reproducible research compliant with ML reproducibility checklists."*  

---

5. visualization.py — Interactive Training Analytics
   Output: training_metrics.html  
- Dual-axis plot: Loss (left) + Accuracy (right)  
- Train/val comparison: Clear overfitting detection  
- Plotly-powered: Zoom, pan, and export to PNG/SVG

![Training Metrics Screenshot](docs/assets/training_metrics.png)  

   "*Zero external dependencies: Self-contained HTML file viewable in any browser.*"  

   ---
## Integration Across the Framework  


| Component       | Uses / Utils For                                   |
|-----------------|---------------------------------------------------|
| `training/`     | Checkpointing, logging, hardware detection, visualization |
| `deployment/`   | Model loading from checkpoints                    |
| `qml_app.export`| TorchScript / ONNX serialization                 |
| `tests/`        | Reproducible hardware environment                |


  ---
  ## License  





##  Philosophy

“Infrastructure should disappear.”  
— Quantum-AI Hybrid Cloud Framework Design Principle   

 The utils/ module embodies invisible excellence: robust, unobtrusive tooling that enables researchers to focus on quantum innovation rather than engineering overhead.  

 ---






   
