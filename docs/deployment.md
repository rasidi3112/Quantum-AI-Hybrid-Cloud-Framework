## Deployment Module
Quantum-AI Hybrid Cloud Framework
Enterprise-Ready Inference, API, and Visualization for Hybrid Quantum-Classical Models

## Overview

The deployment/ module provides production-grade tooling for serving, monitoring, and interacting with trained hybrid quantum-classical models. It enables:

- RESTful inference via a secure, scalable FastAPI service
- Interactive experimentation through a Streamlit dashboard
- Cross-platform compatibility (CPU, CUDA, MPS)
- Quantum-aware visualization (Bloch sphere, training metrics)
  
 This module is designed for reproducible research, enterprise deployment, and cloud-native quantum AI workflows.

## Module Structure
```bash
deployment/
├── __init__.py          # Exports core components
├── api.py               # FastAPI inference server
├── dashboard.py         # Streamlit monitoring & playground
└── inference.py         # Model loading & prediction logic
```

## Core Components

1. HybridInferenceService (inference.py)
     
    A. Stateless service that:
     - Loads checkpointed hybrid models (.pt) with full architecture reconstruction
     - Auto-infers model topology from weights (input dim, qubits, classes)
     - Supports local simulators (default.qubit) and cloud QPUs (IBM, Rigetti, D-Wave)
     - Handles device mapping (CPU/CUDA/MPS) transparently
     - Exposes backend metadata (e.g., ibmq_manila, rigetti_aspen-m-3)
  
   *"Key Feature: No retraining needed — deploy any checkpoint from runs/. "*

2. FastAPI REST Service (api.py)

     A. Production-ready inference API with:
       

| ENDPOINT        | METHOD | DESCRIPTION |
|-----------------|---------|-------------|
| `/Q/health`     | `GET`  | Returns service status and active quantum backend |
| `/Q/predict`    | `POST` | Accepts `[[features]]`, returns predictions + probabilities |

   
 
