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

  Security & Reliability  
     - Model loading at startup (503 if not ready)  
     - Type-safe request/response (Pydantic)  
     - Structured logging (utils/logging_utils)  
     - Graceful error handling  
       
     
   Launch Command  
   ```bash
      python -m deployment.api \
        --model-path runs/iris-experiment/checkpoints/best.pt \
        --device mps \
        --host 0.0.0.0 \
        --port 8000
  ```
   ---
   Example request 

```json
{
  "samples": [[5.1, 3.5, 1.4, 0.2]]
}
```

   Example Response  

```json
    {
  "predictions": [0],
  "probabilities": [[0.95, 0.03, 0.02]],
  "backend": "default.qubit"
}
``` 

3. Streamlit Dashboard (dashboard.py)  
   An interactive dashboard for researchers and ML engineers featuring:  
     Training Metrics  
        -  Real-time loss & accuracy curves (loaded from metrics.json)  
        -  Epoch-wise comparison (train vs. validation)  
  
     Inference Playground  
        -  Input sample via text field 
        -  Live prediction with probability breakdown  
        -  Device selector (CPU / CUDA / MPS)  
   
    Quantum State Visualizer  
        -  Bloch Sphere for single-qubit state representation  
        -  Interactive sliders for θ (theta) and φ (phi)  
        -  Built with Plotly for high-fidelity rendering  

     *💡 Tip: Works seamlessly on Apple Silicon  via MPS acceleration.*

##   Quick Start
Prerequisites

- Python ≥ 3.11
- Trained model checkpoint (e.g., runs/*/checkpoints/best.pt)
- requirements.txt installed

  Launch API
  ```bash
  python -m deployment.api --model-path runs/your_run/checkpoints/best.pt --device cpu
   ```
  Launch Dashboard
  ```bash
  streamlit run deployment/dashboard.py
     ```
     Then open http://localhost:8501

 ## Cloud & Enterprise Readiness

| Feature | Supported |
|----------|------------|
| Docker containerization |  (Dockerfile in root) |
| Kubernetes deployment |  (via FastAPI + Uvicorn) |
| API authentication |  (Planned via middleware) |
| Model versioning |  (via checkpoint + config) |
| Cross-platform |  (Windows, Linux, macOS) |
| QPU credential injection |  (`QAI_IBM_TOKEN`, etc.) |

---

## License  
  Distributed under the [MIT License](LICENSE). See `LICENSE` for details.   

---

## Acknowledgements  

This module integrates:

- FastAPI for high-performance APIs  
- Streamlit for rapid dashboard prototyping  
- PennyLane + Qiskit for quantum backend abstraction  
- PyTorch for classical-quantum hybrid execution  
 Designed for international research collaboration and quantum-ready enterprise AI.   
---
Part of the [Quantum-AI Hybrid Cloud Framework](https://github.com/rasidi3112/Quantum-AI-Hybrid-Cloud-Framework)  
GitHub Repository | Author: [Ahmad Rasidi](https://github.com/rasidi3112)



    

   
    
       
 
