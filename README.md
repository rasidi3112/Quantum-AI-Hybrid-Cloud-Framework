## My project is still under development☺️

# Quantum-AI Hybrid Cloud Framework

Quantum-AI Hybrid Cloud Framework is an advanced modular platform for hybrid classical–quantum AI model development, training, and deployment. It integrates PyTorch for classical neural networks and PennyLane/Qiskit for quantum layers, with support for both local simulators and cloud-based QPUs from IBM, Rigetti, and D-Wave. The framework is fully cross-platform (macOS, Linux, Windows) and designed for reproducible international research and enterprise applications.

---

## Features

- **Hybrid Modeling**: Combine classical neural networks with variational quantum circuits.
- **Cross-Device Compatibility**: Fully portable across laptops, CPUs, GPUs, and QPUs.
- **Backend Auto-Detection**: Automatically switches between CPU, MPS (macOS), CUDA, or QPU.
- **Configurable Pipelines**: Modular configuration via YAML or CLI.
- **Integrated Logging & Checkpointing**: Automatic saving of models, metrics, and artifacts.
- **Cloud-Ready Deployment**: FastAPI endpoints and Streamlit dashboards.
- **Continuous Integration Support**: Preconfigured for GitHub Actions, Docker, and testing.

---

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/rasidi3112/Quantum-AI-Hybrid-Cloud-Framework.git
cd Quantum-AI-Hybrid-Cloud-Framework

# Create a new Python virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: verify installation
pytest

```

## 2. Running on Another Laptop or Device
If you want to use the same project on another machine:
1. Clone the repository from GitHub on the new device:
 ```bash
   git clone https://github.com/rasidi3112/Quantum-AI-Hybrid-Cloud-Framework.git
   cd Quantum-AI-Hybrid-Cloud-Framework
```
2. Recreate the virtual environment:
 ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate

```
3. Install dependencies using the same requirements.txt:
    ```bash
    pip install -r requirements.txt
   ```

4. Restore saved configurations and checkpoints (if you synced them via Git):
```bash
   cp -r runs/ /path/to/new/environment/
```
5. Ensure device compatibility:

  - macOS (M1/M2 / Apple silicon ): use --device mps
  - Windows/Linux (CPU): use --device cpu
  - GPU (NVIDIA): use --device cuda

6. Confirm reproducibility:
   ```bash
   python -m qml_app.main evaluate --model vqc --config config/default.yaml
``

## 3. Quick Start Training
 ```bash
   python -m qml_app.main train \
       --model vqc \
       --dataset examples/iris.csv \
       --backend default.qubit \
       --epochs 50 \
       --batch-size 16 \
       --device cpu
```
Output:
- runs/<experiment_name>/checkpoints/best.pt
- runs/<experiment_name>/metrics.json
- Training plots (loss, accuracy)

 ### Training Metrics Visualization

After running the training command, the framework automatically generates `training_metrics.html` under the `runs/<experiment_name>/` directory.  
This file provides a real-time visualization of loss and accuracy curves during training.

Example output:

![Training Metrics Screenshot](assets/training_metrics.png)

To train on a QPU:
```bash
python -m qml_app.main train \
    --model hybrid \
    --backend ibmq_qasm_simulator \
    --use-qpu True
```
## 4. Evaluation and Inference
```bash
   python -m qml_app.main evaluate \
    --model hybrid \
    --config config/default.yaml \
    --checkpoint runs/iris/checkpoints/best.pt
```
The evaluation script automatically detects available devices and logs performance metrics to runs/<experiment>/metrics.json.

## 5. Deployment (FastAPI & Streamlit) 
   FastAPI REST Service
   ```bash
python -m qml_app.main serve \
    --model-path runs/iris/checkpoints/best.pt \
    --device cpu \
    --host 0.0.0.0 --port 8000
```
   Endpoints:
   -GET /health → { "status": "ok", "backend": "default.qubit" }
   -POST /predict → send JSON samples and receive predictions.
   Example:
 ```bash
{
  "samples": [[5.1, 3.5, 1.4, 0.2]]
}
```

   Response:  
   ```
    {
"predictions": [0],
  "probabilities": [[0.95, 0.03, 0.02]],
  "backend": "default.qubit"
}

```

Streamlit Dashboard:

 ```bash
streamlit run deployment/dashboard.py

```
### Dashboard Screenshots

**Quantum-AI Hybrid Dashboard – Overview**
![Dashboard Screenshot 1](assets/dashboard_overview1.png)  
*Shows real-time training metrics and accuracy/loss curves. On macOS Apple Silicon, the dashboard runs optimally using MPS device.*

![Dashboard Screenshot 2](assets/dashboard_overview2.png)  
*Additional metrics visualization and training logs. MPS on Apple Silicon ensures GPU-like performance.*

**Inference Playground**
![Inference Playground](assets/inference_playground.png)  
*Interactive input for predictions using the hybrid model. Predictions are computed on MPS for Apple Silicon.*

**Bloch Sphere Visualizer**
![Bloch Sphere](assets/bloch_sphere.png)  
*Visualizing quantum states on the Bloch sphere. MPS device provides accelerated computation on Apple M1/M2.*

This dashboard provides:
- Real-time training metrics visualization.
- Quantum state plots (Bloch Sphere).
- Model inference playground.
  
## 6. Export and Integration
 ```bash
   # Export to TorchScript
   python -m qml_app.export \
     --checkpoint runs/iris/checkpoints/best.pt \
     --torchscript exports/model.pt
   
   # Export to ONNX
   python -m qml_app.export \
     --checkpoint runs/iris/checkpoints/best.pt \
     --onnx exports/model.onnx
```
  These exported models can be used for deployment in other frameworks or cloud environments.

  ## 7. Docker Support
  ```bash
   # Build the container
   docker build -t quantum-ai-hybrid .
   
   # Run the container
   docker run -it -p 8000:8000 quantum-ai-hybrid

The Docker image ensures consistent runtime environments across machines.
```
 ## 8. GitHub Actions are configured for:
- Linting (flake8)
- Unit tests (pytest)
- Automatic Docker build validation

 ## 9. Directory Structure
 ```bash
Quantum-AI-Hybrid-Cloud-Framework/
├─ config/                # YAML configs for models and backends
├─ deployment/            # FastAPI + Streamlit deployment scripts
├─ examples/              # Datasets and notebooks
├─ models/                # Hybrid and quantum model definitions
├─ training/              # Training pipelines
├─ utils/                 # Logging, checkpointing, and device utilities
├─ tests/                 # Unit and integration tests
├─ runs/                  # Experiment outputs
└─ requirements.txt
```
 ## 10. Cloud Integration
To connect to a cloud quantum provider, export your credentials:
  ```bash
# IBM Quantum
export QAI_IBM_TOKEN=<your_ibm_token>

# Rigetti
export QAI_RIGETTI_KEY=<your_rigetti_key>

# D-Wave
export QAI_DWAVE_KEY=<your_dwave_key>
 ```
The framework automatically falls back to the local simulator if credentials are not provided.

 ## 11. Multi-Device Synchronization (Best Practice)

- Keep all configurations (config/), checkpoints (runs/), and requirements.txt versioned in Git.
- Always recreate virtual environments per device.
- Use consistent Python version (>=3.11).
- To ensure reproducibility, set random seeds via config/training.yaml or CLI flag --seed.
- For Apple Silicon users, prefer --device mps over CPU for optimal performance.

## 12. License
   This project is licensed under the MIT License, allowing modification and commercial use with proper attribution.

  
