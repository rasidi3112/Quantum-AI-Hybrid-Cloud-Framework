
# Quantum-AI Hybrid Cloud Framework

**Quantum-AI Hybrid Cloud Framework** is an advanced modular platform for **hybrid classical–quantum AI** model development, training, and deployment. It integrates **PyTorch** for classical neural networks and **PennyLane/Qiskit** for quantum layers, supporting both **local simulators** and **cloud QPUs** (IBM, Rigetti, D-Wave). Fully **cross-platform** (macOS, Linux, Windows), designed for reproducible research and enterprise applications.

## Key Features

- **Hybrid Modeling**  
  Combine classical neural networks with variational quantum circuits in a modular architecture.

- **Cross-Device Compatibility**  
  Fully portable across laptops, CPUs, GPUs, and cloud QPUs for flexible development and deployment.

- **Backend Auto-Detection**  
  Automatically switches between CPU, MPS (macOS), CUDA, or QPU depending on available hardware.

- **Configurable Pipelines**  
  Modular configuration via **YAML** or **CLI** for easy experimentation and adaptation.

- **Integrated Logging & Checkpointing**  
  Automatically saves models, metrics, and artifacts to ensure reproducibility.

- **Cloud-Ready Deployment**  
  FastAPI endpoints and Streamlit dashboards ready for inference and monitoring.

- **Continuous Integration Support**  
  Preconfigured for GitHub Actions, Docker, and testing workflows for professional development.

## Purpose

Provide a **professional, reproducible hybrid AI ecosystem** ready for integration in **global research or industry projects**, bridging **quantum computing** with **classical deep learning**.

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
#### Dashboard Screenshots

**1. Overview: Training Metrics**
![Dashboard Overview 1](assets/dashboard_overview1.png)  
*Real-time visualization of training metrics (loss & accuracy) across epochs. On Mac Apple Silicon, the MPS device is automatically used if selected.*

![Dashboard Overview 2](assets/dashboard_overview2.png)  
*Detailed metrics and logs. Device selection (CPU/MPS/CUDA) ensures consistent cross-platform performance.*

**2. Inference Playground**
![Inference Playground](assets/inference_playground.png)  
*Interactive input interface for predictions. Choose device according to your hardware for optimal inference speed.*

**3. Bloch Sphere Visualizer**
![Bloch Sphere](assets/bloch_sphere.png)  
*Quantum state visualization using a Bloch Sphere. Accelerated on MPS (Apple Silicon) or CUDA (NVIDIA GPU).*

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
├─ .github/workflows/   # CI/CD pipelines: automated testing, linting, Docker build
├─ deployment/          # FastAPI services and Streamlit dashboard scripts
├─ examples/            # Sample datasets, notebooks, and usage examples
├─ models/              # Hybrid and quantum model definitions (PyTorch + PennyLane/Qiskit)
├─ training/            # Training pipeline scripts and utilities
├─ utils/               # Logging, checkpointing, device utilities, and helper functions
├─ tests/               # Unit and integration tests for reproducibility
├─ runs/                # Experiment outputs: checkpoints, metrics, and visualizations
└─ requirements.txt     # Python dependencies for reproducible environment

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

  
