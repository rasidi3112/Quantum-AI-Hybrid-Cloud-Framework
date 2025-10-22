## Docker Deployment

Quantum-AI Hybrid Cloud Framework
Containerized, Reproducible, and Cloud-Native Inference for Hybrid Quantum-Classical Models



![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blueviolet.svg)


---

Overview  

The Dockerfile provides a lightweight, production-ready container image for deploying the Quantum-AI Hybrid Cloud Framework as a RESTful inference service. It enables:
  
  - Environment reproducibility across development, testing, and cloud environments  
  - Zero-install deployment on any Docker-compatible host (local, VM, Kubernetes, AWS ECS, etc.)  
  - Secure, isolated execution with minimal attack surface (python:3.11-slim base)  
  - Seamless integration with CI/CD pipelines and container orchestration platforms  

This image serves the FastAPI inference endpoint (/predict, /health) and is optimized for scalable quantum AI microservices.

## Dockerfile Structure
```bash
FROM python:3.11-slim AS base

# Environment hardening
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (for PennyLane/Qiskit compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency caching layer
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy full application
COPY . /app

EXPOSE 8000

# Launch FastAPI inference server
ENTRYPOINT ["python", "-m", "deployment.api"]
```
Best Practices Applied: 

- Layer caching: requirements.txt copied before code for faster rebuilds  
- Minimal base image: slim variant reduces image size (~200–300 MB)  
- Security: No root persistence, no cache, no byte compilation  
- Idempotent: Fully reproducible from source

## Quick Start  
  Build the Image  
  ```bash
docker build -t quantum-ai-hybrid .
```
Run the Container  
```bash
docker run -it -p 8000:8000 \
  -v $(pwd)/runs:/app/runs \
  quantum-ai-hybrid \
  --model-path runs/iris-experiment/checkpoints/best.pt \
  --device cpu
```
"*Mount checkpoints: Use -v to bind-mount your trained models into the container.*"  

Verify Service  
```bash
curl http://localhost:8000/health
# Response: {"status":"ok","backend":"default.qubit"}
```
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"samples": [[5.1, 3.5, 1.4, 0.2]]}'
```

## Cloud & Enterprise Deployment  
  Environment Variables (for Cloud QPUs)  
  Inject quantum provider credentials at runtime:
  ```bash
docker run -it -p 8000:8000 \
  -e QAI_IBM_TOKEN=your_ibm_token \
  -e QAI_RIGETTI_KEY=your_rigetti_key \
  quantum-ai-hybrid \
  --model-path ... \
  --backend ibmq_qasm_simulator
```
"*Security Note: Never hardcode secrets in the image. Always use runtime injection.*"  

## Kubernetes / Helm  
Compatible with standard Kubernetes deployments:
```bash
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: quantum-ai
        image: quantum-ai-hybrid:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: quantum-credentials
        args: ["--model-path", "/models/best.pt", "--device", "cpu"]
```
## Image Specifications
| Property           | Value                                                                 |
|-------------------|-----------------------------------------------------------------------|
| Base Image         | `python:3.11-slim`                                                    |
| Exposed Port       | 8000 (FastAPI)                                                        |
| Default Command    | `python -m deployment.api`                                            |
| Supported Backends | `default.qubit`, IBM QPU, Rigetti QVM (with credentials)             |
| Hardware Support   | CPU only (cloud QPUs via remote execution)                            |
| Image Size         | ~280 MB (compressed)                                                  |

  
  "*Note: GPU (CUDA) and MPS (Apple Silicon) are not supported in Docker by default. For GPU inference, use native deployment or NVIDIA Container Toolkit.*" 

---

## Integration with CI/CD  
The Docker image is automatically validated in .github/workflows/ci.yml:

   - docker build succeeds  
   - Container starts without error  
   - /health endpoint returns 200 OK
     
This ensures every commit is production-deployable.



---


## License

Test code is distributed under the [MIT License](https://github.com/rasidi3112/Quantum-AI-Hybrid-Cloud-Framework/blob/main/LICENSE), consistent with the rest of the framework.


---

## Philosophy

“Ship your quantum models like classical microservices.”  
*— Quantum-AI Hybrid Cloud Framework* 

 This Docker image bridges the gap between cutting-edge quantum research and enterprise-grade MLOps, enabling teams to deploy hybrid models with the same reliability as traditional AI services.


 ---


 Part of the [Quantum-AI Hybrid Cloud Framework]  (https://github.com/rasidi3112/Quantum-AI-Hybrid-Cloud-Framework)   
GitHub Repository | Author: [Ahmad Rasidi](https://github.com/rasidi3112)
