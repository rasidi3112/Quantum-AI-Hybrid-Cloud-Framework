# Quantum-AI Hybrid Cloud Framework

Quantum-AI Hybrid Cloud Framework adalah platform sumber terbuka berbasis Python untuk melatih model Artificial Intelligence (AI) yang menggabungkan komputasi klasik (CPU/GPU) dengan komputasi kuantum (simulator maupun QPU cloud). Framework ini dirancang modular, siap produksi, dan kompatibel lintas platform (macOS ARM64/M1, Linux, Windows).

## Fitur Utama

- **Hybrid Modeling**: Lapisan klasik (PyTorch) dan lapisan kuantum (PennyLane/Qiskit) dalam satu arsitektur.
- **Otomatisasi Backend**: Beralih ke simulator lokal ketika QPU cloud tidak tersedia.
- **Lifecycle Lengkap**: Training, hyperparameter tuning sederhana, logging, checkpoint, dan eksport model (TorchScript/ONNX).
- **Deployment Terintegrasi**: FastAPI untuk inference service dan Streamlit dashboard untuk visualisasi.
- **CI/CD**: Workflow GitHub Actions untuk linting, testing, dan build Docker.
- **Notebook Demo**: Visualisasi Bloch Sphere untuk lapisan kuantum.

## Persyaratan Sistem

- Python ≥ 3.11
- macOS (M1/M2) dengan `torch` MPS support, Linux, atau Windows (x86_64)
- GPU (opsional) dengan CUDA 11.8+ bila menggunakan akselerasi GPU
- Akses internet (opsional) untuk koneksi ke QPU cloud (IBM Quantum, Rigetti, D-Wave)

## Instalasi

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/quantum-ai-hybrid.git
   cd quantum-ai-hybrid