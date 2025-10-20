import os
import platform
import torch  # type: ignore
import pennylane as qml  # type: ignore
from dataclasses import dataclass

@dataclass
class QuantumBackendInfo:
    name: str
    is_hardware: bool
    provider: str
    shots: int | None

def get_quantum_device(backend, n_wires, shots=None):
    if backend is None:
        backend = os.getenv("QAI_DEFAULT_BACKEND", "default.qubit")

    backend_lower = backend.lower()

    # PennyLane default / simulator backends
    if backend_lower.startswith("default") or backend_lower.startswith("lightning"):
        try:
            device = qml.device(backend, wires=n_wires, shots=shots)
        except Exception:
            device = qml.device("default.qubit", wires=n_wires, shots=shots)
            backend = "default.qubit"
        info = QuantumBackendInfo(name=backend, is_hardware=False, provider="PennyLane", shots=shots)
        return device, info

    # IBM Qiskit backend
    elif backend_lower.startswith("ibm"):
        from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore
        token = os.getenv("QAI_IBM_TOKEN")
        if token:
            service = QiskitRuntimeService(channel="cloud", token=token)
            backend_name = backend.split(":", 1)[1] if ":" in backend else backend
            device = qml.device("qiskit.ibmq", wires=n_wires, backend=backend_name, shots=shots or 1024, service=service)
            info = QuantumBackendInfo(name=backend_name, is_hardware=True, provider="IBM", shots=shots or 1024)
            return device, info
        else:
            device = qml.device("default.qubit", wires=n_wires, shots=shots)
            info = QuantumBackendInfo(name="default.qubit", is_hardware=False, provider="Fallback", shots=shots)
            return device, info

    # Rigetti Forest / Quil backend
    elif backend_lower.startswith("rigetti"):
        try:
            device = qml.device("forest.qvm", wires=n_wires, shots=shots or 1024)
            info = QuantumBackendInfo(name="rigetti.qvm", is_hardware=False, provider="Rigetti", shots=shots or 1024)
            return device, info
        except Exception:
            device = qml.device("default.qubit", wires=n_wires, shots=shots)
            info = QuantumBackendInfo(name="default.qubit", is_hardware=False, provider="Fallback", shots=shots)
            return device, info

    # Fallback untuk backend tidak dikenal
    else:
        device = qml.device("default.qubit", wires=n_wires, shots=shots)
        info = QuantumBackendInfo(name="default.qubit", is_hardware=False, provider="Fallback", shots=shots)
        return device, info


def summarize_hardware() -> None:
    """
    Print a summary of the current hardware (CPU, GPU, Quantum backend)
    """
    print("=== Hardware Summary ===")
    
    # Platform
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # CPU info
    print(f"CPU cores: {os.cpu_count()}")
    print(f"PyTorch threads: {torch.get_num_threads()}")

    # GPU info
    if torch.cuda.is_available():
        print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA devices count: {torch.cuda.device_count()}")
    else:
        print("No CUDA GPU detected")

    # PennyLane info
    print(f"PennyLane version: {qml.__version__}")
    print("========================")
