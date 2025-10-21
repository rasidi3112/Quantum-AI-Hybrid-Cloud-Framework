## Models Module  
Quantum-AI Hybrid Cloud Framework
Modular, Reproducible, and Hardware-Aware Hybrid Quantum-Classical Architectures  


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)


---

## Overview  
The models/ module defines the core neural architectures of the Quantum-AI Hybrid Cloud Framework. It provides:  

- Classical feature extractors (fully configurable MLPs)  
- Parametrized quantum layers (PennyLane-based variational circuits)  
- End-to-end hybrid classifiers with skip connections and post-processing heads  


All components are:  

- Type-safe (using dataclass and type hints)  
- Hardware-aware (auto-adapts to CPU, MPS, CUDA, or QPU backends)  
- Reproducible (config-driven, seed-controlled)  
- Interoperable (compatible with PyTorch ecosystem, ONNX, TorchScript)  

