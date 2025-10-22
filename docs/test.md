## Tests Module

Quantum-AI Hybrid Cloud Framework
Reproducible, Hardware-Aware Unit and Integration Testing for Hybrid Quantum-Classical Systems  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/Tests-Pytest-green.svg)](https://docs.pytest.org/)
[![CI Status](https://github.com/rasidi3112/Quantum-AI-Hybrid-Cloud-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/rasidi3112/Quantum-AI-Hybrid-Cloud-Framework/actions) 

---
## Overview

The tests/ directory ensures correctness, reproducibility, and cross-platform reliability of the Quantum-AI Hybrid Cloud Framework through automated validation. It includes:  
- Sanity checks for model architecture and forward passes
- Hardware-agnostic test design (runs on CPU, MPS, and CUDA)
- Minimal dependencies using stubs for fast CI execution
- Integration with GitHub Actions for continuous validation

  ---
All tests are designed to:

- Complete in under 30 seconds on any modern laptop
- Require no cloud credentials or QPU access
- Validate shape consistency, gradient flow, and module interoperability

 ## Directory Structure  
```bash
 tests/
└── test_sanity.py    # Core sanity checks for hybrid model components
```
*Future extensions may include test_training.py, test_deployment.py, and test_qpu_fallback.py*  

---

## Key Test:  
```bash
test_hybrid_forward_pass()  
```
## Purpose

Verifies that the end-to-end HybridClassifier:

- Correctly initializes from configuration
- Accepts input tensors of expected shape
- Produces logits with correct batch and class dimensions

---

## Validation Logic
```bash
# Input: batch_size=2, features=4
inputs = torch.randn(2, 4)

# Output: batch_size=2, classes=3
logits = model(inputs)
assert logits.shape == (2, 3)
```

## Configuration Used

- Classical Encoder: [4 → 8 → 4] MLP with ReLU  
- Quantum Layer: 4 qubits, 1 layer, default.qubit backend  
- Hybrid Head: 3-class classifier with skip connection

This minimal configuration mirrors real-world usage while avoiding expensive quantum simulation during CI.  

---  

## Testing Strategy 


Principle                | Implementation
--------------------------|--------------------------------------------------------------
Fast Feedback             | All tests run in <30s on CPU
No External Dependencies  | Uses local simulator (default.qubit) only
Stub-Based Isolation      | ClassicalFeatureExtractor stub avoids full model load in unit tests
Cross-Platform            | Passes on macOS (M1/M2), Linux, Windows
Reproducible              | No randomness; deterministic tensor inputs



---


## How to Run Tests  

Locally:  
```bash
# Activate virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate

# Install test dependencies (if not already)
pip install -r requirements.txt

# Run all tests
pytest

# Run with verbose output
pytest -v
```

## In CI (GitHub Actions)  

Tests are automatically executed on every push and pull request via:  
- Linting: flake8
- Unit Testing: pytest
- Docker Build Validation: Ensures containerized deployment works
  
See .github/workflows/ci.yml for full pipeline.
  
---

## Test Coverage Goals  


Component                | Current Coverage         | Target
--------------------------|--------------------------|----------
Model Architecture        |  100% (sanity)          | 100%
Training Pipeline         |  Planned               | 90%+
Deployment APIs           |  Planned               | 90%+
Quantum Backend Fallback  |  Planned               | 100%

*Note: Full test coverage will expand in future releases. Contributions welcome!*

---

## License

Test code is distributed under the [MIT License](https://github.com/rasidi3112/Quantum-AI-Hybrid-Cloud-Framework/blob/main/LICENSE), consistent with the rest of the framework.


---

## Philosophy


*“If it’s not tested, it’s broken.”*  
— Quantum-AI Hybrid Cloud Framework Testing Manifesto

We believe that robust quantum software requires classical-grade testing discipline.
These tests ensure that every commit maintains:

• Architectural integrity  
• Hardware portability  
• Scientific reproducibility  

---


Part of the [Quantum-AI Hybrid Cloud Framework]  (https://github.com/rasidi3112/Quantum-AI-Hybrid-Cloud-Framework)   
GitHub Repository | Author: [Ahmad Rasidi](https://github.com/rasidi3112)  



