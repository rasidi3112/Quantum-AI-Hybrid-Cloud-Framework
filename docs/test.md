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







