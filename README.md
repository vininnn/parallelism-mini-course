# Parallelism Mini-Course

## Overview

This repository explores parallel computing techniques across both **C++** (OpenMP, CUDA) and **Python** (Multiprocessing, Numba CPU, Numba CUDA). The project evaluates performance speedups by solving two computationally intensive tasks:

1. **Matrix Multiplication ($2048 \times 2048$)** in C++ / CUDA.
2. **Mandelbrot Fractal Generation ($2048 \times 1536$)** in Python.

> **Note on Acknowledgement:** The core algorithms and base templates in this project were provided as part of coursework material by my professor, with subsequent modifications, optimizations (such as matrix access pattern tuning and JIT warm-ups), bug fixes, and parallel extensions applied by me.

---

## Hardware & Environment Specifications

All benchmarks were conducted on a laptop running Linux with the following specifications:

* **GPU:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM)
* **CUDA Driver Version:** 591.91
* **CUDA Toolkit Version:** 13.1 / 12.4
* **Compiler Toolchains:** `g++` (GCC) with OpenMP support, `nvcc` (NVIDIA CUDA Compiler)

---

## Benchmarks & Performance Summary

*Note: Execution times are approximations based on average benchmark runs. Individual runtimes may vary slightly depending on system load and power limits.*

### 1. Matrix Multiplication ($2048 \times 2048$) — C++ & CUDA

| Implementation | Paradigm          | Approx. Execution Time |
| :------------- | :---------------- | :--------------------- |
| **Sequential** | C++ Single-Thread | ~4.27 seconds          |
| **OpenMP**     | C++ OpenMP        | ~1.26 seconds          |
| **CUDA**       | CUDA Kernel       | ~0.86 seconds          |

### 2. Mandelbrot Fractal Generation ($2048 \times 1536$) — Python

| Implementation      | Paradigm                              | Approx. Execution Time |
| :------------------ | :------------------------------------ | :--------------------- |
| **Sequential**      | Python Pure CPU                       | ~14.05 seconds         |
| **Multiprocessing** | Python `multiprocessing.Pool`         | ~4.22 seconds          |
| **Numba CPU**       | Parallel JIT (`@njit(parallel=True)`) | ~0.40 seconds          |
| **Numba CUDA**      | GPU Kernel JIT (`@cuda.jit`)          | **~0.057 seconds**     |

---

## Requirements & Setup

### Hardware Requirements

The CPU-based implementations can run without a CUDA-capable GPU.

> **Important:** The CUDA implementations (`matrixCuda.cu` and `fractal_cuda.py`) require a compatible **NVIDIA GPU** with up-to-date NVIDIA drivers installed. CUDA acceleration is not supported on non-NVIDIA hardware, such as AMD GPUs or Intel integrated graphics.

### System Dependencies

Make sure the required C++, Make, and CUDA toolchains are installed:

```bash
sudo apt update
sudo apt install build-essential nvidia-cuda-toolkit
```

> **Note:** The CUDA toolkit is only required if you intend to build or run the CUDA implementations.

### Python Environment & Dependencies

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Verify CUDA Installation

If you intend to use the CUDA implementations, verify that your NVIDIA GPU and CUDA compiler are properly installed:

```bash
nvidia-smi
nvcc --version
```

---

## How to Build & Run

### C++ Implementations

A `Makefile` is provided to automate compilation and keep build artifacts organized in the `bin/` directory.

**Build All C++ Implementations**

```bash
make
```

**Build a Specific Implementation**

```bash
make matrix                # Sequential C++
make matrixMultiprocessing # OpenMP Parallel C++
make matrixCuda            # CUDA GPU
```

**Run the Executables**

```bash
./bin/matrix
./bin/matrixMultiprocessing
./bin/matrixCuda
```

> **Note:** `matrixCuda` requires a compatible NVIDIA GPU and a working CUDA installation.

**Clean Build Artifacts**

To remove the compiled binaries from the `bin/` directory:

```bash
make clean
```

### Python Implementations

Make sure the Python virtual environment is activated before running the scripts:

```bash
source venv/bin/activate
```

The project includes the following implementations:

```bash
# Sequential Execution
python3 fractal_sequencial.py

# Multiprocessing Execution
python3 fractal_multiprocessing.py

# Numba CPU Parallel Execution
python3 fractal_numba.py

# Numba CUDA GPU Acceleration
python3 fractal_cuda.py
```

> **Note:** `fractal_cuda.py` requires a compatible NVIDIA GPU and a working CUDA installation.
