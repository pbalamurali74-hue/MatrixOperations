# 🧮 Matrix Operations Tool

Interactive Python application for performing common **matrix computations** through both a desktop GUI and command-line interface.

## Supported operations
- Addition and subtraction
- Matrix multiplication
- Transpose
- Determinant
- Inverse
- Scalar multiplication
- Matrix power
- Matrix rank

## Highlights
- Tkinter desktop GUI.
- CLI mode with `--cli`.
- Dynamic dimension validation.
- Identity/random matrix presets.
- Clipboard-ready formatted results.
- Defensive error handling.
- Automated unit tests.

## Tech Stack
Python • NumPy • Tkinter • unittest

## Run
```bash
pip install -r requirements.txt
python3 matrix_operations.py
```

CLI:
```bash
python3 matrix_operations.py --cli
```

Tests:
```bash
python3 -m unittest test_matrix_operations.py
```

> A compact software-engineering project demonstrating numerical computing, modular design, input validation and testing.