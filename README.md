# Matrix Operations Tool

A clean, modular, and interactive dual-interface **Matrix Operations Tool** built with Python and NumPy. This application is designed to perform standard matrix computations with robust user input validation and error handling. It features both a modern dark-themed Desktop GUI (built using `tkinter`) and a structured, menu-driven CLI.

*This project was developed as an **Internship Task** to demonstrate software engineering best practices, including interactive user experience, type safety, modular backend logic, and thorough test coverage.*

---

## Features

- **Dual-Interface Mode**: Launches a desktop Graphical User Interface (GUI) by default, or runs in a console-based Command Line Interface (CLI) when using the `--cli` flag.
- **Dynamic Dimension Binding (GUI)**:
  - Automatically syncs shapes for Matrix addition and subtraction.
  - Dynamically locks the row dimension of Matrix B to the column dimension of Matrix A for matrix multiplication.
  - Automatically locks dimensions to a square grid for determinant, inverse, and power calculations.
- **Convenient Preset Loaders (GUI)**:
  - **Identity**: Instantly generate an identity matrix.
  - **Random**: Fill the matrix grid with random integers between -9 and 9 for quick testing.
  - **Clear**: Clear all fields back to 0.
- **Formatted Outputs & Clipboard Sync**:
  - Structured mathematical layout using unicode delimiters (`┌ ┐`, `│ │`, `└ ┘`) in both GUI and CLI.
  - Copy result matrices or scalar outputs directly to your system clipboard with a single click.
- **Robust Exception Handling**: Prevents application crashes from malformed, empty, or non-numeric inputs, showing readable error alerts instead.
- **High-Quality Test Coverage**: Comprehensive suite of automated unit tests covering all core engine calculations and edge cases.

---

## Supported Matrix Operations

1. **Matrix Addition ($A + B$)**: Add two matrices of matching shapes element-by-element.
2. **Matrix Subtraction ($A - B$)**: Subtract Matrix B from Matrix A element-by-element.
3. **Matrix Multiplication ($A \times B$)**: Dot product of Matrix A and Matrix B.
4. **Matrix Transpose ($A^T$)**: Transpose Matrix A (swaps rows and columns).
5. **Matrix Determinant ($\det(A)$)**: Compute the determinant of a square matrix.
6. **Matrix Inverse ($A^{-1}$)**: Compute the inverse of a non-singular square matrix.
7. **Scalar Multiplication ($c \times A$)**: Multiply Matrix A by a real scalar number $c$.
8. **Matrix Power ($A^n$)**: Multiply square Matrix A by itself $n$ times (where $n$ is an integer).
9. **Matrix Rank ($\text{rank}(A)$)**: Computes the mathematical rank of Matrix A.

---

## Project Structure

```
MatrixOperations/
├── matrix_operations.py        # Core application (Mathematical Engine, CLI, and Tkinter GUI)
├── test_matrix_operations.py   # Automated Unit Test Suite (unittest framework)
├── requirements.txt            # Project dependencies (NumPy)
├── LICENSE                     # MIT License
└── README.md                   # Documentation
```

---

## Prerequisites

- **Python 3.x**
- **NumPy**

---

## Installation

1. Clone or copy the project files to your local machine.
2. Install the dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Graphical User Interface (GUI) - Default

Execute the program from your terminal to launch the desktop application:

```bash
python3 matrix_operations.py
```

*Note: If your system is headless or lacks Tkinter support, the program will automatically display a warning and fall back to the CLI mode.*

### 2. Command Line Interface (CLI)

Add the `--cli` argument to run the menu-driven command-line solver inside the terminal:

```bash
python3 matrix_operations.py --cli
```

---

## Running Unit Tests

To verify the mathematical engine and boundary validations, execute the unit test suite using the built-in `unittest` runner:

```bash
python3 -m unittest test_matrix_operations.py
```

Expected output:
```
Ran 17 tests in 0.020s

OK
```

---

## Sample CLI Usage

```
==================================================
          MATRIX OPERATIONS TOOL (CLI)
==================================================
1. Matrix Addition (A + B)
2. Matrix Subtraction (A - B)
3. Matrix Multiplication (A × B)
4. Matrix Transpose (Aᵀ)
5. Matrix Determinant (det(A))
6. Matrix Inverse (A⁻¹)
7. Scalar Multiplication (c × A)
8. Matrix Power (Aⁿ)
9. Matrix Rank
10. Exit
==================================================

Enter your choice (1-10): 4

Enter Matrix A
Enter number of rows: 2
Enter number of columns: 3

Enter each row separated by spaces
Row 1: 1 2 3
Row 2: 4 5 6

=============================================
 TRANSPOSE OF MATRIX A 
=============================================
┌  1  4  ┐
│  2  5  │
└  3  6  ┘
=============================================
```

---

## Internship Project Highlights

- **Clean Architectural Separation**: The calculations are abstracted into pure functions in a core engine section, isolated from display logic (CLI/GUI).
- **Defensive Programming**: Validates input dimensions for shape mismatches, filters out non-numeric values, and gracefully handles singular (non-invertible) matrix errors.
- **Production Grade Tooling**: Built-in unit testing suite ensures functional reliability and eases future enhancement integrations.
- **Usability Focus**: Features dynamic GUI dimensions (e.g. locking addition dimensions or multiplication constraints) which prevent users from executing invalid matrix math operations.
