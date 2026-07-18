import numpy as np
import sys
import random

# GUI imports wrapped in try/except to allow CLI mode in headless/no-tkinter environments
GUI_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import messagebox
    GUI_AVAILABLE = True
except ImportError:
    pass


# -----------------------------------------------------------------------------
# 1. CORE ENGINE MATHEMATICAL FUNCTIONS
# -----------------------------------------------------------------------------

def add_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Adds two matrices of matching shapes element-by-element.

    Args:
        A (np.ndarray): The first input matrix.
        B (np.ndarray): The second input matrix.

    Returns:
        np.ndarray: The element-wise sum of matrices A and B.

    Raises:
        ValueError: If A and B do not have the same shape.
    """
    if A.shape != B.shape:
        raise ValueError(f"Matrix addition shape mismatch: A {A.shape} and B {B.shape} must be identical.")
    return A + B


def subtract_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Subtracts the second matrix from the first matrix element-by-element.

    Args:
        A (np.ndarray): The matrix to subtract from.
        B (np.ndarray): The matrix to subtract.

    Returns:
        np.ndarray: The element-wise difference of A and B (A - B).

    Raises:
        ValueError: If A and B do not have the same shape.
    """
    if A.shape != B.shape:
        raise ValueError(f"Matrix subtraction shape mismatch: A {A.shape} and B {B.shape} must be identical.")
    return A - B


def multiply_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the dot product of two matrices (A * B).

    Args:
        A (np.ndarray): Left-hand matrix.
        B (np.ndarray): Right-hand matrix.

    Returns:
        np.ndarray: The matrix product of A and B.

    Raises:
        ValueError: If the number of columns in A does not match the number of rows in B.
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Matrix multiplication shape mismatch: Columns of A ({A.shape[1]}) must match rows of B ({B.shape[0]}).")
    return np.dot(A, B)


def transpose_matrix(A: np.ndarray) -> np.ndarray:
    """Computes the transpose of the given matrix.

    Args:
        A (np.ndarray): Input matrix.

    Returns:
        np.ndarray: Transposed matrix (swapped rows and columns).
    """
    return A.T


def determinant_matrix(A: np.ndarray) -> float:
    """Computes the determinant of a square matrix.

    Args:
        A (np.ndarray): Input square matrix.

    Returns:
        float: Determinant of the matrix.

    Raises:
        ValueError: If the input matrix is not square.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Determinant requires a square matrix. Current shape is {A.shape}.")
    if A.shape == (1, 1):
        return float(A[0, 0])
    return float(np.linalg.det(A))


def inverse_matrix(A: np.ndarray) -> np.ndarray:
    """Computes the multiplicative inverse of a non-singular square matrix.

    Args:
        A (np.ndarray): Input square matrix.

    Returns:
        np.ndarray: Inverse of the matrix A.

    Raises:
        ValueError: If the matrix is not square or is singular (not invertible).
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix inversion requires a square matrix. Current shape is {A.shape}.")
    try:
        det = np.linalg.det(A)
        if np.abs(det) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted (determinant is zero).")
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular and cannot be inverted.")


def scalar_multiply(A: np.ndarray, scalar: float) -> np.ndarray:
    """Multiplies every element of a matrix by a real scalar number.

    Args:
        A (np.ndarray): Input matrix.
        scalar (float): Real number scalar multiplier.

    Returns:
        np.ndarray: Scaled matrix.

    Raises:
        ValueError: If the scalar cannot be parsed as a float.
    """
    try:
        val = float(scalar)
    except (ValueError, TypeError):
        raise ValueError("Scalar value must be a valid real number.")
    return A * val


def matrix_power(A: np.ndarray, power: int) -> np.ndarray:
    """Raises a square matrix to an integer power.

    Args:
        A (np.ndarray): Input square matrix.
        power (int): Integer exponent.

    Returns:
        np.ndarray: Matrix raised to the given power.

    Raises:
        ValueError: If the matrix is not square or exponent is not an integer.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix power requires a square matrix. Current shape is {A.shape}.")
    try:
        p = int(power)
    except (ValueError, TypeError):
        raise ValueError("Power exponent must be an integer.")
    return np.linalg.matrix_power(A, p)


def matrix_rank(A: np.ndarray) -> int:
    """Computes the rank of a matrix using singular value decomposition.

    Args:
        A (np.ndarray): Input matrix.

    Returns:
        int: Rank of the matrix.
    """
    return int(np.linalg.matrix_rank(A))


# -----------------------------------------------------------------------------
# 2. CLI HELPER FUNCTIONS & RUNNER
# -----------------------------------------------------------------------------

def format_matrix_to_string(matrix):
    """Formats a 1D or 2D NumPy array into an aligned matrix with nice brackets."""
    if not isinstance(matrix, np.ndarray):
        return str(matrix)
    
    if len(matrix.shape) == 1:
        matrix = matrix.reshape(1, -1)
    elif len(matrix.shape) == 0:
        return f"{matrix.item():.4g}"
        
    if matrix.size == 0:
        return "[]"

    # Convert all elements to string, formatting floats to integers when whole numbers
    formatted = []
    for r in matrix:
        formatted_row = []
        for val in r:
            if np.isnan(val):
                formatted_row.append("NaN")
            elif np.isinf(val):
                formatted_row.append("∞" if val > 0 else "-∞")
            elif float(val).is_integer():
                formatted_row.append(f"{int(val)}")
            else:
                # Use general format with up to 4 significant digits
                formatted_row.append(f"{val:.4g}")
        formatted.append(formatted_row)
    
    # Calculate column widths
    cols = matrix.shape[1]
    col_widths = [max(len(formatted[r][c]) for r in range(matrix.shape[0])) for c in range(cols)]
    
    # Create lines
    lines = []
    for row in formatted:
        row_str = "  ".join(val.rjust(w) for val, w in zip(row, col_widths))
        lines.append(f"  {row_str}  ")
    
    if len(lines) == 1:
        return f"[ {lines[0].strip()} ]"
    
    # Construct vertical brackets
    result_lines = []
    for i, line in enumerate(lines):
        if i == 0:
            result_lines.append("┌" + line + "┐")
        elif i == len(lines) - 1:
            result_lines.append("└" + line + "┘")
        else:
            result_lines.append("│" + line + "│")
            
    return "\n".join(result_lines)


def display_matrix(title, matrix):
    print("\n" + "=" * 45)
    print(f" {title.upper()} ")
    print("=" * 45)
    print(format_matrix_to_string(matrix))
    print("=" * 45)


def input_matrix(name):
    print(f"\nEnter Matrix {name}")
    while True:
        try:
            rows = int(input("Enter number of rows: "))
            if rows <= 0:
                print("Number of rows must be a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input! Please enter a valid integer.")

    while True:
        try:
            cols = int(input("Enter number of columns: "))
            if cols <= 0:
                print("Number of columns must be a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input! Please enter a valid integer.")

    matrix = []
    print("\nEnter each row separated by spaces")
    for i in range(rows):
        while True:
            try:
                line = input(f"Row {i+1}: ").strip()
                row = list(map(float, line.split()))
                if len(row) == cols:
                    matrix.append(row)
                    break
                else:
                    print(f"Please enter exactly {cols} values.")
            except ValueError:
                print("Invalid input! Please enter only space-separated numbers.")

    return np.array(matrix)


def run_cli():
    while True:
        print("\n" + "=" * 50)
        print("          MATRIX OPERATIONS TOOL (CLI)")
        print("=" * 50)
        print("1. Matrix Addition (A + B)")
        print("2. Matrix Subtraction (A - B)")
        print("3. Matrix Multiplication (A × B)")
        print("4. Matrix Transpose (Aᵀ)")
        print("5. Matrix Determinant (det(A))")
        print("6. Matrix Inverse (A⁻¹)")
        print("7. Scalar Multiplication (c × A)")
        print("8. Matrix Power (Aⁿ)")
        print("9. Matrix Rank")
        print("10. Exit")
        print("=" * 50)

        choice = input("\nEnter your choice (1-10): ").strip()

        if choice == "10":
            print("\nThank you for using Matrix Operations Tool!")
            break

        try:
            if choice == "1":
                A = input_matrix("A")
                B = input_matrix("B")
                res = add_matrices(A, B)
                display_matrix("Addition Result (A + B)", res)
            elif choice == "2":
                A = input_matrix("A")
                B = input_matrix("B")
                res = subtract_matrices(A, B)
                display_matrix("Subtraction Result (A - B)", res)
            elif choice == "3":
                A = input_matrix("A")
                B = input_matrix("B")
                res = multiply_matrices(A, B)
                display_matrix("Multiplication Result (A × B)", res)
            elif choice == "4":
                A = input_matrix("A")
                res = transpose_matrix(A)
                display_matrix("Transpose of Matrix A", res)
            elif choice == "5":
                A = input_matrix("A")
                res = determinant_matrix(A)
                print(f"\n=============================================")
                print(f" DETERMINANT: {res:.6g}")
                print(f"=============================================")
            elif choice == "6":
                A = input_matrix("A")
                res = inverse_matrix(A)
                display_matrix("Inverse of Matrix A", res)
            elif choice == "7":
                A = input_matrix("A")
                while True:
                    try:
                        scalar_input = input("Enter scalar value: ").strip()
                        scalar = float(scalar_input)
                        break
                    except ValueError:
                        print("Invalid input! Please enter a valid number.")
                res = scalar_multiply(A, scalar)
                display_matrix(f"Scalar Multiplication ({scalar} × A)", res)
            elif choice == "8":
                A = input_matrix("A")
                while True:
                    try:
                        power_input = input("Enter integer power: ").strip()
                        power = int(power_input)
                        break
                    except ValueError:
                        print("Invalid input! Please enter a valid integer.")
                res = matrix_power(A, power)
                display_matrix(f"Matrix Power (A^{power})", res)
            elif choice == "9":
                A = input_matrix("A")
                res = matrix_rank(A)
                print(f"\n=============================================")
                print(f" MATRIX RANK: {res}")
                print(f"=============================================")
            else:
                print("\nInvalid Choice! Please enter a number between 1 and 10.")
        except Exception as e:
            print(f"\nError: {e}")


# -----------------------------------------------------------------------------
# 3. INTERACTIVE GRAPHICAL USER INTERFACE (GUI)
# -----------------------------------------------------------------------------

if GUI_AVAILABLE:
    class MatrixApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Matrix Operations Tool")
            self.geometry("1080x720")
            self.configure(bg="#121212")
            self.minsize(1000, 680)
            
            # Setup State Variables
            self.active_op = tk.StringVar(value="Add")
            self.rows_A = tk.IntVar(value=3)
            self.cols_A = tk.IntVar(value=3)
            self.rows_B = tk.IntVar(value=3)
            self.cols_B = tk.IntVar(value=3)
            
            self.scalar_val = tk.StringVar(value="2")
            self.power_val = tk.StringVar(value="2")
            
            self.entries_A = []
            self.entries_B = []
            self.last_result_str = ""
            
            self.create_layout()
            self.on_op_change()

        def create_layout(self):
            # -----------------------------
            # Left Sidebar Pane
            # -----------------------------
            sidebar = tk.Frame(self, bg="#1e1e1e", width=260)
            sidebar.pack(side="left", fill="y")
            sidebar.pack_propagate(False)
            
            # Logo/Header
            logo_frame = tk.Frame(sidebar, bg="#1e1e1e", pady=25)
            logo_frame.pack(fill="x")
            
            logo_lbl = tk.Label(logo_frame, text="🔢 MATRIX OP", font=("Helvetica", 18, "bold"), fg="#00adb5", bg="#1e1e1e")
            logo_lbl.pack()
            sub_lbl = tk.Label(logo_frame, text="Interactive Solver", font=("Helvetica", 9, "italic"), fg="#777777", bg="#1e1e1e")
            sub_lbl.pack()
            
            divider = tk.Frame(sidebar, bg="#333333", height=1)
            divider.pack(fill="x", padx=15, pady=5)
            
            # Operations list
            ops = [
                ("Matrix Addition", "Add"),
                ("Matrix Subtraction", "Sub"),
                ("Matrix Multiplication", "Mult"),
                ("Matrix Transpose", "Transpose"),
                ("Matrix Determinant", "Det"),
                ("Matrix Inverse", "Inverse"),
                ("Scalar Multiply", "Scalar"),
                ("Matrix Power", "Power"),
                ("Matrix Rank", "Rank"),
            ]
            
            self.op_buttons = {}
            for label, op_code in ops:
                btn = tk.Button(
                    sidebar,
                    text=f"  {label}",
                    anchor="w",
                    font=("Helvetica", 11),
                    bg="#1e1e1e",
                    fg="#dddddd",
                    activebackground="#2a2a2a",
                    activeforeground="#00adb5",
                    bd=0,
                    relief="flat",
                    height=2,
                    command=lambda oc=op_code: self.select_op(oc)
                )
                btn.pack(fill="x", padx=10, pady=2)
                # Bind animations
                btn.bind("<Enter>", lambda e, b=btn: self.on_button_hover(b, True))
                btn.bind("<Leave>", lambda e, b=btn: self.on_button_hover(b, False))
                self.op_buttons[op_code] = btn

            # Sidebar footer with version info
            footer = tk.Label(sidebar, text="NumPy Engine v2.5\nBuilt with Python & Tkinter", font=("Helvetica", 8), fg="#555555", bg="#1e1e1e")
            footer.pack(side="bottom", pady=15)

            # -----------------------------
            # Right Workspace Pane
            # -----------------------------
            self.workspace = tk.Frame(self, bg="#121212", padx=20, pady=20)
            self.workspace.pack(side="right", fill="both", expand=True)

            # Header Panel
            self.header_frame = tk.Frame(self.workspace, bg="#121212")
            self.header_frame.pack(fill="x", pady=(0, 15))
            
            self.op_title_lbl = tk.Label(self.header_frame, text="Matrix Addition", font=("Helvetica", 20, "bold"), fg="#ffffff", bg="#121212")
            self.op_title_lbl.pack(anchor="w")
            
            self.op_desc_lbl = tk.Label(
                self.header_frame,
                text="Adds two matrices of the same shape element-by-element.",
                font=("Helvetica", 10),
                fg="#888888",
                bg="#121212",
                wraplength=700,
                justify="left"
            )
            self.op_desc_lbl.pack(anchor="w", pady=(2, 0))

            # Config Section (Scalar and Power inputs)
            self.config_panel = tk.Frame(self.workspace, bg="#121212")
            self.config_panel.pack(fill="x", pady=5)
            
            # Scalar Frame
            self.scalar_frame = tk.Frame(self.config_panel, bg="#1a1a1a", padx=15, pady=10)
            tk.Label(self.scalar_frame, text="Scalar multiplier (c):", font=("Helvetica", 10), fg="#ffffff", bg="#1a1a1a").pack(side="left", padx=5)
            self.scalar_entry = tk.Entry(self.scalar_frame, textvariable=self.scalar_val, width=8, font=("Consolas", 11), bg="#262626", fg="#ffffff", bd=0, justify="center")
            self.scalar_entry.pack(side="left", padx=5)
            
            # Power Frame
            self.power_frame = tk.Frame(self.config_panel, bg="#1a1a1a", padx=15, pady=10)
            tk.Label(self.power_frame, text="Integer power exponent (n):", font=("Helvetica", 10), fg="#ffffff", bg="#1a1a1a").pack(side="left", padx=5)
            self.power_entry = tk.Entry(self.power_frame, textvariable=self.power_val, width=8, font=("Consolas", 11), bg="#262626", fg="#ffffff", bd=0, justify="center")
            self.power_entry.pack(side="left", padx=5)

            # Container for Matrix Grids
            self.grid_container = tk.Frame(self.workspace, bg="#121212")
            self.grid_container.pack(fill="both", expand=True, pady=10)

            # --- Matrix A Container ---
            self.matrix_a_frame = tk.Frame(self.grid_container, bg="#1a1a1a", padx=15, pady=15)
            self.matrix_a_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
            
            lbl_a_title = tk.Label(self.matrix_a_frame, text="Matrix A", font=("Helvetica", 13, "bold"), fg="#00adb5", bg="#1a1a1a")
            lbl_a_title.pack(anchor="w", pady=(0, 10))
            
            self.dims_frame_A = tk.Frame(self.matrix_a_frame, bg="#1a1a1a")
            self.dims_frame_A.pack(anchor="w", fill="x", pady=5)
            
            self.row_ctrl_A = self.make_dim_control(self.dims_frame_A, "Rows:", self.rows_A, self.update_matrix_a_grid)
            self.row_ctrl_A.pack(side="left", padx=5)
            
            self.col_ctrl_A = self.make_dim_control(self.dims_frame_A, "Cols:", self.cols_A, self.update_matrix_a_grid)
            self.col_ctrl_A.pack(side="left", padx=5)

            self.grid_frame_A = tk.Frame(self.matrix_a_frame, bg="#1a1a1a")
            self.grid_frame_A.pack(fill="both", expand=True, pady=10)
            
            # Preset panel A
            presets_A = tk.Frame(self.matrix_a_frame, bg="#1a1a1a")
            presets_A.pack(fill="x", side="bottom")
            self.make_preset_buttons(presets_A, self.entries_A, self.rows_A, self.cols_A)

            # --- Matrix B Container ---
            self.matrix_b_frame = tk.Frame(self.grid_container, bg="#1a1a1a", padx=15, pady=15)
            self.matrix_b_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
            
            lbl_b_title = tk.Label(self.matrix_b_frame, text="Matrix B", font=("Helvetica", 13, "bold"), fg="#00adb5", bg="#1a1a1a")
            lbl_b_title.pack(anchor="w", pady=(0, 10))
            
            # Label shown when B is synced to A
            self.lbl_b_synced = tk.Label(self.matrix_b_frame, text="Dimensions synced to Matrix A", font=("Helvetica", 9, "italic"), fg="#888888", bg="#1a1a1a")
            
            self.dims_frame_B_controls = tk.Frame(self.matrix_b_frame, bg="#1a1a1a")
            self.row_ctrl_B = self.make_dim_control(self.dims_frame_B_controls, "Rows:", self.rows_B, self.update_matrix_b_grid)
            self.col_ctrl_B = self.make_dim_control(self.dims_frame_B_controls, "Cols:", self.cols_B, self.update_matrix_b_grid)

            self.grid_frame_B = tk.Frame(self.matrix_b_frame, bg="#1a1a1a")
            self.grid_frame_B.pack(fill="both", expand=True, pady=10)
            
            # Preset panel B
            presets_B = tk.Frame(self.matrix_b_frame, bg="#1a1a1a")
            presets_B.pack(fill="x", side="bottom")
            self.make_preset_buttons(presets_B, self.entries_B, self.rows_B, self.cols_B)

            # --- Calculation trigger ---
            self.action_frame = tk.Frame(self.workspace, bg="#121212")
            self.action_frame.pack(fill="x", pady=15)
            
            self.calc_btn = tk.Button(
                self.action_frame,
                text="CALCULATE RESULT",
                font=("Helvetica", 12, "bold"),
                bg="#00adb5",
                fg="#ffffff",
                activebackground="#00cbd3",
                activeforeground="#ffffff",
                bd=0,
                padx=20,
                pady=8,
                relief="flat",
                command=self.execute_calculation
            )
            self.calc_btn.pack(side="left")
            self.calc_btn.bind("<Enter>", lambda e: self.calc_btn.config(bg="#00cbd3"))
            self.calc_btn.bind("<Leave>", lambda e: self.calc_btn.config(bg="#00adb5"))

            # --- Results Container ---
            self.result_container = tk.Frame(self.workspace, bg="#1a1a1a", padx=15, pady=15)
            self.result_container.pack(fill="both", expand=True, pady=(10, 0))
            self.result_container.pack_forget() # Hidden initially

        def make_dim_control(self, parent, label_text, var, callback):
            frame = tk.Frame(parent, bg="#1a1a1a")
            
            lbl = tk.Label(frame, text=label_text, font=("Helvetica", 10), fg="#aaaaaa", bg="#1a1a1a", anchor="w")
            lbl.pack(side="left", padx=5)
            
            def dec():
                val = var.get()
                if val > 1:
                    var.set(val - 1)
                    callback()
                        
            def inc():
                val = var.get()
                if val < 10: # Cap at 10 to keep GUI clean
                    var.set(val + 1)
                    callback()
                        
            btn_dec = tk.Button(frame, text="-", font=("Consolas", 10, "bold"), width=3, bg="#333333", fg="#ffffff",
                                activebackground="#444444", activeforeground="#ffffff", bd=0, relief="flat", command=dec)
            btn_dec.pack(side="left", padx=2)
            btn_dec.bind("<Enter>", lambda e: btn_dec.config(bg="#444444"))
            btn_dec.bind("<Leave>", lambda e: btn_dec.config(bg="#333333"))
            
            val_lbl = tk.Label(frame, textvariable=var, font=("Helvetica", 11, "bold"), width=3, fg="#ffffff", bg="#1a1a1a")
            val_lbl.pack(side="left", padx=2)
            
            btn_inc = tk.Button(frame, text="+", font=("Consolas", 10, "bold"), width=3, bg="#333333", fg="#ffffff",
                                activebackground="#444444", activeforeground="#ffffff", bd=0, relief="flat", command=inc)
            btn_inc.pack(side="left", padx=2)
            btn_inc.bind("<Enter>", lambda e: btn_inc.config(bg="#444444"))
            btn_inc.bind("<Leave>", lambda e: btn_inc.config(bg="#333333"))
            
            return frame

        def make_preset_buttons(self, parent, entries_list, rows_var, cols_var):
            btn_frame = tk.Frame(parent, bg="#1a1a1a")
            btn_frame.pack(fill="x", pady=(5, 0))
            
            def load_identity():
                r, c = rows_var.get(), cols_var.get()
                for i in range(r):
                    for j in range(c):
                        entries_list[i][j].delete(0, tk.END)
                        entries_list[i][j].insert(0, "1" if i == j else "0")
                        
            def load_random():
                r, c = rows_var.get(), cols_var.get()
                for i in range(r):
                    for j in range(c):
                        entries_list[i][j].delete(0, tk.END)
                        entries_list[i][j].insert(0, str(random.randint(-9, 9)))
                        
            def clear_grid():
                r, c = rows_var.get(), cols_var.get()
                for i in range(r):
                    for j in range(c):
                        entries_list[i][j].delete(0, tk.END)
                        entries_list[i][j].insert(0, "0")

            presets = [("Identity", load_identity), ("Random", load_random), ("Clear", clear_grid)]
            for name, cmd in presets:
                b = tk.Button(
                    btn_frame,
                    text=name,
                    font=("Helvetica", 9),
                    bg="#2d2d2d",
                    fg="#cccccc",
                    activebackground="#3e3e3e",
                    activeforeground="#ffffff",
                    bd=0,
                    padx=10,
                    pady=2,
                    relief="flat",
                    command=cmd
                )
                b.pack(side="left", padx=2)
                b.bind("<Enter>", lambda e, btn=b: btn.config(bg="#3e3e3e"))
                b.bind("<Leave>", lambda e, btn=b: btn.config(bg="#2d2d2d"))

        def create_matrix_grid(self, container_frame, rows, cols, entries_list):
            for widget in container_frame.winfo_children():
                widget.destroy()
            entries_list.clear()
            
            # Configure frame grid geometry matching size
            for r in range(rows):
                row_entries = []
                for c in range(cols):
                    entry = tk.Entry(
                        container_frame,
                        width=6,
                        font=("Consolas", 12),
                        bg="#262626",
                        fg="#ffffff",
                        insertbackground="#ffffff",
                        bd=1,
                        relief="solid",
                        highlightbackground="#333333",
                        justify="center"
                    )
                    entry.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
                    entry.insert(0, "0")
                    row_entries.append(entry)
                entries_list.append(row_entries)

        def update_matrix_a_grid(self):
            self.create_matrix_grid(self.grid_frame_A, self.rows_A.get(), self.cols_A.get(), self.entries_A)
            
            # Multi-matrix dimension syncing rules
            op = self.active_op.get()
            if op in ["Add", "Sub"]:
                self.rows_B.set(self.rows_A.get())
                self.cols_B.set(self.cols_A.get())
                self.update_matrix_b_grid()
            elif op == "Mult":
                self.rows_B.set(self.cols_A.get())
                self.update_matrix_b_grid()

        def update_matrix_b_grid(self):
            self.create_matrix_grid(self.grid_frame_B, self.rows_B.get(), self.cols_B.get(), self.entries_B)

        def on_button_hover(self, btn, is_hover):
            op = self.active_op.get()
            btn_op = None
            for code, button in self.op_buttons.items():
                if button == btn:
                    btn_op = code
                    break
            
            if btn_op == op:
                btn.config(bg="#00adb5", fg="#ffffff")
            else:
                if is_hover:
                    btn.config(bg="#2a2a2a", fg="#00adb5")
                else:
                    btn.config(bg="#1e1e1e", fg="#dddddd")

        def select_op(self, op_code):
            self.active_op.set(op_code)
            self.on_op_change()

        def on_op_change(self):
            op = self.active_op.get()
            
            # Reset visual active buttons
            for code, btn in self.op_buttons.items():
                if code == op:
                    btn.config(bg="#00adb5", fg="#ffffff")
                else:
                    btn.config(bg="#1e1e1e", fg="#dddddd")

            # Setup screen titles and description
            details = {
                "Add": ("Matrix Addition", "Adds matrices A and B element-wise. Requires both matrices to have the exact same shape.", "A + B"),
                "Sub": ("Matrix Subtraction", "Subtracts Matrix B from Matrix A element-wise. Requires both matrices to have the exact same shape.", "A - B"),
                "Mult": ("Matrix Multiplication", "Calculates the dot product of Matrix A and B. Columns of A must equal the Rows of B.", "A × B"),
                "Transpose": ("Matrix Transpose", "Transposes Matrix A (swaps rows and columns). Operates on any shape.", "Aᵀ"),
                "Det": ("Matrix Determinant", "Calculates the determinant of Matrix A. Requires a square matrix.", "det(A)"),
                "Inverse": ("Matrix Inverse", "Calculates the multiplicative inverse of Matrix A. Requires a square, non-singular matrix.", "A⁻¹"),
                "Scalar": ("Scalar Multiplication", "Multiplies every element in Matrix A by a real scalar number c.", "c × A"),
                "Power": ("Matrix Power", "Multiplies Matrix A by itself n times. Requires a square matrix and integer power.", "Aⁿ"),
                "Rank": ("Matrix Rank", "Calculates the vector space dimension spanned by Matrix A's rows/columns.", "rank(A)"),
            }

            title, desc, math_symbol = details.get(op, ("", "", ""))
            self.op_title_lbl.config(text=f"{title} ({math_symbol})")
            self.op_desc_lbl.config(text=desc)

            # Clear any previously shown results
            self.result_container.pack_forget()

            # Dynamic Form Layout changes depending on operation
            if op in ["Add", "Sub"]:
                # Show A and B
                self.matrix_b_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
                # Lock dimension changes for B
                self.dims_frame_B_controls.pack_forget()
                self.lbl_b_synced.pack(anchor="w", pady=(0, 10))
                
                # Show column control on A
                self.col_ctrl_A.pack(side="left", padx=5)
                
                # Hide scalar/power inputs
                self.scalar_frame.pack_forget()
                self.power_frame.pack_forget()
                
                # Run dimension updates
                self.rows_B.set(self.rows_A.get())
                self.cols_B.set(self.cols_A.get())
                self.update_matrix_a_grid()
                
            elif op == "Mult":
                # Show A and B
                self.matrix_b_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
                # Allow changing columns of B but lock rows B to cols A
                self.lbl_b_synced.config(text="Rows of B are locked to Columns of A")
                self.lbl_b_synced.pack(anchor="w", pady=(0, 10))
                
                self.dims_frame_B_controls.pack(anchor="w", fill="x", pady=5)
                self.row_ctrl_B.pack_forget() # Locked
                self.col_ctrl_B.pack(side="left", padx=5)
                
                # Show column control on A
                self.col_ctrl_A.pack(side="left", padx=5)
                
                # Hide scalar/power
                self.scalar_frame.pack_forget()
                self.power_frame.pack_forget()
                
                # Update grids
                self.rows_B.set(self.cols_A.get())
                self.update_matrix_a_grid()

            elif op == "Scalar":
                # Hide B
                self.matrix_b_frame.pack_forget()
                # Show scalar control, hide power
                self.scalar_frame.pack(side="left", padx=5)
                self.power_frame.pack_forget()
                
                # Show col control on A
                self.col_ctrl_A.pack(side="left", padx=5)
                self.update_matrix_a_grid()

            elif op == "Power":
                # Hide B
                self.matrix_b_frame.pack_forget()
                # Show power control, hide scalar
                self.scalar_frame.pack_forget()
                self.power_frame.pack(side="left", padx=5)
                
                # Lock columns to rows (Square matrix required)
                self.col_ctrl_A.pack_forget()
                self.cols_A.set(self.rows_A.get())
                self.update_matrix_a_grid()

            elif op in ["Det", "Inverse"]:
                # Hide B, scalar, power
                self.matrix_b_frame.pack_forget()
                self.scalar_frame.pack_forget()
                self.power_frame.pack_forget()
                
                # Lock columns to rows (Square matrix required)
                self.col_ctrl_A.pack_forget()
                self.cols_A.set(self.rows_A.get())
                self.update_matrix_a_grid()

            else: # Transpose, Rank (Unary, flexible dimensions)
                self.matrix_b_frame.pack_forget()
                self.scalar_frame.pack_forget()
                self.power_frame.pack_forget()
                
                # Show columns control for A
                self.col_ctrl_A.pack(side="left", padx=5)
                self.update_matrix_a_grid()

        def get_matrix_values(self, entries, rows, cols, label):
            matrix = []
            for r in range(rows):
                row = []
                for c in range(cols):
                    val_str = entries[r][c].get().strip()
                    if not val_str:
                        raise ValueError(f"Empty value in {label} at row {r+1}, col {c+1}.")
                    try:
                        row.append(float(val_str))
                    except ValueError:
                        raise ValueError(f"Invalid numeric input '{val_str}' in {label} at row {r+1}, col {c+1}.")
                matrix.append(row)
            return np.array(matrix)

        def execute_calculation(self):
            op = self.active_op.get()
            
            # 1. Read Matrix A
            try:
                A = self.get_matrix_values(self.entries_A, self.rows_A.get(), self.cols_A.get(), "Matrix A")
            except ValueError as ex:
                messagebox.showerror("Input Error", str(ex))
                return

            # 2. Read Matrix B if binary operation
            B = None
            if op in ["Add", "Sub", "Mult"]:
                try:
                    B = self.get_matrix_values(self.entries_B, self.rows_B.get(), self.cols_B.get(), "Matrix B")
                except ValueError as ex:
                    messagebox.showerror("Input Error", str(ex))
                    return

            # 3. Perform operation
            try:
                if op == "Add":
                    res = add_matrices(A, B)
                    self.display_result_matrix(res, "A + B Result")
                elif op == "Sub":
                    res = subtract_matrices(A, B)
                    self.display_result_matrix(res, "A - B Result")
                elif op == "Mult":
                    res = multiply_matrices(A, B)
                    self.display_result_matrix(res, "A × B Result")
                elif op == "Transpose":
                    res = transpose_matrix(A)
                    self.display_result_matrix(res, "Aᵀ Transpose Result")
                elif op == "Det":
                    res = determinant_matrix(A)
                    self.display_result_scalar(res, "Determinant det(A)")
                elif op == "Inverse":
                    res = inverse_matrix(A)
                    self.display_result_matrix(res, "A⁻¹ Inverse Result")
                elif op == "Scalar":
                    c_str = self.scalar_val.get().strip()
                    try:
                        c = float(c_str)
                    except ValueError:
                        raise ValueError(f"Scalar multiplier must be a valid real number. Got '{c_str}'")
                    res = scalar_multiply(A, c)
                    self.display_result_matrix(res, f"Scalar product ({c} × A)")
                elif op == "Power":
                    n_str = self.power_val.get().strip()
                    try:
                        n = int(n_str)
                    except ValueError:
                        raise ValueError(f"Power exponent must be an integer. Got '{n_str}'")
                    res = matrix_power(A, n)
                    self.display_result_matrix(res, f"Matrix power (A^{n})")
                elif op == "Rank":
                    res = matrix_rank(A)
                    self.display_result_scalar(res, "Matrix Rank")
            except Exception as err:
                messagebox.showerror("Calculation Error", str(err))

        def display_result_scalar(self, val, title):
            # Clean container
            for widget in self.result_container.winfo_children():
                widget.destroy()
                
            self.result_container.pack(fill="both", expand=False, pady=(15, 0))
            
            # Header
            lbl_title = tk.Label(self.result_container, text=title, font=("Helvetica", 12, "bold"), fg="#00adb5", bg="#1a1a1a")
            lbl_title.pack(anchor="w", padx=10, pady=(5, 5))
            
            # Value formatting
            if isinstance(val, (int, np.integer)):
                val_str = f"{val}"
            else:
                val_str = f"{val:.6g}"
            
            self.last_result_str = f"{title}: {val_str}"
            
            result_frame = tk.Frame(self.result_container, bg="#262626", padx=20, pady=15)
            result_frame.pack(fill="x", padx=10, pady=5)
            
            lbl_val = tk.Label(result_frame, text=val_str, font=("Helvetica", 20, "bold"), fg="#ffffff", bg="#262626")
            lbl_val.pack(anchor="center")
            
            # Action frame
            btn_frame = tk.Frame(self.result_container, bg="#1a1a1a")
            btn_frame.pack(fill="x", padx=10, pady=(5, 5))
            
            copy_btn = tk.Button(
                btn_frame,
                text="Copy Value",
                font=("Helvetica", 9, "bold"),
                bg="#333333",
                fg="#ffffff",
                activebackground="#444444",
                activeforeground="#ffffff",
                bd=0,
                padx=10,
                pady=4,
                relief="flat",
                command=self.copy_to_clipboard
            )
            copy_btn.pack(side="right")
            copy_btn.bind("<Enter>", lambda e: copy_btn.config(bg="#444444"))
            copy_btn.bind("<Leave>", lambda e: copy_btn.config(bg="#333333"))

        def display_result_matrix(self, matrix_res, title):
            # Clean container
            for widget in self.result_container.winfo_children():
                widget.destroy()
                
            self.result_container.pack(fill="both", expand=True, pady=(15, 0))
            
            self.last_result_str = format_matrix_to_string(matrix_res)
            
            # Title
            lbl_title = tk.Label(self.result_container, text=title, font=("Helvetica", 12, "bold"), fg="#00adb5", bg="#1a1a1a")
            lbl_title.pack(anchor="w", padx=10, pady=(5, 5))
            
            # Grid and Text pane structure
            split_frame = tk.Frame(self.result_container, bg="#1a1a1a")
            split_frame.pack(fill="both", expand=True, padx=10, pady=5)
            
            # Left pane: Grid display
            left_pane = tk.Frame(split_frame, bg="#1a1a1a")
            left_pane.pack(side="left", fill="both", expand=True, padx=(0, 10))
            
            tk.Label(left_pane, text="Visual Grid View:", font=("Helvetica", 10, "bold"), fg="#888888", bg="#1a1a1a").pack(anchor="w", pady=(0, 5))
            
            grid_border = tk.Frame(left_pane, bg="#262626", padx=10, pady=10)
            grid_border.pack(anchor="nw")
            
            rows, cols = matrix_res.shape
            for r in range(rows):
                for c in range(cols):
                    val = matrix_res[r, c]
                    if np.isnan(val):
                        val_str = "NaN"
                    elif np.isinf(val):
                        val_str = "∞" if val > 0 else "-∞"
                    elif float(val).is_integer():
                        val_str = f"{int(val)}"
                    else:
                        val_str = f"{val:.4g}"
                        
                    cell = tk.Entry(
                        grid_border,
                        width=7,
                        font=("Consolas", 11),
                        bg="#1a1a1a",
                        fg="#ffffff",
                        bd=1,
                        relief="solid",
                        justify="center"
                    )
                    cell.grid(row=r, column=c, padx=2, pady=2)
                    cell.insert(0, val_str)
                    cell.config(state="readonly")
            
            # Right pane: Nice brackets ASCII display
            right_pane = tk.Frame(split_frame, bg="#1a1a1a")
            right_pane.pack(side="right", fill="both", expand=True, padx=(10, 0))
            
            tk.Label(right_pane, text="Structured ASCII Output:", font=("Helvetica", 10, "bold"), fg="#888888", bg="#1a1a1a").pack(anchor="w", pady=(0, 5))
            
            text_box = tk.Text(
                right_pane,
                height=max(rows + 2, 5),
                width=30,
                font=("Consolas", 11),
                bg="#262626",
                fg="#ffffff",
                bd=0,
                padx=10,
                pady=10
            )
            text_box.pack(fill="both", expand=True)
            text_box.insert("1.0", self.last_result_str)
            text_box.config(state="disabled")
            
            # Action frame
            btn_frame = tk.Frame(self.result_container, bg="#1a1a1a")
            btn_frame.pack(fill="x", padx=10, pady=(5, 5))
            
            copy_btn = tk.Button(
                btn_frame,
                text="Copy Matrix",
                font=("Helvetica", 9, "bold"),
                bg="#333333",
                fg="#ffffff",
                activebackground="#444444",
                activeforeground="#ffffff",
                bd=0,
                padx=10,
                pady=4,
                relief="flat",
                command=self.copy_to_clipboard
            )
            copy_btn.pack(side="right")
            copy_btn.bind("<Enter>", lambda e: copy_btn.config(bg="#444444"))
            copy_btn.bind("<Leave>", lambda e: copy_btn.config(bg="#333333"))

        def copy_to_clipboard(self):
            if self.last_result_str:
                self.clipboard_clear()
                self.clipboard_append(self.last_result_str)
                # Visual temporary toast/alert
                messagebox.showinfo("Clipboard", "Result copied to clipboard successfully!")


# -----------------------------------------------------------------------------
# 4. ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    # Force CLI mode if specifically requested or if GUI libraries are unavailable
    if "--cli" in sys.argv or not GUI_AVAILABLE:
        if not GUI_AVAILABLE and "--cli" not in sys.argv:
            print("[System Alert] Tkinter GUI libraries not found. Defaulting to CLI interface.\n")
        run_cli()
    else:
        try:
            app = MatrixApp()
            app.mainloop()
        except Exception as e:
            print(f"[System Error] Could not start Graphical Interface: {e}")
            print("Defaulting to CLI Mode...\n")
            run_cli()


if __name__ == "__main__":
    main()
