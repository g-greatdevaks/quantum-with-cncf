# main.py
"""
Main script for the Molecular Orbital Visualization Demo.
KubeCon EU 2025 - Quantum Curious

This script demonstrates:
1.  Loading molecule configuration.
2.  Running a quantum chemistry calculation using Qiskit Nature with PySCF.
3.  Visualizing Molecular Orbital coefficients with Matplotlib.
4.  Generating interactive 3D visualizations of Molecular Orbitals using py3Dmol.
"""
import qiskit
import qiskit_nature
import pyscf
import numpy as np
import py3Dmol
import os
from pathlib import Path
import sys

from config import load_config, AppConfig
from chemistry import get_pyscf_driver, run_pyscf_calculation
from visualization import plot_mo_coefficients, visualize_mo_3d, get_basis_labels

def print_versions():
    """Prints versions of key libraries."""
    print("--- Library Versions ---")
    print(f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"  Qiskit: {qiskit.__version__}")
    # Qiskit Nature version check needs to be compatible with older and newer versions
    try:
        import qiskit_nature.version
        print(f"  Qiskit Nature: {qiskit_nature.version.VERSION}")
    except (ImportError, AttributeError):
        print(f"  Qiskit Nature: {qiskit_nature.__version__}")
    print(f"  PySCF: {pyscf.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  py3Dmol: {py3Dmol.__version__}")
    print("------------------------")

def main():
    """
    Main pipeline for the demo.
    """
    print_versions()

    # Load configuration
    config: AppConfig = load_config()
    print(f"Loaded Configuration: {config}")

    # Ensure output directory exists
    output_dir = Path(config.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Get Driver ---
    driver = get_pyscf_driver(config.molecule)

    # --- Step 2: Run PySCF Calculation ---
    # electronic_structure_result = run_electronic_structure(driver)
    try:
        mol, mf = run_pyscf_calculation(driver)
    except RuntimeError as e:
        print(f"Error during PySCF calculation: {e}")
        return

    # Extract MO coefficients (shape: [num_basis_functions, num_orbitals])
    # The MO coefficients are stored in the 'mo_coeff' attribute of the PySCF mean-field object
    mo_coeffs = mf.mo_coeff
    if mo_coeffs is None:
        print("Error: Could not retrieve mo_coeff from PySCF result (mf).")
        return

    # --- Step 3: Extract MO Coefficients ---
    # The MO coefficients are stored in the 'mo_coeff' attribute of the PySCF mean-field object
    mo_coeffs = mf.mo_coeff
    if mo_coeffs is None:
        print("Error: Could not retrieve mo_coeff from PySCF result (mf).")
        return

    print(f"MO coefficient matrix shape: {mo_coeffs.shape}")
    num_orbitals = mo_coeffs.shape[1]

    # --- Step 4: 2D Bar Plot of Coefficients ---
    basis_labels = get_basis_labels(mol)
    plot_mo_coefficients(mo_coeffs, config.visualization, labels=basis_labels)

    # --- Step 5: 3D Advanced MO Visualization ---
    num_orbitals_3d = min(2, num_orbitals)
    for i in range(num_orbitals_3d):
        print(f"\nGenerating 3D view for MO {i + 1}...")
        visualize_mo_3d(mol, mo_coeffs, orb_index=i, config=config.visualization)

    print("\n--- Demo Complete ---")
    print(f"Outputs are saved in the '{config.visualization.output_dir}' directory.")

if __name__ == "__main__":
    main()
