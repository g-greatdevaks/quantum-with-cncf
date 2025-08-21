# main.py
"""
Main script for the Molecular Orbital Visualization Demo.
KubeCon EU 2025 - Quantum Curious

L100: Classical MO visualization
L150: Local Quantum Simulation with VQE
"""
import qiskit
from qiskit_algorithms.minimum_eigensolvers import VQE
# Changed from SLSQP to COBYLA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Estimator as AerEstimator
import qiskit_aer

# Qiskit Nature imports
import qiskit_nature
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy

import pyscf
import numpy as np
import py3Dmol
import os
from pathlib import Path
import sys
import time
import traceback

from config import load_config, AppConfig
from chemistry import get_pyscf_driver, run_pyscf_calculation
from visualization import plot_mo_coefficients, visualize_mo_3d, get_basis_labels

def print_versions():
    """Prints versions of key libraries."""
    print("--- Library Versions ---")
    print(f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"  Qiskit: {qiskit.__version__}")
    print(f"  Qiskit Aer: {qiskit_aer.__version__}")
    import qiskit_algorithms
    print(f"  Qiskit Algorithms: {qiskit_algorithms.__version__}")
    print(f"  Qiskit Nature: {qiskit_nature.__version__}")
    # assert qiskit_nature.__version__ == "0.7.2"
    print(f"  PySCF: {pyscf.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  py3Dmol: {py3Dmol.__version__}")
    print("------------------------")

def run_vqe_simulation(driver: PySCFDriver, config: AppConfig, mol: pyscf.gto.Mole, mf: pyscf.scf.hf.RHF):
    """
    L150: Runs VQE simulation for the molecule.
    """
    print("\n--- Starting L150: VQE Quantum Simulation ---")
    start_time = time.time()

    # 1. Create ElectronicStructureProblem
    print("  Creating ElectronicStructureProblem...")
    problem = driver.run()
    print("  ElectronicStructureProblem instance created.")

    # 2. Generate Second Quantized Operators
    try:
        print("  Calling problem.second_q_ops()...")
        second_q_ops = problem.second_q_ops()
        if not second_q_ops:
            print("  Error: second_q_ops() returned empty list or None.")
            return None
        hamiltonian = second_q_ops[0]
        print("  Second quantized operators obtained.")
    except Exception as e:
        print(f"  Error calling second_q_ops(): {e}")
        traceback.print_exc()
        return None

    # 3. Define the Qubit Mapper and map the Hamiltonian
    mapper = JordanWignerMapper()
    try:
        print("  Mapping Hamiltonian to qubits...")
        qubit_op = mapper.map(hamiltonian)
        if qubit_op is None:
             print("  Error: mapper.map() returned None")
             return None
        print(f"  Qubit Hamiltonian created with {qubit_op.num_qubits} qubits.")
    except Exception as e:
        print(f"  Error during mapper.map(): {e}")
        traceback.print_exc()
        return None

    # 4. Ansatz Setup
    num_spatial_orbitals = mf.mo_coeff.shape[1]
    num_particles = (mol.nelec[0], mol.nelec[1])
    ansatz = UCCSD(num_spatial_orbitals, num_particles, mapper)
    print(f"  UCCSD Ansatz created with {ansatz.num_parameters} parameters.")

    # 5. Estimator
    estimator = AerEstimator()

    # 6. Optimizer
    # Switched to COBYLA and increased maxiter
    optimizer = COBYLA(maxiter=1000, tol=1e-4)
    print(f"  Optimizer: COBYLA, maxiter=1000")

    # 7. Initial Point for VQE
    initial_point = np.zeros(ansatz.num_parameters)

    # 8. VQE Solver
    vqe_solver = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point  # Set the initial point
    )

    # 9. Run VQE
    print("  Running VQE.compute_minimum_eigenvalue...")
    vqe_result = vqe_solver.compute_minimum_eigenvalue(qubit_op)
    end_time = time.time()
    print(f"  VQE calculation finished in {end_time - start_time:.2f} seconds.")
    print(f"  Optimizer evaluations: {vqe_result.cost_function_evals}")

    # 10. Display Results
    vqe_energy = vqe_result.optimal_value
    print("\n--- VQE Results ---")
    print(f"  VQE Ground State Energy: {vqe_energy:.6f} Hartree")
    print(f"  Hartree-Fock Energy (from PySCF): {mf.e_tot:.6f} Hartree")
    energy_diff = vqe_energy - mf.e_tot
    print(f"  VQE Energy - HF Energy: {energy_diff:.6f} Hartree")

    if vqe_energy > mf.e_tot + 1e-4:
         print("  Warning: VQE energy is still higher than HF. Optimization may not have converged well.")
    elif vqe_energy < mf.e_tot - 1e-9:
         print(f"  VQE found a lower energy: {vqe_energy:.6f} (Correlation energy: {energy_diff:.6f})")
    else:
         print("  VQE energy is very close to HF energy.")

    print("--- L150 Complete ---")
    return vqe_result

def main():
    """
    Main pipeline for the demo.
    """
    print_versions()
    config: AppConfig = load_config()
    print(f"Loaded Configuration: {config}")
    output_dir = Path(config.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- L100: Classical Calculations & Visualization ---
    print("\n--- Starting L100: Classical Calculations & Visualization ---")
    driver = get_pyscf_driver(config.molecule)
    try:
        mol, mf = run_pyscf_calculation(driver)
    except RuntimeError as e:
        print(f"Error during PySCF calculation: {e}")
        return

    mo_coeffs = mf.mo_coeff
    if mo_coeffs is None:
        print("Error: Could not retrieve mo_coeff from PySCF result (mf).")
        return

    basis_labels = get_basis_labels(mol)
    plot_mo_coefficients(mo_coeffs, config.visualization, labels=basis_labels)

    num_orbitals = mo_coeffs.shape[1]
    num_orbitals_3d = min(2, num_orbitals)
    for i in range(num_orbitals_3d):
        visualize_mo_3d(mol, mo_coeffs, orb_index=i, config=config.visualization)
    print("--- L100 Complete ---")
    print(f"  Classical outputs are saved in the '{config.visualization.output_dir}' directory.")

    # --- L150: VQE Quantum Simulation ---
    run_vqe_simulation(driver, config, mol, mf)

if __name__ == "__main__":
    main()
