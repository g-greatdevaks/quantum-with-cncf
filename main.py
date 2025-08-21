# main.py
"""
Main script for the Molecular Orbital Visualization Demo.
KubeCon EU 2025 - Quantum Curious

L100: Classical MO visualization
L150: Local Quantum Simulation with VQE
"""
import qiskit
from qiskit_algorithms.minimum_eigensolvers import VQE
# Import additional optimizers
from qiskit_algorithms.optimizers import COBYLA, SLSQP, SPSA, L_BFGS_B
from qiskit_aer.primitives import Estimator as AerEstimator
import qiskit_aer

# Qiskit Nature imports
import qiskit_nature
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.operators import FermionicOp

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
    print(f"  PySCF: {pyscf.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  py3Dmol: {py3Dmol.__version__}")
    print("------------------------")

def run_vqe_simulation(driver: PySCFDriver, config: AppConfig, mol: pyscf.gto.Mole, mf: pyscf.scf.hf.RHF, optimizer_name: str = "COBYLA"):
    """
    L150: Runs VQE simulation for the molecule.
    optimizer_name can be "COBYLA", "SLSQP", "SPSA", "L-BFGS-B"
    """
    print(f"\n--- Starting L150: VQE Quantum Simulation with {optimizer_name} ---")
    start_time = time.time()

    # 1. Get the ElectronicStructureProblem object from driver.run()
    problem: ElectronicStructureProblem = driver.run()

    # 2. Generate Second Quantized Operators
    try:
        second_q_ops = problem.second_q_ops()
        electronic_hamiltonian = second_q_ops[0]
    except Exception as e:
        print(f"  Error calling second_q_ops(): {e}")
        return None

    # 3. Get Nuclear Repulsion Energy
    nuclear_repulsion_energy = problem.nuclear_repulsion_energy
    if nuclear_repulsion_energy is None: nuclear_repulsion_energy = mf.energy_nuc()
    total_hamiltonian = electronic_hamiltonian + FermionicOp({"": nuclear_repulsion_energy})

    # 4. Define the Qubit Mapper
    mapper = JordanWignerMapper()

    # 5. Map the Hamiltonian to qubits
    try:
        qubit_op = mapper.map(total_hamiltonian)
    except Exception as e:
        print(f"  Error during mapper.map(): {e}")
        return None

    # 6. Ansatz Setup
    num_spatial_orbitals = mf.mo_coeff.shape[1]
    num_particles = (mol.nelec[0], mol.nelec[1])
    hartree_fock_init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    ansatz = UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=hartree_fock_init_state)

    # 7. Estimator
    estimator = AerEstimator()

    # 8. Optimizer Selection
    if optimizer_name == "COBYLA":
        optimizer = COBYLA(maxiter=2000, tol=1e-6, rhobeg=0.1)
    elif optimizer_name == "SLSQP":
        optimizer = SLSQP(maxiter=1000, tol=1e-6)
    elif optimizer_name == "SPSA":
        # SPSA is good with function evaluations, 2 per iteration.
        optimizer = SPSA(maxiter=500)
    elif optimizer_name == "L-BFGS-B":
        optimizer = L_BFGS_B(maxiter=1000, tol=1e-6)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    print(f"  Optimizer: {optimizer_name}, Settings: {optimizer.settings}")

    initial_point = np.zeros(ansatz.num_parameters)
    vqe_solver = VQE(estimator, ansatz, optimizer, initial_point=initial_point)

    # 11. Run VQE
    print("  Running VQE.compute_minimum_eigenvalue...")
    vqe_result = vqe_solver.compute_minimum_eigenvalue(qubit_op)
    end_time = time.time()
    print(f"  VQE calculation finished in {end_time - start_time:.2f} seconds.")
    if hasattr(vqe_result, 'cost_function_evals'):
        print(f"  Optimizer evaluations: {vqe_result.cost_function_evals}")

    # 12. Display Results
    vqe_energy = vqe_result.optimal_value
    print(f"\n--- VQE Results ({optimizer_name}) ---")
    print(f"  VQE Ground State Energy: {vqe_energy:.6f} Hartree")
    print(f"  Hartree-Fock Energy (from PySCF): {mf.e_tot:.6f} Hartree")
    energy_diff = vqe_energy - mf.e_tot
    print(f"  VQE Energy - HF Energy: {energy_diff:.6f} Hartree")

    if vqe_energy < mf.e_tot - 1e-4:
         print(f"  VQE found a lower energy: {vqe_energy:.6f} (Correlation energy: {energy_diff:.6f})")
    elif abs(vqe_energy - mf.e_tot) < 1e-4:
         print("  VQE energy is very close to HF energy.")
    else:
         print("  Warning: VQE energy is higher than HF. Optimization may not have converged well.")
    print("--- L150 Complete ---")
    return vqe_result

def main():
    """
    Main pipeline for the demo.
    """
    print_versions()
    config: AppConfig = load_config()
    # ... (L100 setup)
    print("\n--- Starting L100: Classical Calculations & Visualization ---")
    driver = get_pyscf_driver(config.molecule)
    try:
        mol, mf = run_pyscf_calculation(driver)
    except RuntimeError as e:
        print(f"Error during PySCF calculation: {e}")
        return
    # ... (L100 visualizations)
    mo_coeffs = mf.mo_coeff
    basis_labels = get_basis_labels(mol)
    plot_mo_coefficients(mo_coeffs, config.visualization, labels=basis_labels)
    num_orbitals = mo_coeffs.shape[1]
    num_orbitals_3d = min(2, num_orbitals)
    for i in range(num_orbitals_3d):
        visualize_mo_3d(mol, mo_coeffs, orb_index=i, config=config.visualization)
    print("--- L100 Complete ---")

    # --- L150: VQE Quantum Simulation ---
    print("\n\n=== Running COBYLA ===")
    run_vqe_simulation(driver, config, mol, mf, optimizer_name="COBYLA")

    print("\n\n=== Running SLSQP ===")
    run_vqe_simulation(driver, config, mol, mf, optimizer_name="SLSQP")

    print("\n\n=== Running SPSA ===")
    run_vqe_simulation(driver, config, mol, mf, optimizer_name="SPSA")

    print("\n\n=== Running L-BFGS-B ===")
    run_vqe_simulation(driver, config, mol, mf, optimizer_name="L-BFGS-B")

if __name__ == "__main__":
    main()
