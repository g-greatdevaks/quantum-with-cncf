# main.py
"""
Main script for the Molecular Orbital Visualization Demo.
KubeCon EU 2025 - Quantum Curious

L100: Classical MO visualization
L150: Local Quantum Simulation with VQE
"""
import qiskit
from qiskit_algorithms.minimum_eigensolvers import VQE
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
from tabulate import tabulate # For pretty table printing

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
    try:
        import tabulate
        print(f"  Tabulate: {tabulate.__version__}")
    except ImportError:
        print("  Tabulate: Not installed")
    print("------------------------")

def run_vqe_simulation(qubit_op, ansatz, optimizer, initial_point):
    """
    Core VQE execution.
    """
    estimator = AerEstimator()
    vqe_solver = VQE(estimator, ansatz, optimizer, initial_point=initial_point)

    print(f"  Running VQE with {optimizer.__class__.__name__}...")
    start_time = time.time()
    vqe_result = vqe_solver.compute_minimum_eigenvalue(qubit_op)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"  VQE finished in {runtime:.2f} seconds.")

    evaluations = vqe_result.cost_function_evals if hasattr(vqe_result, 'cost_function_evals') else 'N/A'
    return vqe_result.optimal_value, evaluations, runtime

def compare_optimizers(driver: PySCFDriver, config: AppConfig, mol: pyscf.gto.Mole, mf: pyscf.scf.hf.RHF):
    """
    L150: Runs VQE with different optimizers and compares results.
    """
    print("\n--- Starting L150: VQE Optimizer Comparison ---")

    # 1. Setup the problem and map to qubits (common for all optimizers)
    problem: ElectronicStructureProblem = driver.run()
    try:
        second_q_ops = problem.second_q_ops()
        electronic_hamiltonian = second_q_ops[0]
    except Exception as e:
        print(f"  Error calling second_q_ops(): {e}")
        return

    nuclear_repulsion_energy = problem.nuclear_repulsion_energy
    if nuclear_repulsion_energy is None: nuclear_repulsion_energy = mf.energy_nuc()
    total_hamiltonian = electronic_hamiltonian + FermionicOp({"": nuclear_repulsion_energy})

    mapper = JordanWignerMapper()
    try:
        qubit_op = mapper.map(total_hamiltonian)
    except Exception as e:
        print(f"  Error during mapper.map(): {e}")
        return

    num_spatial_orbitals = mf.mo_coeff.shape[1]
    num_particles = (mol.nelec[0], mol.nelec[1])
    hartree_fock_init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    ansatz = UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=hartree_fock_init_state)
    initial_point = np.zeros(ansatz.num_parameters)

    print(f"  Qubit Hamiltonian created with {qubit_op.num_qubits} qubits.")
    print(f"  UCCSD Ansatz created with {ansatz.num_parameters} parameters.")

    # 2. Define Optimizer Configurations
    optimizer_configs = [
        {"name": "COBYLA", "instance": COBYLA(maxiter=2000, tol=1e-6, rhobeg=0.1)},
        {"name": "SLSQP", "instance": SLSQP(maxiter=1000, tol=1e-6)},
        {"name": "SPSA", "instance": SPSA(maxiter=500)},
         {"name": "L-BFGS-B", "instance": L_BFGS_B(maxiter=1000, tol=1e-6)},
    ]

    # 3. Run VQE for each optimizer
    results = []
    hf_energy = mf.e_tot
    print(f"\n  Hartree-Fock Energy (from PySCF): {hf_energy:.6f} Hartree")

    for opt_config in optimizer_configs:
        opt_name = opt_config["name"]
        optimizer = opt_config["instance"]
        print(f"\n--- Testing Optimizer: {opt_name} ---")
        print(f"  Settings: {optimizer.settings}")

        try:
            vqe_energy, evaluations, runtime = run_vqe_simulation(qubit_op, ansatz, optimizer, initial_point)
            correlation_energy = vqe_energy - hf_energy
            results.append({
                "Optimizer": opt_name,
                "VQE Energy (H)": f"{vqe_energy:.6f}",
                "Evals": evaluations,
                "Time (s)": f"{runtime:.2f}",
                "Corr. Energy (H)": f"{correlation_energy:.6f}"
            })
        except Exception as e:
            print(f"  Error running VQE with {opt_name}: {e}")
            traceback.print_exc()
            results.append({
                "Optimizer": opt_name,
                "VQE Energy (H)": "Error",
                "Evals": "Error",
                "Time (s)": "Error",
                "Corr. Energy (H)": "Error"
            })

    # 4. Display Comparison Table
    print("\n\n--- VQE Optimizer Comparison Results ---")
    if results:
        headers = results[0].keys()
        rows = [list(r.values()) for r in results]
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print("  No results to display.")

    print("\n--- L150 Complete ---")

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
    basis_labels = get_basis_labels(mol)
    plot_mo_coefficients(mo_coeffs, config.visualization, labels=basis_labels)
    num_orbitals = mo_coeffs.shape[1]
    num_orbitals_3d = min(2, num_orbitals)
    for i in range(num_orbitals_3d):
        visualize_mo_3d(mol, mo_coeffs, orb_index=i, config=config.visualization)
    print("--- L100 Complete ---")
    print(f"  Classical outputs are saved in the '{config.visualization.output_dir}' directory.")

    # --- L150: VQE Quantum Simulation ---
    compare_optimizers(driver, config, mol, mf)

if __name__ == "__main__":
    main()
