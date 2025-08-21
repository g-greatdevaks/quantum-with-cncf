# main.py
"""
Main script for the Molecular Orbital Visualization Demo.
KubeCon EU 2025 - Quantum Curious

L100: Classical MO visualization
L150: Local Quantum Simulation with VQE
"""
import qiskit
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Estimator as AerEstimator
import qiskit_aer

# Qiskit Nature imports
import qiskit_nature
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
# Import HartreeFock for the initial state
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

    # 1. Get the ElectronicStructureProblem object from driver.run()
    # This seems to be what works in your environment
    print("  Running driver.run() to get ElectronicStructureProblem...")
    problem: ElectronicStructureProblem = driver.run()
    if not isinstance(problem, ElectronicStructureProblem):
         print(f"  Error: driver.run() did not return an ElectronicStructureProblem. Type: {type(problem)}")
         return None
    print("  ElectronicStructureProblem obtained.")

    # 2. Generate Second Quantized Operators
    try:
        print("  Calling problem.second_q_ops()...")
        second_q_ops = problem.second_q_ops()
        electronic_hamiltonian = second_q_ops[0]
        print("  Electronic Hamiltonian (FermionicOp) obtained.")
    except Exception as e:
        print(f"  Error calling second_q_ops(): {e}")
        traceback.print_exc()
        return None

    # 3. Get Nuclear Repulsion Energy
    nuclear_repulsion_energy = problem.nuclear_repulsion_energy
    if nuclear_repulsion_energy is None:
        nuclear_repulsion_energy = mf.energy_nuc()
    print(f"  Nuclear Repulsion Energy: {nuclear_repulsion_energy}")
    total_hamiltonian = electronic_hamiltonian + FermionicOp({"": nuclear_repulsion_energy})

    # 4. Define the Qubit Mapper
    mapper = JordanWignerMapper()

    # 5. Map the Hamiltonian to qubits
    try:
        print("  Mapping Hamiltonian to qubits...")
        qubit_op = mapper.map(total_hamiltonian)
        print(f"  Qubit Hamiltonian created with {qubit_op.num_qubits} qubits.")
    except Exception as e:
        print(f"  Error during mapper.map(): {e}")
        traceback.print_exc()
        return None

    # 6. Ansatz Setup
    num_spatial_orbitals = mf.mo_coeff.shape[1]
    num_particles = (mol.nelec[0], mol.nelec[1])

    # --- CRITICAL CHANGE: Add HartreeFock initial state ---
    hartree_fock_init_state = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper
    )

    ansatz = UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=hartree_fock_init_state # Set the initial state
    )
    print(f"  UCCSD Ansatz created with {ansatz.num_parameters} parameters, using Hartree-Fock initial state.")

    # 7. Estimator
    estimator = AerEstimator()

    # 8. Optimizer
    optimizer = COBYLA(maxiter=2000, tol=1e-6, rhobeg=0.1)
    print(f"  Optimizer: COBYLA, maxiter=2000")

    # 9. Initial Point for VQE - parameters for the *excitations*
    initial_point = np.zeros(ansatz.num_parameters)

    # 10. VQE Solver
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
    print("\n--- VQE Results ---")
    print(f"  VQE Ground State Energy: {vqe_energy:.6f} Hartree")
    print(f"  Hartree-Fock Energy (from PySCF): {mf.e_tot:.6f} Hartree")
    energy_diff = vqe_energy - mf.e_tot
    print(f"  VQE Energy - HF Energy: {energy_diff:.6f} Hartree")

    if vqe_energy < mf.e_tot - 1e-4:
         print(f"  VQE found a lower energy: {vqe_energy:.6f} (Correlation energy: {energy_diff:.6f})")
    elif abs(vqe_energy - mf.e_tot) < 1e-4:
         print("  VQE energy is very close to HF energy.")
    else:
         print("  Warning: VQE energy is higher than HF. This should not happen with proper setup.")
    print("--- L150 Complete ---")
    return vqe_result

def main():
    # ... (rest of main function is unchanged)
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
