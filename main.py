# main.py
"""
Main script for the Molecular Orbital Visualization and Quantum Simulation Demo.
KubeCon EU 2025 - Quantum Curious: Your Cloud-Native Launchpad

This script demonstrates a hybrid classical-quantum workflow:

L100: Classical Chemistry Visualization
      - Uses PySCF to calculate molecular orbitals (MOs) for a given molecule.
      - Visualizes MO coefficients and 3D orbital shapes.
      This stage is purely classical.

L150: Local Quantum Simulation with VQE
      - Employs the Variational Quantum Eigensolver (VQE) algorithm to find the
        ground state energy of the molecule.
      - Maps the molecule's electronic Hamiltonian to a qubit Hamiltonian.
      - Uses a Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.
      - Compares different classical optimizers for the VQE loop.
      - Runs on a local classical simulator (Qiskit Aer).
      This stage introduces quantum computation concepts.
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
import matplotlib.pyplot as plt
import py3Dmol
import os
from pathlib import Path
import sys
import time
import traceback
from tabulate import tabulate
import pandas as pd
import argparse

# Local imports
from config import AppConfig, OptimizerSettings
from chemistry import get_pyscf_driver, run_pyscf_calculation
from visualization import plot_mo_coefficients, visualize_mo_3d, get_basis_labels, draw_circuit

def print_versions():
    """Prints versions of key libraries for debugging and reproducibility."""
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
    try:
        import pandas
        print(f"  Pandas: {pandas.__version__}")
    except ImportError:
        print("  Pandas: Not installed")
    print("------------------------")

def run_vqe_simulation(qubit_op, ansatz, optimizer, initial_point):
    """
    Executes the VQE algorithm for a given qubit operator, ansatz, and optimizer.

    Args:
        qubit_op (BaseOperator): The qubit Hamiltonian.
        ansatz (QuantumCircuit): The parameterized trial circuit.
        optimizer (Optimizer): The classical optimizer instance.
        initial_point (np.ndarray): Initial parameters for the ansatz.

    Returns:
        tuple: (vqe_energy, evaluations, runtime, vqe_result)
    """
    # AerEstimator is used to calculate expectation values on the Aer simulator.
    estimator = AerEstimator()

    # VQE algorithm setup.
    vqe_solver = VQE(estimator, ansatz, optimizer, initial_point=initial_point)

    print(f"  Running VQE with {optimizer.__class__.__name__}...")
    start_time = time.time()
    # This is the core VQE execution, where the optimizer minimizes the energy.
    vqe_result = vqe_solver.compute_minimum_eigenvalue(qubit_op)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"  VQE finished in {runtime:.2f} seconds.")

    # Number of times the VQE energy was evaluated by the optimizer.
    evaluations = vqe_result.cost_function_evals if hasattr(vqe_result, 'cost_function_evals') else -1
    return vqe_result.optimal_value, evaluations, runtime, vqe_result

def get_optimizer_instance(name: str, settings: OptimizerSettings):
    """
    Instantiates an optimizer object based on its name and settings from the config.

    Args:
        name (str): The name of the optimizer (e.g., "COBYLA").
        settings (OptimizerSettings): Pydantic model containing the optimizer parameters.

    Returns:
        Optimizer: An instance of the requested Qiskit optimizer.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    # Convert Pydantic settings to a dictionary, excluding unset values.
    settings_dict = settings.model_dump(exclude_unset=True)
    if name == "COBYLA":
        return COBYLA(**settings_dict)
    elif name == "SLSQP":
        return SLSQP(**settings_dict)
    elif name == "SPSA":
        return SPSA(**settings_dict)
    elif name == "L-BFGS-B":
        return L_BFGS_B(**settings_dict)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def compare_optimizers(driver: PySCFDriver, config: AppConfig, mol: pyscf.gto.Mole, mf: pyscf.scf.hf.RHF, optimizers_to_run: list):
    """
    L150: Runs VQE with different classical optimizers and compares their performance.

    This function sets up the quantum problem (Hamiltonian and Ansatz) and then
    iterates through the specified list of optimizers, running a full VQE
    calculation for each. The results are collected and displayed in a table.

    Args:
        driver (PySCFDriver): The driver instance used for classical calculations.
        config (AppConfig): The application configuration.
        mol (gto.Mole): The PySCF molecule object.
        mf (scf.hf.RHF): The PySCF Hartree-Fock result object.
        optimizers_to_run (list): A list of optimizer names to test.
    """
    print("\n--- Starting L150: VQE Optimizer Comparison ---")
    output_dir = Path(config.visualization.output_dir)

    # 1. Setup the quantum problem
    problem: ElectronicStructureProblem = driver.run()
    nuclear_repulsion_energy = problem.nuclear_repulsion_energy or mf.energy_nuc()
    electronic_hamiltonian = problem.second_q_ops()[0]
    total_hamiltonian = electronic_hamiltonian + FermionicOp({"": nuclear_repulsion_energy})

    mapper = JordanWignerMapper()
    qubit_op = mapper.map(total_hamiltonian)
    print(f"  Qubit Hamiltonian created with {qubit_op.num_qubits} qubits.")

    # 2. Ansatz Setup
    num_spatial_orbitals = mf.mo_coeff.shape[1]
    num_particles = (mol.nelec[0], mol.nelec[1])
    hartree_fock_init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    ansatz = UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=hartree_fock_init_state)
    initial_point = np.zeros(ansatz.num_parameters)
    print(f"  UCCSD Ansatz created with {ansatz.num_parameters} parameters.")

    if config.visualization.draw_circuits:
        draw_circuit(hartree_fock_init_state, "initial_state_hartree_fock.png", output_dir)
        draw_circuit(ansatz.decompose(), "ansatz_uccsd.png", output_dir)

    # 3. Run VQE for each optimizer
    raw_results = []
    hf_energy = mf.e_tot

    for opt_name in optimizers_to_run:
        if opt_name not in config.optimizers.settings:
            print(f"Warning: Settings for Optimizer '{opt_name}' not found in config. Skipping.")
            continue
        opt_settings = config.optimizers.settings[opt_name]
        print(f"\n--- Testing Optimizer: {opt_name} ---")
        print(f"  Settings: {opt_settings.model_dump(exclude_unset=True)}")

        try:
            optimizer = get_optimizer_instance(opt_name, opt_settings)
            vqe_energy, evaluations, runtime, vqe_result = run_vqe_simulation(qubit_op, ansatz, optimizer, initial_point)
            opt_result = vqe_result.optimizer_result

            # Robustly check for convergence
            converged = getattr(opt_result, 'success', None)
            status = getattr(opt_result, 'status', -99)
            if converged is None:
                converged = (status == 0)

            raw_results.append({
                "Optimizer": opt_name,
                "VQE Energy": vqe_energy,
                "Evals": evaluations,
                "Time": runtime,
                "Converged": converged,
                "NFEV": getattr(opt_result, 'nfev', -1),
                "NIT": getattr(opt_result, 'nit', -1),
                "Status": status,
                "Message": getattr(opt_result, 'message', 'N/A')
            })
        except Exception as e:
            print(f"  Error running VQE with {opt_name}: {e}")
            traceback.print_exc()
            raw_results.append({
                "Optimizer": opt_name, "VQE Energy": float('inf'), "Evals": -1, "Time": float('inf'),
                "Converged": False, "NFEV": -1, "NIT": -1, "Status": -1, "Message": "Error"
            })

    # 4. Create DataFrame and Add Rankings for Comparison
    df = pd.DataFrame(raw_results)

    if not df.empty:
        df["Energy Rank"] = df["VQE Energy"].rank(method='min')
        df["Evals Rank"] = df["Evals"].replace(-1, float('inf')).rank(method='min')
        df["NFEV Rank"] = df["NFEV"].replace(-1, float('inf')).rank(method='min')
        df["NIT Rank"] = df["NIT"].replace(-1, float('inf')).rank(method='min')
        df["Time Rank"] = df["Time"].rank(method='min')
        df["Overall Score"] = df["Energy Rank"] + df["NFEV Rank"] + df["Time Rank"]
        df["Overall Rank"] = df["Overall Score"].rank(method='min')

        display_df = df.copy()
        display_df["VQE Energy (H)"] = display_df["VQE Energy"].apply(lambda x: f"{x:.6f}" if x != float('inf') else "Error")
        display_df["Time (s)"] = display_df["Time"].apply(lambda x: f"{x:.2f}" if x != float('inf') else "Error")
        display_df["Corr. Energy (H)"] = display_df["VQE Energy"].apply(lambda x: f"{x - hf_energy:.6f}" if x != float('inf') else "Error")

        display_df = display_df[["Optimizer", "VQE Energy (H)", "Corr. Energy (H)", "Evals", "NFEV", "NIT", "Time (s)", "Converged", "Status", "Message",
                                 "Energy Rank", "NFEV Rank", "Time Rank", "Overall Rank"]]

        print(f"\n\nReference Hartree-Fock Energy: {hf_energy:.6f} Hartree")
        print("\n--- VQE Optimizer Comparison Results ---")
        print("  Corr. Energy (H): VQE Energy - HF Energy")
        print("  Evals: Total VQE energy evaluations (from vqe.cost_function_evals)")
        print("  NFEV:  Number of Function Evaluations (reported by the optimizer)")
        print("  NIT:   Number of Optimizer Iterations (reported by the optimizer)")
        print("  Status: Optimizer termination status code (0 often means success)")
        try:
            print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False, maxcolwidths=[None, None, None, None, None, None, None, None, None, 15, None, None, None, None]))
        except ImportError:
            print("  'tabulate' library not found. Please install it for a formatted table.")
            print(df)
    else:
        print("  No results to display.")

    print("\n--- L150 Complete ---")

def main():
    """
    Main execution function:
    - Parses command line arguments.
    - Loads configuration.
    - Runs L100 classical chemistry steps.
    - Runs L150 VQE optimizer comparison.
    """
    parser = argparse.ArgumentParser(description="KubeCon Quantum Chemistry Demo")
    parser.add_argument("--config", default="molecule_config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--optimizers", nargs='*', help="Override list of optimizers to test (space separated)")
    args = parser.parse_args()

    print_versions()
    config: AppConfig = AppConfig.load_from_yaml(args.config)

    if args.optimizers:
        config.optimizers.run = args.optimizers
        print(f"Using optimizers from command line: {config.optimizers.run}")
    else:
        print(f"Using optimizers from config: {config.optimizers.run}")

    output_dir = Path(config.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir.resolve()}")

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

    # --- L150: VQE Quantum Simulation ---
    compare_optimizers(driver, config, mol, mf, optimizers_to_run=config.optimizers.run)

if __name__ == "__main__":
    main()
