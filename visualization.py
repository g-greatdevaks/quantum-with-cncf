# visualization.py
"""
This module contains functions for generating visualizations
related to the quantum chemistry calculations, including:
- 2D bar plots of Molecular Orbital coefficients.
- 3D isosurface plots of Molecular Orbitals using py3Dmol.
- Plots of quantum circuits.
"""
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
from pathlib import Path
from typing import Any
from pyscf import gto
from pyscf.tools import cubegen
from config import VisualizationConfig
import traceback

def plot_mo_coefficients(mo_coefficients: np.ndarray, config: VisualizationConfig, labels: list = None):
    """
    Generates and saves bar plots for the coefficients of each atomic orbital
    contributing to the first few molecular orbitals.

    Args:
        mo_coefficients (np.ndarray): The matrix of MO coefficients, typically
                                     of shape (num_basis_functions, num_orbitals).
        config (VisualizationConfig): Visualization settings.
        labels (list, optional): Descriptive labels for each basis function.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_basis_functions = mo_coefficients.shape[0]
    num_orbitals = mo_coefficients.shape[1]

    n_plots = min(config.max_orbitals_plot, num_orbitals)

    for i in range(n_plots):
        plt.figure(figsize=(10, 5))
        coeffs = mo_coefficients[:, i] # Coefficients for the i-th MO

        if labels:
            plt.bar(labels, coeffs)
        else:
            plt.bar(range(num_basis_functions), coeffs)

        plt.title(f"Molecular Orbital {i+1} Coefficients")
        plt.xlabel("Atomic Basis Function")
        plt.ylabel("Coefficient Value")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plot_filename = output_dir / f"mo_{i+1}_coeffs.png"
        plt.savefig(plot_filename)
        print(f"  Saved MO {i+1} coefficient plot to {plot_filename}")
        plt.close()

def get_basis_labels(mol: gto.Mole) -> list[str]:
    """
    Creates human-readable labels for each atomic basis function in the molecule.

    Args:
        mol (gto.Mole): The PySCF molecule object.

    Returns:
        list[str]: A list of labels.
    """
    labels = []
    for i in range(mol.nbas):
        atom_symbol = mol.atom_symbol(mol.bas_atom(i))
        atom_id = mol.bas_atom(i)
        l = mol.bas_angular(i)
        l_map = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
        ang = l_map.get(l, str(l))
        
        # For higher angular momentum, PySCF has multiple functions per basis shell
        n_functions = mol.bas_len_cart(i) # Number of cartesian functions
        if n_functions == 1:
            labels.append(f"{atom_symbol}{atom_id+1} {ang}")
        else:
            if l == 1: suffixes = ['x', 'y', 'z'] # e.g., px, py, pz
            else: suffixes = [f"{j}" for j in range(n_functions)]  # d, f, etc. have more complex cartesian forms
            for j in range(n_functions):
                 labels.append(f"{atom_symbol}{atom_id+1} {ang}{suffixes[j]}")
    return labels

def visualize_mo_3d(mol: gto.Mole, mo_coeff: np.ndarray, orb_index: int, config: VisualizationConfig):
    """
    Generates a 3D visualization of a specific molecular orbital and saves it as an HTML file.

    This function uses PySCF's cubegen to create a .cube file representing the
    orbital's spatial distribution, and then uses py3Dmol to render an interactive
    3D view in HTML.

    Args:
        mol (gto.Mole): The PySCF molecule object.
        mo_coeff (np.ndarray): The MO coefficient matrix.
        orb_index (int): The 0-based index of the molecular orbital to visualize.
        config (VisualizationConfig): Visualization settings.

    Chemistry Note for Beginners:
    This function generates a 3D grid of values representing the electron density
    for the selected MO. py3Dmol then draws an isosurface, which is like a contour map
    in 3D, showing where the electron is likely to be found.
    - Bonding orbitals look like a blob surrounding both nuclei.
    - Anti-bonding orbitals have a node (a region of zero density) between the nuclei.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cube_filename = output_dir / f"mo_{orb_index+1}.cube"

    print(f"Generating cube file for MO {orb_index + 1}: {cube_filename}")
    # Generate the cube file for the selected MO.
    # cubegen.orbital expects the coefficients for *one* orbital, shape (n_basis,).
    cubegen.orbital(mol, str(cube_filename), mo_coeff[:, orb_index], nx=60, ny=60, nz=60)
    print("Cube file generated.")

    # Read the cube data.
    with open(cube_filename, 'r') as f:
        cube_data = f.read()

    # Create the py3Dmol view.
    view = py3Dmol.view(width=600, height=400)
    # Add the volumetric data for the isosurface.
    # Positive and negative phases of the wavefunction are shown in different colors.
    view.addVolumetricData(cube_data, "cube", {'isoval': config.isoval, 'color': "blue", 'opacity': 0.8})
    view.addVolumetricData(cube_data, "cube", {'isoval': -config.isoval, 'color': "red", 'opacity': 0.8})

    # Add the molecular structure (atoms and bonds).
    view.addModel(gto.tostring(mol), "xyz")
    view.setStyle({'stick': {'radius': 0.1}, 'sphere': {'scale': 0.25}})
    view.zoomTo()

    # Save the visualization as an HTML file.
    html_filename = output_dir / f"mo_{orb_index+1}_3d.html"
    with open(html_filename, 'w') as f:
        f.write(view._make_html())
    print(f"  Saved 3D visualization to {html_filename}")

def draw_circuit(circuit, filename, output_dir):
    """
    Draws a Qiskit quantum circuit and saves it to a file.

    Args:
        circuit (QuantumCircuit): The Qiskit circuit to draw.
        filename (str): The name of the output image file.
        output_dir (str): The directory to save the image in.
    """
    try:
        output_path = Path(output_dir) / filename
        # Using 'mpl' output for a matplotlib-generated image.
        # 'fold=-1' prevents line wrapping for wider circuits.
        # Custom styling for UCCSD and HartreeFock gates for clarity.
        style = {'displaycolor': {'UCCSD': ('#c2e0c6', '#000000'), 'HartreeFock': ('#a6cbe3', '#000000')}}
        circuit.draw(output='mpl', style=style, fold=-1).savefig(output_path, bbox_inches='tight')
        print(f"  Circuit diagram saved to {output_path}")
        plt.close() # Close the matplotlib figure to free memory.
    except Exception as e:
        print(f"  Failed to draw circuit {filename}: {e}")
        traceback.print_exc()
