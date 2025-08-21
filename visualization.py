# visualization.py
"""
Functions for visualizing molecular orbitals.
- 2D bar plots of MO coefficients
- 3D isosurface plots of MOs using py3Dmol
"""
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
from pathlib import Path
import os
from typing import Any

from pyscf import gto
from pyscf.tools import cubegen

from config import VisualizationConfig

def plot_mo_coefficients(mo_coefficients: np.ndarray, config: VisualizationConfig, labels: list = None):
    """
    Generates bar plots for the coefficients of the first few molecular orbitals.

    Args:
        mo_coefficients: Array of MO coefficients (typically shape [n_basis, n_orbitals]).
        config: Visualization configuration.
        labels: Optional labels for the basis functions.

    Chemistry Note for Beginners:
    Molecular Orbitals (MOs) are formed by combining Atomic Orbitals (AOs).
    Each bar in the plot shows how much each AO contributes to a specific MO.
    For H2, the lowest energy MO (the bonding orbital) will show roughly equal
    contributions from the 1s orbitals of both Hydrogen atoms.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_basis_functions = mo_coefficients.shape[0]
    num_orbitals = mo_coefficients.shape[1]

    n_plots = min(config.max_orbitals_plot, num_orbitals)

    for i in range(n_plots):
        plt.figure(figsize=(10, 5))
        coeffs = mo_coefficients[:, i]
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
        print(f"Saved MO {i+1} coefficient plot to {plot_filename}")
        plt.close()

def get_basis_labels(mol: gto.Mole) -> list[str]:
    """Creates descriptive labels for each basis function."""
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
            # e.g., px, py, pz
            if l == 1:
                suffixes = ['x', 'y', 'z']
            else: # d, f, etc. have more complex cartesian forms
                suffixes = [f"{j}" for j in range(n_functions)]
            for j in range(n_functions):
                 labels.append(f"{atom_symbol}{atom_id+1} {ang}{suffixes[j]}")
    return labels


def visualize_mo_3d(mol: gto.Mole, mo_coeff: np.ndarray, orb_index: int, config: VisualizationConfig):
    """
    Renders a 3D view of a given molecular orbital and saves it as an HTML file.

    Args:
        mol: PySCF molecule object.
        mo_coeff: MO coefficients array [n_basis, n_orbitals].
        orb_index: Index of the orbital to visualize (0-based).
        config: Visualization configuration.

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
    # cubegen.orbital expects the coefficients for *one* orbital, shape (n_basis,)
    cubegen.orbital(mol, str(cube_filename), mo_coeff[:, orb_index])
    print("Cube file generated.")

    with open(cube_filename, 'r') as f:
        cube_data = f.read()

    view = py3Dmol.view(width=600, height=400)
    view.addVolumetricData(cube_data, "cube", {'isoval': config.isoval, 'color': "blue", 'opacity': 0.8})
    view.addVolumetricData(cube_data, "cube", {'isoval': -config.isoval, 'color': "red", 'opacity': 0.8})

    # Add molecule structure
    view.addModel(gto.tostring(mol), "xyz")
    view.setStyle({'stick': {'radius': 0.1}, 'sphere': {'scale': 0.25}})
    view.zoomTo()

    html_filename = output_dir / f"mo_{orb_index+1}_3d.html"
    with open(html_filename, 'w') as f:
        # Get the HTML content from py3Dmol
        f.write(view._make_html())
    print(f"Saved 3D visualization to {html_filename} - Open this file in a browser.")

