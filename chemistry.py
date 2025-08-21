# chemistry.py
"""
Functions for setting up and running quantum chemistry calculations
using Qiskit Nature and PySCF.
"""
import numpy as np
from typing import Tuple, Any
import os

# Import UnitType
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit

from config import MoleculeConfig

from pyscf import gto, scf

def get_pyscf_driver(config: MoleculeConfig) -> PySCFDriver:
    """
    Constructs a PySCFDriver for a diatomic molecule.

    Args:
        config: Molecule configuration object.

    Returns:
        A PySCFDriver instance.

    Chemistry Note for Beginners:
    A 'driver' is like a bridge to a classical chemistry software package.
    Here, PySCF calculates the properties of the molecule, like how the
    electrons are arranged.
    - atom_str: Defines the atoms and their positions. We place one atom
      at (0,0,0) and the other along the Z-axis.
    - basis: A set of mathematical functions used to approximate the shape
      of electron orbitals. "sto3g" is a simple, minimal basis set, good for demos.
    - unit: Specifies the units for atomic coordinates. ANGSTROM is standard.
    """
    atom_str = (
        f"{config.atom_labels[0]} 0.0 0.0 0.0; "
        f"{config.atom_labels[1]} 0.0 0.0 {config.atom_distance}"
    )
    driver = PySCFDriver(
        atom=atom_str,
        basis=config.basis,
        charge=config.charge,
        spin=config.spin,
        unit=DistanceUnit.ANGSTROM # Changed from "Angstrom"
    )
    print(f"PySCFDriver configured for: {atom_str}, Basis: {config.basis}, Unit: {driver.unit}")
    return driver

def run_electronic_structure(driver: PySCFDriver) -> Any:
    """
    Runs the electronic structure calculation using the driver.

    Args:
        driver: The configured PySCFDriver.

    Returns:
        The result object from the driver's run, which contains
        detailed information about the electronic structure.

    Chemistry Note for Beginners:
    This step performs a Hartree-Fock calculation (via PySCF). This method finds
    the best possible arrangement of electrons in *molecular orbitals* (MOs),
    assuming each electron moves in the average field of all other electrons.
    The result gives us energies and shapes of these MOs.
    """
    print("Running electronic structure calculation with PySCF...")
    electronic_structure_result = driver.run_pyscf()
    print("Calculation complete.")

    # run_pyscf() stores the results in private attributes _mol and _calc
    if not hasattr(driver, '_mol') or not hasattr(driver, '_calc'):
        raise RuntimeError("PySCF calculation did not result in driver._mol or driver._calc.")

    mol: gto.Mole = driver._mol
    mf: scf.hf.RHF = driver._calc
    return mol, mf
    # # Log some basic results
    # print(f"orbital coefficients: {electronic_structure_result.raw_result} and number of spatial orbitals: {electronic_structure_result.num_spatial_orbitals}")
    # mo_coeff = electronic_structure_result.aao_to_mo_coeff
    # if mo_coeff is not None:
    #     print(f"Number of molecular orbitals: {mo_coeff.shape[1]}")
    # else:
    #     print("MO coefficients not found in results.")
    # return electronic_structure_result

def get_pyscf_molecule_and_calculation(driver: PySCFDriver):
    """
    Extracts the underlying PySCF molecule and calculation objects.
    These are needed for detailed operations like generating cube files.

    Args:
        driver: The PySCFDriver, *after* driver.run() has been called.

    Returns:
        A tuple (mol, mf), where mol is the PySCF molecule object and
        mf is the PySCF mean-field (SCF) calculation object.

    Raises:
        RuntimeError: if the driver hasn't run the calculation yet.
    """
    if not hasattr(driver, '_mol') or not hasattr(driver, '_calc'):
         # Attempt to run PySCF if not already done, to populate _mol and _calc
        print("Driver has not been run, attempting to run run_pyscf()")
        driver.run_pyscf()
        if not hasattr(driver, '_mol') or not hasattr(driver, '_calc'):
            raise RuntimeError("The driver.run() method must be called and complete before accessing PySCF objects.")

    mol: gto.Mole = driver._mol
    mf: scf.hf.RHF = driver._calc
    return mol, mf

def run_pyscf_calculation(driver: PySCFDriver) -> Tuple[gto.Mole, scf.hf.RHF]:
    """
    Runs the underlying PySCF calculation and returns the PySCF molecule
    and SCF result objects.

    Args:
        driver: The configured PySCFDriver.

    Returns:
        A tuple (mol, mf), where mol is the PySCF molecule object and
        mf is the PySCF mean-field (SCF) calculation object (e.g., RHF).

    Raises:
        RuntimeError: If the PySCF objects are not found after running.
    """
    print("Running PySCF calculation via driver.run_pyscf()...")
    driver.run_pyscf()  # This executes the SCF calculation
    print("PySCF calculation complete.")

    # run_pyscf() stores the results in private attributes _mol and _calc
    if not hasattr(driver, '_mol') or not hasattr(driver, '_calc'):
        raise RuntimeError("PySCF calculation did not result in driver._mol or driver._calc.")

    mol: gto.Mole = driver._mol
    mf: scf.hf.RHF = driver._calc
    return mol, mf