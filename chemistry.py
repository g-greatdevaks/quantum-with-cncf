# chemistry.py
"""
This module handles the classical quantum chemistry calculations
interfacing with PySCF through Qiskit Nature's drivers.
"""
import numpy as np
from typing import Tuple, Any
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from config import MoleculeConfig
from pyscf import gto, scf

def get_pyscf_driver(config: MoleculeConfig) -> PySCFDriver:
    """
    Constructs a PySCFDriver instance based on the provided molecule configuration.

    The PySCFDriver acts as an interface between Qiskit Nature and the PySCF
    classical chemistry package.

    Args:
        config: A MoleculeConfig object containing atom types, distance, basis set, etc.

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
    # Define the geometry of the molecule in a format PySCF understands.
    # For a diatomic molecule, we place one atom at the origin and the other
    # along the z-axis at the specified bond distance.
    atom_str = (
        f"{config.atom_labels[0]} 0.0 0.0 0.0; "
        f"{config.atom_labels[1]} 0.0 0.0 {config.atom_distance}"
    )

    # Instantiate the driver.
    driver = PySCFDriver(
        atom=atom_str,
        basis=config.basis,          # The set of atomic orbitals used to build molecular orbitals.
        charge=config.charge,        # The total charge of the system.
        spin=config.spin,            # The total spin multiplicity (2S+1). 0 means singlet.
        unit=DistanceUnit.ANGSTROM   # Units for the atomic coordinates.
    )
    print(f"PySCFDriver configured for: {atom_str}, Basis: {config.basis}, Unit: {driver.unit}")
    return driver

def run_pyscf_calculation(driver: PySCFDriver) -> Tuple[gto.Mole, scf.hf.RHF]:
    """
    Executes the classical Hartree-Fock calculation using PySCF via the driver.

    This method populates the driver with the results from PySCF, including
    the molecule object and the mean-field (SCF) calculation object.

    Args:
        driver: The configured PySCFDriver instance.

    Returns:
        A tuple containing:
          - mol (gto.Mole): The PySCF molecule object.
          - mf (scf.hf.RHF): The PySCF Restricted Hartree-Fock object,
                             containing results like total energy and MO coefficients.

    Raises:
        RuntimeError: If the PySCF calculation fails to produce the expected objects.
    """
    # This call runs the SCF calculation within PySCF.
    driver.run_pyscf()
    print("PySCF calculation complete.")

    # The results are stored as private attributes in the driver.
    if not hasattr(driver, '_mol') or not hasattr(driver, '_calc'):
        raise RuntimeError("PySCF calculation did not result in driver._mol or driver._calc.")

    mol: gto.Mole = driver._mol
    mf: scf.hf.RHF = driver._calc
    return mol, mf
