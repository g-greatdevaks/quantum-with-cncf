# config.py
"""
Configuration models for the quantum chemistry simulation.
Using Pydantic ensures that configuration data is validated.
This structure makes it easy to load settings from various sources
like YAML files, environment variables, etc., which is great for
running in different environments like Kubernetes (L200+).
"""
from typing import Tuple
from pydantic import BaseModel, Field

class MoleculeConfig(BaseModel):
    """
    Configuration for defining the molecule.
    """
    atom_labels: Tuple[str, str] = Field(default=('H', 'H'), description="Symbols of the two atoms")
    atom_distance: float = Field(default=0.735, description="Bond length in Angstroms")
    basis: str = Field(default="sto3g", description="Basis set for calculation (e.g., sto3g, 6-31g)")
    charge: int = Field(default=0, description="Charge of the molecule")
    spin: int = Field(default=0, description="Spin multiplicity (0 for singlet, 1 for doublet, etc.)")

class VisualizationConfig(BaseModel):
    """
    Configuration for visualization settings.
    """
    max_orbitals_plot: int = Field(default=4, description="Max MOs to plot in 2D")
    isoval: float = Field(default=0.02, description="Isovalue for 3D orbital rendering")
    output_dir: str = Field(default="outputs", description="Directory to save outputs like cube files and HTML")

class AppConfig(BaseModel):
    """
    Main application configuration.
    """
    molecule: MoleculeConfig = Field(default_factory=MoleculeConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

# Example function to load config from YAML
import yaml
from pathlib import Path

def load_config(config_path: str = "molecule_config.yaml") -> AppConfig:
    """Loads configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Warning: Configuration file not found at {config_path}. Using default settings.")
        return AppConfig()
    with open(path, 'r') as f:
        config_data = yaml.safe_load(f)
    return AppConfig(**config_data)
