# config.py
"""
Configuration models for the quantum chemistry simulation.

This module uses Pydantic with BaseSettings to define and manage configuration.
This allows for type-safe settings that can be loaded from a YAML file,
environment variables, or defaults defined in the code.

Environment variables can override YAML settings. The prefix and delimiter
are set in model_config. For example, to override the molecule's atom distance,
you can set the environment variable: DEMO_MOLECULE_ATOM_DISTANCE=0.74

Nested settings like optimizer parameters can be set using a double underscore, e.g.,
DEMO_OPTIMIZERS__SETTINGS__COBYLA__MAXITER=3000
"""
from typing import Tuple, List, Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path

class MoleculeConfig(BaseSettings):
    """
    Configuration for defining the molecule.
    These settings determine the physical system to be simulated.
    """
    atom_labels: Tuple[str, str] = Field(default=('H', 'H'), description="Symbols of the two atoms")
    atom_distance: float = Field(default=0.735, description="Bond length in Angstroms")
    basis: str = Field(default="sto3g", description="Basis set for calculation (e.g., sto3g, 6-31g)")
    charge: int = Field(default=0, description="Charge of the molecule")
    spin: int = Field(default=0, description="Spin multiplicity (0 for singlet, 1 for doublet, etc.)")

    model_config = SettingsConfigDict(env_prefix='DEMO_MOLECULE_')

class VisualizationConfig(BaseSettings):
    """
    Configuration for visualization settings.
    Controls how outputs are generated and saved.
    """
    max_orbitals_plot: int = Field(default=4, description="Max MOs to plot in 2D")
    isoval: float = Field(default=0.02, description="Isovalue for 3D orbital rendering")
    output_dir: str = Field(default="outputs", description="Directory to save outputs like cube files and HTML")
    draw_circuits: bool = Field(default=True, description="Whether to draw and save quantum circuits")

    model_config = SettingsConfigDict(env_prefix='DEMO_VIS_')

class OptimizerSettings(BaseSettings):
    """
    A flexible container for individual optimizer settings.
    Uses extra='allow' to accept any keyword arguments supported by the specific optimizer.
    """
    maxiter: Optional[int] = None
    tol: Optional[float] = None
    rhobeg: Optional[float] = None
    eps: Optional[float] = None
    # Add other common optimizer params here if needed

    model_config = SettingsConfigDict(extra='allow')

class OptimizerConfig(BaseSettings):
    """
    Configuration for Optimizers to be used in VQE.
    Specifies which optimizers to run and their parameters.
    """
    run: List[str] = Field(default=["COBYLA", "SLSQP", "SPSA", "L-BFGS-B"], description="List of optimizers to test")
    settings: Dict[str, OptimizerSettings] = Field(default={
        "COBYLA": OptimizerSettings(maxiter=2000, tol=1e-6, rhobeg=0.1),
        "SLSQP": OptimizerSettings(maxiter=1000, tol=1e-6),
        "SPSA": OptimizerSettings(maxiter=500),
        "L-BFGS-B": OptimizerSettings(maxiter=1000, tol=1e-6, eps=1e-8),
    }, description="Specific settings for each optimizer")

    model_config = SettingsConfigDict(env_prefix='DEMO_OPTIMIZERS_')

class AppConfig(BaseSettings):
    """
    Main application configuration, composing the other config classes.
    """
    molecule: MoleculeConfig = Field(default_factory=MoleculeConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    optimizers: OptimizerConfig = Field(default_factory=OptimizerConfig)

    # env_nested_delimiter='__' allows setting nested env vars like:
    # DEMO_OPTIMIZERS__RUN='["COBYLA"]'
    # DEMO_OPTIMIZERS__SETTINGS__COBYLA__MAXITER=500
    model_config = SettingsConfigDict(env_nested_delimiter='__')

    @classmethod
    def load_from_yaml(cls, config_path: str = "molecule_config.yaml"):
        """
        Loads configuration from a YAML file.
        Pydantic will then overlay any settings provided by environment variables.
        """
        path = Path(config_path)
        config_data = {}
        if path.exists():
            print(f"Loading configuration defaults from {config_path}")
            with open(path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            print(f"Warning: Configuration file not found at {config_path}. Using code defaults and environment variables.")

        # cls.model_validate will parse the dict and also check environment variables
        return cls.model_validate(config_data)

