"""
4C FEM Simulation Interface Module

Provides interface to 4C Multiphysics solver for running FEM simulations
and parsing results.
"""

from .fourc_interface import FourCSimulator, SimulationConfig, SimulationResult
from .result_parser import parse_4c_results

__all__ = [
    "FourCSimulator",
    "SimulationConfig",
    "SimulationResult",
    "parse_4c_results",
]

