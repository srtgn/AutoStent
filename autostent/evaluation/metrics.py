"""
Biomechanical Metrics

Compute performance metrics from FEM simulation results.
"""

from dataclasses import dataclass
from typing import Dict, Any

from ..simulation import SimulationResult


@dataclass
class BiomechanicalMetrics:
    """Biomechanical performance metrics."""
    
    max_stress: float  # MPa
    max_displacement: float  # mm
    max_strain: float  # dimensionless
    converged: bool
    safety_factor: float  # ratio of yield stress to max stress
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_stress": self.max_stress,
            "max_displacement": self.max_displacement,
            "max_strain": self.max_strain,
            "converged": self.converged,
            "safety_factor": self.safety_factor,
        }


def compute_biomechanical_metrics(
    result: SimulationResult,
    yield_stress: float = 400.0,  # MPa - typical for Nitinol
) -> Dict[str, float]:
    """
    Compute biomechanical metrics from simulation result.
    
    Args:
        result: Simulation result
        yield_stress: Material yield stress (MPa)
        
    Returns:
        Dictionary of metrics
    """
    if not result.success:
        return {
            "max_stress": 0.0,
            "max_displacement": 0.0,
            "max_strain": 0.0,
            "converged": False,
            "safety_factor": 0.0,
        }
    
    # Safety factor (higher is better)
    safety_factor = yield_stress / max(result.max_von_mises_stress, 1e-6)
    
    return {
        "max_stress": result.max_von_mises_stress,
        "max_displacement": result.max_displacement,
        "max_strain": result.max_principal_strain,
        "converged": result.converged,
        "safety_factor": safety_factor,
    }


