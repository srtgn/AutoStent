"""
Observation Space Definition

Defines the observation space for the RL environment based on simulation outputs.
"""

import numpy as np
from gymnasium import spaces
from typing import Dict, Any


class ObservationSpace:
    """
    Observation space for stent design RL environment.
    
    Observations include:
    - Design parameters (normalized)
    - Simulation-derived quantities (stress, displacement, etc.)
    - Convergence status
    """
    
    def __init__(
        self,
        include_design_params: bool = True,
        include_simulation_results: bool = True,
        include_convergence: bool = True,
    ):
        """
        Initialize observation space.
        
        Args:
            include_design_params: Include normalized design parameters
            include_simulation_results: Include simulation-derived quantities
            include_convergence: Include convergence status
        """
        self.include_design_params = include_design_params
        self.include_simulation_results = include_simulation_results
        self.include_convergence = include_convergence
        
        # Build observation dimension
        dim = 0
        
        if include_design_params:
            # 7 design parameters: diameter, length, strut_thickness, strut_width,
            # num_struts, crown_height, crown_radius
            dim += 7
        
        if include_simulation_results:
            # 3 simulation quantities: max_stress, max_displacement, max_strain
            dim += 3
        
        if include_convergence:
            # 2 convergence indicators: converged (binary), residual_norm
            dim += 2
        
        # Box space with normalized values [0, 1]
        self.space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(dim,),
            dtype=np.float32,
        )
    
    def encode(
        self,
        design_params: Dict[str, float],
        simulation_results: Dict[str, float],
        converged: bool,
        residual_norm: float,
    ) -> np.ndarray:
        """
        Encode observations into vector.
        
        Args:
            design_params: Dictionary of design parameters
            simulation_results: Dictionary of simulation results
            converged: Whether simulation converged
            residual_norm: Residual norm (normalized)
            
        Returns:
            Observation vector
        """
        obs = []
        
        if self.include_design_params:
            # Normalize design parameters
            # Ranges are approximate - should be tuned based on design space
            ranges = {
                "diameter": (5.0, 15.0),  # mm
                "length": (10.0, 30.0),  # mm
                "strut_thickness": (0.05, 0.2),  # mm
                "strut_width": (0.1, 0.3),  # mm
                "num_struts": (8, 16),  # count
                "crown_height": (0.5, 2.0),  # mm
                "crown_radius": (0.2, 1.0),  # mm
            }
            
            param_order = [
                "diameter", "length", "strut_thickness", "strut_width",
                "num_struts", "crown_height", "crown_radius"
            ]
            
            for param in param_order:
                value = design_params[param]
                low, high = ranges[param]
                normalized = (value - low) / (high - low)
                normalized = np.clip(normalized, 0.0, 1.0)
                obs.append(normalized)
        
        if self.include_simulation_results:
            # Normalize simulation results
            # Ranges are approximate - should be tuned
            stress_norm = np.clip(simulation_results.get("max_stress", 0.0) / 500.0, 0.0, 1.0)
            disp_norm = np.clip(simulation_results.get("max_displacement", 0.0) / 10.0, 0.0, 1.0)
            strain_norm = np.clip(simulation_results.get("max_strain", 0.0) / 0.1, 0.0, 1.0)
            
            obs.extend([stress_norm, disp_norm, strain_norm])
        
        if self.include_convergence:
            obs.append(1.0 if converged else 0.0)
            obs.append(np.clip(residual_norm / 1e-3, 0.0, 1.0))
        
        return np.array(obs, dtype=np.float32)


