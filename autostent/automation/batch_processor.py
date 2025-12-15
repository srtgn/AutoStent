"""
Batch Processing Utilities

Handles batch execution of multiple simulations for parameter sweeps
and RL rollouts.
"""

from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
import json
from dataclasses import asdict

from ..geometry import StentGeometryParameters, SplineStentGeometry, export_to_4c_mesh
from ..simulation import FourCSimulator, SimulationConfig, SimulationResult
from .yaml_generator import FourCYAMLGenerator


class BatchProcessor:
    """
    Batch processor for running multiple simulations.
    
    Supports:
    - Parameter sweeps
    - RL rollouts
    - Design of experiments
    """
    
    def __init__(
        self,
        fourc_executable: Optional[str] = None,
        working_directory: Optional[Path] = None,
    ):
        """
        Initialize batch processor.
        
        Args:
            fourc_executable: Path to 4C executable
            working_directory: Working directory for batch jobs
        """
        self.simulator = FourCSimulator(fourc_executable=fourc_executable)
        self.yaml_generator = FourCYAMLGenerator()
        self.working_directory = working_directory or Path.cwd() / "batch_output"
        self.working_directory.mkdir(parents=True, exist_ok=True)
    
    def run_parameter_sweep(
        self,
        parameter_ranges: Dict[str, List[float]],
        base_params: StentGeometryParameters,
        output_dir: Optional[Path] = None,
    ) -> List[SimulationResult]:
        """
        Run parameter sweep over specified ranges.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to value lists
            base_params: Base parameters to vary
            output_dir: Output directory (default: working_directory)
            
        Returns:
            List of simulation results
        """
        if output_dir is None:
            output_dir = self.working_directory / "parameter_sweep"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate parameter combinations
        combinations = self._generate_combinations(parameter_ranges)
        
        results = []
        
        for i, combo in enumerate(combinations):
            # Create modified parameters
            params_dict = asdict(base_params)
            params_dict.update(combo)
            params = StentGeometryParameters(**params_dict)
            
            # Generate geometry and run simulation
            result = self._run_single_simulation(
                params,
                output_dir / f"sim_{i:04d}",
            )
            
            results.append(result)
        
        return results
    
    def _generate_combinations(self, ranges: Dict[str, List[float]]) -> List[Dict[str, float]]:
        """Generate all combinations of parameter values."""
        import itertools
        
        keys = list(ranges.keys())
        values = [ranges[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _run_single_simulation(
        self,
        params: StentGeometryParameters,
        output_dir: Path,
    ) -> SimulationResult:
        """Run a single simulation."""
        # Generate geometry
        geometry = SplineStentGeometry(params)
        
        # Export mesh
        mesh_path = output_dir / "geometry.vtk"
        export_to_4c_mesh(geometry, mesh_path)
        
        # Generate YAML
        yaml_path = output_dir / "input.4C.yaml"
        self.yaml_generator.generate(
            geometry_params=params,
            mesh_path=mesh_path,
            output_path=yaml_path,
        )
        
        # Run simulation
        config = SimulationConfig(
            yaml_input_path=yaml_path,
            output_directory=output_dir / "results",
            timeout=600.0,
        )
        
        return self.simulator.run_simulation(config)

