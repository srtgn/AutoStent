"""
4C Multiphysics Simulation Interface

Handles execution of 4C FEM simulations and result parsing.
"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for a 4C simulation."""
    
    yaml_input_path: Path
    output_directory: Path
    fourc_executable: str = "fourc"
    timeout: float = 600.0  # seconds
    num_threads: int = 1
    verbose: bool = False
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        if not self.yaml_input_path.exists():
            errors.append(f"YAML input file not found: {self.yaml_input_path}")
        if self.timeout <= 0:
            errors.append("timeout must be positive")
        if self.num_threads < 1:
            errors.append("num_threads must be >= 1")
        return errors


@dataclass
class SimulationResult:
    """Results from a 4C FEM simulation."""
    
    success: bool
    output_directory: Path
    
    # Mechanical quantities
    max_von_mises_stress: float  # MPa
    max_displacement: float  # mm
    max_principal_strain: float  # dimensionless
    
    # Convergence information
    converged: bool
    num_iterations: int
    residual_norm: float
    
    # Error information
    error_message: Optional[str] = None
    log_path: Optional[Path] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output_directory": str(self.output_directory),
            "max_von_mises_stress": self.max_von_mises_stress,
            "max_displacement": self.max_displacement,
            "max_principal_strain": self.max_principal_strain,
            "converged": self.converged,
            "num_iterations": self.num_iterations,
            "residual_norm": self.residual_norm,
            "error_message": self.error_message,
            "log_path": str(self.log_path) if self.log_path else None,
            "metadata": self.metadata or {},
        }


class FourCSimulator:
    """
    Interface to 4C Multiphysics solver.
    
    Handles:
    - Executing 4C simulations from YAML input files
    - Parsing simulation results
    - Error handling and validation
    - Logging and metadata tracking
    """
    
    def __init__(
        self,
        fourc_executable: Optional[str] = None,
        working_directory: Optional[Path] = None,
        default_timeout: float = 600.0,
    ):
        """
        Initialize 4C simulator.
        
        Args:
            fourc_executable: Path to 4C executable (searches PATH if None)
            working_directory: Default working directory for simulations
            default_timeout: Default timeout for simulations (seconds)
        """
        self.fourc_executable = fourc_executable or self._find_fourc_executable()
        self.working_directory = working_directory or Path.cwd()
        self.default_timeout = default_timeout
        
        self.working_directory.mkdir(parents=True, exist_ok=True)
    
    def _find_fourc_executable(self) -> str:
        """
        Try to find 4C executable in PATH.
        
        Returns:
            Executable name or path
        """
        possible_names = ["fourc", "4c"]
        
        for name in possible_names:
            try:
                result = subprocess.run(
                    ["which", name],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                continue
        
        # Default - user must ensure it's in PATH
        return "fourc"
    
    def run_simulation(
        self,
        config: SimulationConfig,
    ) -> SimulationResult:
        """
        Run a 4C simulation.
        
        Args:
            config: Simulation configuration
            
        Returns:
            SimulationResult object
        """
        # Validate configuration
        errors = config.validate()
        if errors:
            return SimulationResult(
                success=False,
                output_directory=config.output_directory,
                max_von_mises_stress=0.0,
                max_displacement=0.0,
                max_principal_strain=0.0,
                converged=False,
                num_iterations=0,
                residual_norm=0.0,
                error_message=f"Configuration errors: {', '.join(errors)}",
            )
        
        # Create output directory
        config.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Prepare command
        cmd = [
            self.fourc_executable,
            str(config.yaml_input_path),
            "-o",
            str(config.output_directory),
        ]
        
        if config.verbose:
            cmd.append("-v")
        
        # Run simulation
        try:
            result = subprocess.run(
                cmd,
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=config.timeout,
            )
            
            # Check for errors
            if result.returncode != 0:
                return SimulationResult(
                    success=False,
                    output_directory=config.output_directory,
                    max_von_mises_stress=0.0,
                    max_displacement=0.0,
                    max_principal_strain=0.0,
                    converged=False,
                    num_iterations=0,
                    residual_norm=0.0,
                    error_message=result.stderr,
                    log_path=config.output_directory / "fourc.log",
                )
            
            # Parse results
            return self._parse_results(config.output_directory)
            
        except subprocess.TimeoutExpired:
            return SimulationResult(
                success=False,
                output_directory=config.output_directory,
                max_von_mises_stress=0.0,
                max_displacement=0.0,
                max_principal_strain=0.0,
                converged=False,
                num_iterations=0,
                residual_norm=0.0,
                error_message=f"Simulation timed out after {config.timeout}s",
                log_path=config.output_directory / "fourc.log",
            )
        except Exception as e:
            return SimulationResult(
                success=False,
                output_directory=config.output_directory,
                max_von_mises_stress=0.0,
                max_displacement=0.0,
                max_principal_strain=0.0,
                converged=False,
                num_iterations=0,
                residual_norm=0.0,
                error_message=str(e),
                log_path=config.output_directory / "fourc.log",
            )
    
    def _parse_results(self, output_directory: Path) -> SimulationResult:
        """
        Parse 4C simulation results.
        
        This is a placeholder - in production, would parse VTK/VTU files
        to extract stress, displacement, strain fields.
        
        Args:
            output_directory: Directory containing simulation outputs
            
        Returns:
            SimulationResult object
        """
        # Look for result files
        vtk_files = list(output_directory.glob("*.vtk"))
        vtu_files = list(output_directory.glob("*.vtu"))
        result_files = vtk_files + vtu_files
        
        # Look for JSON summary if available
        json_file = output_directory / "results.json"
        if json_file.exists():
            return self._parse_json_results(json_file)
        
        # If no result files, simulation may have failed
        if not result_files:
            return SimulationResult(
                success=False,
                output_directory=output_directory,
                max_von_mises_stress=0.0,
                max_displacement=0.0,
                max_principal_strain=0.0,
                converged=False,
                num_iterations=0,
                residual_norm=0.0,
                error_message="No result files found in output directory",
            )
        
        # In production, would parse VTK/VTU files here
        # For now, return placeholder indicating files exist
        return SimulationResult(
            success=True,
            output_directory=output_directory,
            max_von_mises_stress=0.0,  # Would extract from VTK
            max_displacement=0.0,  # Would extract from VTK
            max_principal_strain=0.0,  # Would extract from VTK
            converged=True,  # Would check log files
            num_iterations=0,  # Would parse from log
            residual_norm=0.0,  # Would parse from log
            metadata={"result_files": [str(f) for f in result_files]},
        )
    
    def _parse_json_results(self, json_path: Path) -> SimulationResult:
        """Parse results from JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        
        return SimulationResult(
            success=data.get("success", True),
            output_directory=Path(data.get("output_directory", "")),
            max_von_mises_stress=data.get("max_von_mises_stress", 0.0),
            max_displacement=data.get("max_displacement", 0.0),
            max_principal_strain=data.get("max_principal_strain", 0.0),
            converged=data.get("converged", True),
            num_iterations=data.get("num_iterations", 0),
            residual_norm=data.get("residual_norm", 0.0),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )

