"""
4C Multiphysics Simulation Interface

Handles execution of 4C FEM simulations and result parsing.
"""

import os
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import time


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
    
    Supports both local executable and Docker-based execution.
    """
    
    DOCKER_IMAGE = "ghcr.io/4c-multiphysics/4c:main"
    DOCKER_FOURC_PATH = "/home/user/4C/build/4C"
    
    def __init__(
        self,
        fourc_executable: Optional[str] = None,
        working_directory: Optional[Path] = None,
        default_timeout: float = 600.0,
        use_docker: bool = False,
    ):
        """
        Initialize 4C simulator.
        
        Args:
            fourc_executable: Path to 4C executable (searches PATH if None)
            working_directory: Default working directory for simulations
            default_timeout: Default timeout for simulations (seconds)
            use_docker: If True, run simulations in Docker container
        """
        self.use_docker = use_docker
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
        
        # Prepare command based on execution mode
        if self.use_docker:
            # Docker execution - mount working directory
            # NOTE: This path is rarely used now (we're usually in the 4C image)
            work_dir_abs = self.working_directory.resolve()
            yaml_rel = config.yaml_input_path.resolve().relative_to(work_dir_abs)
            # 4C command format: 4C input.yaml output_name (no -o flag)
            output_name = config.output_directory.name
            cmd = [
                "docker", "run", "--rm",
                "--platform", "linux/amd64",  # Required for Apple Silicon
                "-v", f"{work_dir_abs}:/workspace",
                "-w", "/workspace",
                self.DOCKER_IMAGE,
                self.DOCKER_FOURC_PATH,
                str(yaml_rel),
                output_name,
            ]
        else:
            # Local execution (we're in the 4C image)
            # 4C command format: 4C input.yaml output_name
            # Output files go to same directory as input YAML file
            output_name = config.output_directory.name
            cmd = [
                self.fourc_executable,
                str(config.yaml_input_path),
                output_name,
            ]
        
        if config.verbose:
            cmd.append("-v")
        
        # Run simulation
        try:
            # Set up environment with LD_LIBRARY_PATH for 4C libraries
            env = os.environ.copy()
            if 'LD_LIBRARY_PATH' not in env or not env['LD_LIBRARY_PATH']:
                env['LD_LIBRARY_PATH'] = '/home/user/4C/build:/usr/local/lib:/usr/lib/x86_64-linux-gnu'
            else:
                env['LD_LIBRARY_PATH'] = '/home/user/4C/build:/usr/local/lib:/usr/lib/x86_64-linux-gnu:' + env['LD_LIBRARY_PATH']
            
            # Run 4C in the same directory as the YAML file (4C expects output in same dir)
            yaml_dir = config.yaml_input_path.parent
            t0 = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(yaml_dir),
                capture_output=True,
                text=True,
                timeout=config.timeout,
                env=env,
            )
            
            # 4C "output_name" is often used as an output *prefix* (not necessarily a directory).
            # Historically we assumed it was a directory (yaml_dir/output_name), but some 4C
            # configurations write VTU/VTK directly next to the input YAML file.
            #
            # So we search both:
            # - yaml_dir / output_name (directory style)
            # - yaml_dir (prefix style)
            actual_output_dir = yaml_dir / output_name
            
            # Check for errors
            if result.returncode != 0:
                # Combine stdout and stderr for comprehensive error message
                # 4C often outputs critical errors to stdout (e.g., MPI_ABORT messages)
                error_msg = ""
                if result.stdout:
                    error_msg += f"STDOUT:\n{result.stdout}\n"
                if result.stderr:
                    error_msg += f"STDERR:\n{result.stderr}\n"
                if not error_msg:
                    error_msg = f"4C simulation failed with return code {result.returncode}"
                
                # Log full error for debugging
                print(f"4C simulation failed (returncode={result.returncode}):")
                print(error_msg)
                
                return SimulationResult(
                    success=False,
                    output_directory=actual_output_dir if actual_output_dir.exists() else config.output_directory,
                    max_von_mises_stress=0.0,
                    max_displacement=0.0,
                    max_principal_strain=0.0,
                    converged=False,
                    num_iterations=0,
                    residual_norm=0.0,
                    error_message=error_msg,
                    log_path=actual_output_dir / "fourc.log" if actual_output_dir.exists() else config.output_directory / "fourc.log",
                )
            
            # Parse results. Note: 4C may write outputs either into yaml_dir/output_name
            # or into yaml_dir with filenames prefixed by output_name.
            return self._parse_results(
                output_directory=actual_output_dir,
                output_name=output_name,
                yaml_dir=yaml_dir,
                run_start_time=t0,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
            
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
    
    def _parse_results(
        self,
        output_directory: Path,
        output_name: str,
        yaml_dir: Path,
        run_start_time: float,
        stdout: str,
        stderr: str,
    ) -> SimulationResult:
        """
        Parse 4C simulation results.
        
        This is a placeholder - in production, would parse VTK/VTU files
        to extract stress, displacement, strain fields.
        
        Args:
            output_directory: Directory containing simulation outputs
            
        Returns:
            SimulationResult object
        """
        def _is_new_enough(p: Path) -> bool:
            try:
                return p.stat().st_mtime >= (run_start_time - 1.0)
            except Exception:
                return True

        # 1) Prefer directory-style outputs (yaml_dir/output_name)
        vtk_files = list(output_directory.glob("*.vtk")) if output_directory.exists() else []
        vtu_files = list(output_directory.glob("*.vtu")) if output_directory.exists() else []
        pvtu_files = list(output_directory.glob("*.pvtu")) if output_directory.exists() else []
        result_files = vtk_files + vtu_files + pvtu_files

        # 2) If none found, search yaml_dir for prefix-style outputs: output_name*.{vtu,vtk,pvtu}
        if not result_files:
            for ext in ("*.vtu", "*.vtk", "*.pvtu"):
                for p in yaml_dir.glob(ext):
                    if p.name.startswith(output_name) and _is_new_enough(p):
                        result_files.append(p)

        # 3) Final fallback: any new VTU/VTK produced under yaml_dir (depth 2) after run start
        if not result_files:
            try:
                for p in yaml_dir.rglob("*.vtu"):
                    if _is_new_enough(p):
                        result_files.append(p)
                for p in yaml_dir.rglob("*.vtk"):
                    if _is_new_enough(p):
                        result_files.append(p)
                for p in yaml_dir.rglob("*.pvtu"):
                    if _is_new_enough(p):
                        result_files.append(p)
            except Exception:
                pass
        
        # Look for JSON summary if available (directory-style)
        json_file = output_directory / "results.json"
        if json_file.exists():
            return self._parse_json_results(json_file)
        
        # If no result files, don't automatically call it a failure.
        # 4C can exit successfully even if runtime VTK output is disabled/misconfigured.
        # We return success=True but with zeroed quantities and attach stdout/stderr
        # to help diagnose.
        if not result_files:
            return SimulationResult(
                success=True,
                output_directory=yaml_dir,
                max_von_mises_stress=0.0,
                max_displacement=0.0,
                max_principal_strain=0.0,
                converged=True,
                num_iterations=0,
                residual_norm=0.0,
                error_message=None,
                metadata={
                    "result_files": [],
                    "note": "4C exited successfully but no VTU/VTK files were discovered.",
                    "stdout_tail": stdout[-2000:] if stdout else "",
                    "stderr_tail": stderr[-2000:] if stderr else "",
                },
            )
        
        # In production, would parse VTK/VTU files here. For now, return placeholders and
        # provide discovered result files in metadata for downstream parsers.
        return SimulationResult(
            success=True,
            output_directory=output_directory if output_directory.exists() else yaml_dir,
            max_von_mises_stress=0.0,  # Would extract from VTK
            max_displacement=0.0,  # Would extract from VTK
            max_principal_strain=0.0,  # Would extract from VTK
            converged=True,  # Would check log files
            num_iterations=0,  # Would parse from log
            residual_norm=0.0,  # Would parse from log
            metadata={
                "result_files": [str(f) for f in sorted(set(result_files))],
                "stdout_tail": stdout[-2000:] if stdout else "",
                "stderr_tail": stderr[-2000:] if stderr else "",
            },
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



