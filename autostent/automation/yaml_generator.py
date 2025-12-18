"""
4C YAML Input File Generator

Generates valid 4C YAML input files from geometry and simulation parameters.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from ..geometry import StentGeometryParameters


class FourCYAMLGenerator:
    """
    Generator for 4C YAML input files.
    
    Creates valid 4C input files with:
    - Geometry references
    - Material properties
    - Boundary conditions
    - Solver settings
    """
    
    def __init__(self, template_path: Optional[Path] = None):
        """
        Initialize generator.
        
        Args:
            template_path: Path to template YAML file (optional)
        """
        self.template_path = template_path
    
    def generate(
        self,
        geometry_params: StentGeometryParameters,
        mesh_path: Path,
        output_path: Path,
        material_properties: Optional[Dict[str, float]] = None,
        boundary_conditions: Optional[Dict[str, Any]] = None,
        solver_settings: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Generate 4C YAML input file.
        
        Args:
            geometry_params: Stent geometry parameters
            mesh_path: Path to geometry mesh file
            output_path: Path to output YAML file
            material_properties: Material properties (optional)
            boundary_conditions: Boundary conditions (optional)
            solver_settings: Solver settings (optional)
            
        Returns:
            Path to generated YAML file
        """
        # Default material properties (Nitinol-like)
        if material_properties is None:
            material_properties = {
                "young_modulus": 200000.0,  # MPa
                "poisson_ratio": 0.3,
                "density": 6.45,  # g/cm³
            }
        
        # Default boundary conditions
        if boundary_conditions is None:
            boundary_conditions = {
                "radial_pressure": 0.0,  # MPa
                "axial_force": 0.0,  # N
            }
        
        # Default solver settings
        if solver_settings is None:
            solver_settings = {
                "solver_type": "static",
                "max_iterations": 1000,
                "convergence_tolerance": 1e-6,
            }
        
        # Build YAML structure
        yaml_content = {
            "TITLE": f"Stent design: diameter={geometry_params.diameter}mm, length={geometry_params.length}mm",
            "PROBLEM TYPE": {
                "PROBLEMTYPE": "Structure",
            },
            "IO": {
                "OUTPUT_SPRING": True,
                "STRUCT_STRESS": "Cauchy",
                "STRUCT_STRAIN": "GL",
                "VERBOSITY": "Standard",
                "WRITE_INITIAL_STATE": False,
            },
            "IO/RUNTIME VTK OUTPUT": {
                "INTERVAL_STEPS": 1,
                "OUTPUT_DATA_FORMAT": "binary",
            },
            "IO/RUNTIME VTK OUTPUT/STRUCTURE": {
                "OUTPUT_STRUCTURE": True,
                "DISPLACEMENT": True,
                "STRESS_STRAIN": True,
                "GAUSS_POINT_DATA_OUTPUT_TYPE": "nodes",
            },
            "SOLVER 1": {
                "SOLVER": "Superlu",
                "NAME": "Structure_Solver",
            },
            "STRUCTURAL DYNAMIC": {
                "INT_STRATEGY": "Standard",
                "DYNAMICTYPE": "Statics",
                "TIMESTEP": 10,
                "NUMSTEP": 5,
                "MAXTIME": 50,
                "TOLDISP": solver_settings["convergence_tolerance"],
                "TOLRES": solver_settings["convergence_tolerance"],
                "LOADLIN": True,
                "LINEAR_SOLVER": 1,
            },
            "MATERIALS": [
                {
                    "MAT": 1,
                    "ELAST_IsoNeoHooke": {
                        "MUE": {
                            "constant": material_properties["young_modulus"] / (2 * (1 + material_properties["poisson_ratio"])),
                        },
                    },
                },
                {
                    "MAT": 2,
                    "ELAST_VolSussmanBathe": {
                        "KAPPA": {
                            "constant": material_properties["young_modulus"] / (3 * (1 - 2 * material_properties["poisson_ratio"])),
                        },
                    },
                },
            ],
            "STRUCTURE GEOMETRY": {
                "FILE": str(mesh_path),
                "ELEMENT_BLOCKS": [
                    {
                        "ID": 1,
                        "SOLID": {
                            "HEX8": {
                                "MAT": 1,
                                "KINEM": "nonlinear",
                            },
                        },
                    },
                ],
            },
        }
        
        # Add boundary conditions if specified
        if boundary_conditions.get("radial_pressure", 0.0) != 0.0:
            yaml_content["DESIGN SURF NEUMANN CONDITIONS"] = [
                {
                    "E": 1,
                    "ENTITY_TYPE": "node_set_id",
                    "NUMDOF": 3,
                    "ONOFF": [1, 0, 0],
                    "VAL": [boundary_conditions["radial_pressure"], 0, 0],
                    "FUNCT": [0, 0, 0],
                    "TYPE": "orthopressure",
                },
            ]
        
        # Write YAML file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        return output_path





