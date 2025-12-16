"""
End-to-End Design Pipeline

Complete workflow from patient data to optimized stent design.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from ..data import PatientData, load_patient_data
from ..geometry import SplineStentGeometry, StentGeometryParameters, export_to_4c_mesh
from ..simulation import FourCSimulator, SimulationConfig
from ..automation import FourCYAMLGenerator
from ..evaluation import compute_biomechanical_metrics


class DesignPipeline:
    """
    End-to-end pipeline for patient-specific stent design.
    
    Workflow:
    1. Load patient data
    2. Generate initial stent geometry
    3. Run FEM simulation
    4. Evaluate performance
    5. (Optional) Optimize design
    """
    
    def __init__(
        self,
        fourc_executable: Optional[str] = None,
        working_directory: Optional[Path] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            fourc_executable: Path to 4C executable
            working_directory: Working directory for pipeline execution
        """
        self.simulator = FourCSimulator(fourc_executable=fourc_executable)
        self.yaml_generator = FourCYAMLGenerator()
        self.working_directory = working_directory or Path.cwd() / "pipeline_output"
        self.working_directory.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        patient_data: PatientData,
        initial_params: Optional[StentGeometryParameters] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run complete design pipeline.
        
        Args:
            patient_data: Patient-specific data
            initial_params: Initial stent parameters (optional)
            output_dir: Output directory (optional)
            
        Returns:
            Dictionary with results and metrics
        """
        if output_dir is None:
            output_dir = self.working_directory / patient_data.patient_id
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate initial parameters if not provided
        if initial_params is None:
            initial_params = StentGeometryParameters(
                diameter=patient_data.vessel_diameter * 1.1,  # 10% oversizing
                length=30.0,  # mm
                strut_thickness=0.1,  # mm
                strut_width=0.15,  # mm
                num_struts=12,
                crown_height=1.0,  # mm
                crown_radius=0.5,  # mm
            )
        
        # Generate geometry
        geometry = SplineStentGeometry(initial_params)
        
        # Export mesh
        mesh_path = output_dir / "stent_geometry.vtk"
        export_to_4c_mesh(geometry, mesh_path)
        
        # Generate YAML
        yaml_path = output_dir / "simulation.4C.yaml"
        self.yaml_generator.generate(
            geometry_params=initial_params,
            mesh_path=mesh_path,
            output_path=yaml_path,
        )
        
        # Run simulation
        config = SimulationConfig(
            yaml_input_path=yaml_path,
            output_directory=output_dir / "results",
            timeout=600.0,
        )
        
        result = self.simulator.run_simulation(config)
        
        # Compute metrics
        metrics = compute_biomechanical_metrics(result)
        
        return {
            "patient_id": patient_data.patient_id,
            "parameters": {
                "diameter": initial_params.diameter,
                "length": initial_params.length,
                "strut_thickness": initial_params.strut_thickness,
                "strut_width": initial_params.strut_width,
                "num_struts": initial_params.num_struts,
                "crown_height": initial_params.crown_height,
                "crown_radius": initial_params.crown_radius,
            },
            "simulation_result": result.to_dict(),
            "metrics": metrics,
            "output_directory": str(output_dir),
        }


