"""
4C Result Parser

Parse VTK/VTU output files from 4C simulations to extract mechanical quantities.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pyvista as pv


def parse_4c_results(
    result_file: Path,
    quantities: Optional[list] = None
) -> Dict[str, Any]:
    """
    Parse 4C result file (VTK/VTU) to extract mechanical quantities.
    
    Args:
        result_file: Path to VTK or VTU file
        quantities: List of quantities to extract (default: all available)
        
    Returns:
        Dictionary with extracted quantities
    """
    if quantities is None:
        quantities = [
            "von_mises_stress",
            "displacement",
            "principal_strain",
            "pressure",
        ]
    
    try:
        mesh = pv.read(str(result_file))
    except Exception as e:
        return {"error": f"Failed to read result file: {e}"}
    
    results = {}
    
    # Extract point data
    point_data = mesh.point_data
    
    # Extract von Mises stress
    if "von_mises_stress" in quantities:
        if "von_mises_stress" in point_data:
            stress = point_data["von_mises_stress"]
            results["max_von_mises_stress"] = float(np.max(stress))
            results["mean_von_mises_stress"] = float(np.mean(stress))
            results["von_mises_stress_field"] = stress
    
    # Extract displacement
    if "displacement" in quantities:
        if "displacement" in point_data:
            disp = point_data["displacement"]
            disp_magnitude = np.linalg.norm(disp, axis=1)
            results["max_displacement"] = float(np.max(disp_magnitude))
            results["mean_displacement"] = float(np.mean(disp_magnitude))
            results["displacement_field"] = disp
    
    # Extract principal strain
    if "principal_strain" in quantities:
        if "principal_strain" in point_data:
            strain = point_data["principal_strain"]
            results["max_principal_strain"] = float(np.max(strain))
            results["mean_principal_strain"] = float(np.mean(strain))
            results["principal_strain_field"] = strain
    
    return results

