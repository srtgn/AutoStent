"""
Patient Data Handling

Abstracted interface for patient-specific data (vessel geometry, etc.).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class PatientData:
    """Patient-specific data for stent design."""
    
    patient_id: str
    vessel_centerline: np.ndarray  # (N, 3) array of centerline points
    vessel_diameter: float  # mm - average vessel diameter
    aneurysm_location: Optional[float] = None  # position along centerline (0-1)
    aneurysm_diameter: Optional[float] = None  # mm
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "patient_id": self.patient_id,
            "vessel_centerline": self.vessel_centerline.tolist(),
            "vessel_diameter": self.vessel_diameter,
            "aneurysm_location": self.aneurysm_location,
            "aneurysm_diameter": self.aneurysm_diameter,
        }


def load_patient_data(data_path: Path) -> PatientData:
    """
    Load patient data from file.
    
    This is a placeholder - in production would load from DICOM,
    STL, or other medical imaging formats.
    
    Args:
        data_path: Path to patient data file
        
    Returns:
        PatientData object
    """
    # Placeholder implementation
    # In production, would parse actual medical imaging data
    
    # For now, create synthetic centerline
    centerline = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0],
        [0.0, 0.0, 20.0],
        [0.0, 0.0, 30.0],
    ])
    
    return PatientData(
        patient_id="synthetic_001",
        vessel_centerline=centerline,
        vessel_diameter=10.0,
        aneurysm_location=0.5,
        aneurysm_diameter=15.0,
    )




