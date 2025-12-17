"""
VTU Result Parser for 4C Simulations

Parses VTK/VTU output files to extract stress, displacement, and strain results.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("WARNING: PyVista not available - VTU parsing disabled")


def compute_von_mises_stress(stress_tensor: np.ndarray) -> np.ndarray:
    """
    Compute von Mises stress from stress tensor.
    
    Args:
        stress_tensor: (N, 6) array of stress components [σxx, σyy, σzz, σxy, σyz, σxz]
        
    Returns:
        von_mises: (N,) array of von Mises stress values
    """
    if stress_tensor.shape[1] == 6:
        sxx, syy, szz, sxy, syz, sxz = stress_tensor.T
    elif stress_tensor.shape[1] == 9:
        # Full tensor format
        sxx = stress_tensor[:, 0]
        syy = stress_tensor[:, 4]
        szz = stress_tensor[:, 8]
        sxy = stress_tensor[:, 1]
        syz = stress_tensor[:, 5]
        sxz = stress_tensor[:, 2]
    else:
        raise ValueError(f"Unexpected stress tensor shape: {stress_tensor.shape}")
    
    # Von Mises: sqrt(0.5 * ((sxx-syy)² + (syy-szz)² + (szz-sxx)² + 6*(sxy² + syz² + sxz²)))
    von_mises = np.sqrt(
        0.5 * (
            (sxx - syy)**2 + 
            (syy - szz)**2 + 
            (szz - sxx)**2 + 
            6 * (sxy**2 + syz**2 + sxz**2)
        )
    )
    
    return von_mises


def compute_principal_strain(strain_tensor: np.ndarray) -> np.ndarray:
    """
    Compute maximum principal strain from strain tensor.
    
    Args:
        strain_tensor: (N, 6) array of strain components
        
    Returns:
        principal_strain: (N,) array of max principal strain
    """
    if strain_tensor.shape[1] == 6:
        exx, eyy, ezz, exy, eyz, exz = strain_tensor.T
    else:
        exx = strain_tensor[:, 0]
        eyy = strain_tensor[:, 4]
        ezz = strain_tensor[:, 8]
        exy = strain_tensor[:, 1]
        eyz = strain_tensor[:, 5]
        exz = strain_tensor[:, 2]
    
    # For each point, compute eigenvalues of strain tensor
    n_points = len(exx)
    principal_strains = np.zeros(n_points)
    
    for i in range(n_points):
        # Build  strain tensor matrix
        tensor = np.array([
            [exx[i], exy[i], exz[i]],
            [exy[i], eyy[i], eyz[i]],
            [exz[i], eyz[i], ezz[i]]
        ])
        eigenvalues = np.linalg.eigvalsh(tensor)
        principal_strains[i] = np.max(np.abs(eigenvalues))
    
    return principal_strains


def parse_vtu_file(vtu_path: Path) -> Tuple[float, float, float, bool]:
    """
    Parse 4C VTU output file.
    
    Args:
        vtu_path: Path to VTU file
        
    Returns:
        max_von_mises_stress: Maximum von Mises stress (MPa)
        max_displacement: Maximum displacement magnitude (mm)
        max_principal_strain: Maximum principal strain
        success: Whether parsing succeeded
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available - returning placeholder values")
        return 0.0, 0.0, 0.0, False
    
    try:
        # Read VTU file
        mesh = pv.read(str(vtu_path))
        
        # Extract stress
        max_stress = 0.0
        if 'stress' in mesh.point_data:
            stress_tensor = mesh.point_data['stress']
            von_mises = compute_von_mises_stress(stress_tensor)
            max_stress = float(np.max(von_mises))
        elif 'cauchy' in mesh.point_data:
            stress_tensor = mesh.point_data['cauchy']
            von_mises = compute_von_mises_stress(stress_tensor)
            max_stress = float(np.max(von_mises))
        
        # Extract displacement
        max_disp = 0.0
        if 'displacement' in mesh.point_data:
            disp = mesh.point_data['displacement']
            if disp.ndim == 1:
                max_disp = float(np.max(np.abs(disp)))
            else:
                disp_magnitude = np.linalg.norm(disp, axis=1)
                max_disp = float(np.max(disp_magnitude))
        elif 'disp' in mesh.point_data:
            disp = mesh.point_data['disp']
            disp_magnitude = np.linalg.norm(disp, axis=1)
            max_disp = float(np.max(disp_magnitude))
        
        # Extract strain
        max_strain = 0.0
        if 'strain' in mesh.point_data:
            strain_tensor = mesh.point_data['strain']
            principal_strain = compute_principal_strain(strain_tensor)
            max_strain = float(np.max(principal_strain))
        elif 'GL_strain' in mesh.point_data:
            strain_tensor = mesh.point_data['GL_strain']
            principal_strain = compute_principal_strain(strain_tensor)
            max_strain = float(np.max(principal_strain))
        
        return max_stress, max_disp, max_strain, True
        
    except Exception as e:
        print(f"Error parsing VTU file: {e}")
        return 0.0, 0.0, 0.0, False


def find_latest_vtu(output_dir: Path) -> Optional[Path]:
    """Find the latest VTU file in output directory."""
    vtu_files = sorted(output_dir.glob("*.vtu"))
    vtk_files = sorted(output_dir.glob("*.vtk"))
    
    all_files = vtu_files + vtk_files
    if not all_files:
        return None
    
    # Return latest by modification time
    return max(all_files, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    # Test VTU parsing
    import sys
    
    if len(sys.argv) > 1:
        vtu_path = Path(sys.argv[1])
        if vtu_path.exists():
            stress, disp, strain, success = parse_vtu_file(vtu_path)
            print(f"Parsed {vtu_path}:")
            print(f"  Max von Mises stress: {stress:.2f} MPa")
            print(f"  Max displacement: {disp:.4f} mm")
            print(f"  Max principal strain: {strain:.6f}")
            print(f"  Success: {success}")
        else:
            print(f"File not found: {vtu_path}")
    else:
        print("Usage: python vtu_parser.py <path_to_vtu_file>")
