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

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    # Don't print loudly; meshio is optional


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
    try:
        # Prefer PyVista when available, but fall back to meshio (headless-friendly)
        if PYVISTA_AVAILABLE:
            mesh = pv.read(str(vtu_path))
            point_data = dict(mesh.point_data)
            cell_data = dict(mesh.cell_data) if hasattr(mesh, "cell_data") else {}
        else:
            if not MESHIO_AVAILABLE:
                print("Neither PyVista nor meshio available - cannot parse VTU")
                return 0.0, 0.0, 0.0, False
            m = meshio.read(str(vtu_path))
            point_data = m.point_data or {}
            # meshio cell_data is dict[str, list[np.ndarray]] (one per cell block)
            cell_data = m.cell_data or {}

        # Case-insensitive lookup helper
        pd_lower = {k.lower(): k for k in point_data.keys()}
        cd_lower = {k.lower(): k for k in cell_data.keys()}

        def _get_point_array(*names: str):
            for n in names:
                key = pd_lower.get(n.lower())
                if key is not None:
                    return np.asarray(point_data[key])
            return None

        def _get_cell_array(*names: str):
            for n in names:
                key = cd_lower.get(n.lower())
                if key is not None:
                    blocks = cell_data[key]
                    # flatten blocks to one array if possible
                    if isinstance(blocks, list) and blocks:
                        try:
                            return np.concatenate([np.asarray(b) for b in blocks], axis=0)
                        except Exception:
                            return np.asarray(blocks[0])
                    return np.asarray(blocks)
            return None
        
        # Extract stress - try multiple 4C naming conventions
        max_stress = 0.0
        # Print available arrays for debugging
        print(f"VTU point arrays: {list(point_data.keys())}")
        print(f"VTU cell arrays: {list(cell_data.keys())}")
        
        stress_tensor = _get_point_array(
            "stress", "cauchy", "cauchy_stress", "sigma", 
            "element_cauchy_stress_xyz", "nodal_cauchy_stress_xyz",
            "Cauchy", "Stress", "vonMises", "von_mises"
        ) or _get_cell_array(
            "stress", "cauchy", "cauchy_stress", "sigma",
            "element_cauchy_stress_xyz", "Cauchy", "Stress"
        )
        if stress_tensor is not None:
            stress_tensor = np.asarray(stress_tensor)
            print(f"Found stress tensor with shape: {stress_tensor.shape}")
            if stress_tensor.ndim == 3 and stress_tensor.shape[1:] == (3, 3):
                stress_tensor = stress_tensor.reshape(stress_tensor.shape[0], 9)
            if stress_tensor.ndim == 2 and stress_tensor.shape[1] in (6, 9):
                von_mises = compute_von_mises_stress(stress_tensor)
                max_stress = float(np.max(von_mises))
            elif stress_tensor.ndim == 1:
                # Already scalar (e.g., von Mises directly)
                max_stress = float(np.max(stress_tensor))
        else:
            print("No stress array found in VTU")
        
        # Extract displacement
        max_disp = 0.0
        disp = _get_point_array("displacement", "disp", "u")
        if disp is not None:
            disp = np.asarray(disp)
            if disp.ndim == 1:
                max_disp = float(np.max(np.abs(disp)))
            else:
                max_disp = float(np.max(np.linalg.norm(disp, axis=1)))
        
        # Extract strain
        max_strain = 0.0
        strain_tensor = _get_point_array("strain", "gl_strain") or _get_cell_array("strain", "gl_strain")
        if strain_tensor is not None:
            strain_tensor = np.asarray(strain_tensor)
            if strain_tensor.ndim == 3 and strain_tensor.shape[1:] == (3, 3):
                strain_tensor = strain_tensor.reshape(strain_tensor.shape[0], 9)
            if strain_tensor.ndim == 2 and strain_tensor.shape[1] in (6, 9):
                principal_strain = compute_principal_strain(strain_tensor)
                max_strain = float(np.max(principal_strain))
        
        # We consider parsing successful if we got at least displacement or stress
        ok = (max_disp > 0.0) or (max_stress > 0.0) or (max_strain > 0.0)
        return max_stress, max_disp, max_strain, ok
        
    except Exception as e:
        print(f"Error parsing VTU file: {e}")
        return 0.0, 0.0, 0.0, False


def find_latest_vtu(output_dir: Path) -> Optional[Path]:
    """Find the latest VTU file in output directory."""
    # Direct hits
    vtu_files = list(output_dir.glob("*.vtu"))
    vtk_files = list(output_dir.glob("*.vtk"))

    # Recursive hits (4C often writes into <output_name>-vtk-files/)
    if not vtu_files and not vtk_files:
        vtu_files = list(output_dir.rglob("*.vtu"))
        vtk_files = list(output_dir.rglob("*.vtk"))

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
