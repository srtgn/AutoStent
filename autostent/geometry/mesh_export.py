"""
Mesh Export Utilities

Export spline-based geometries to formats compatible with 4C meshing.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import pyvista as pv

from .spline_stent import SplineStentGeometry


def export_to_4c_mesh(
    geometry: SplineStentGeometry,
    output_path: Path,
    mesh_resolution: float = 0.5,
    format: str = "vtk"
) -> Path:
    """
    Export stent geometry to mesh format compatible with 4C.
    
    Args:
        geometry: SplineStentGeometry instance
        output_path: Output file path
        mesh_resolution: Target mesh element size (mm)
        format: Output format ('vtk', 'stl', 'vtp')
        
    Returns:
        Path to exported mesh file
    """
    # Sample points from spline curves
    points = geometry.sample_points(num_points_per_strut=100)
    
    # Flatten to single point cloud
    all_points = points.reshape(-1, 3)
    
    # Create point cloud
    point_cloud = pv.PolyData(all_points)
    
    # Generate surface mesh (simplified - in production would use proper
    # meshing algorithm like Delaunay or marching cubes)
    # For now, create a tube around each strut centerline
    
    mesh = pv.PolyData()
    
    for strut_points in points:
        # Create tube around centerline
        tube = pv.Tube(strut_points, radius=geometry.params.strut_width / 2)
        if mesh.n_points == 0:
            mesh = tube
        else:
            mesh = mesh + tube
    
    # Merge and clean
    mesh = mesh.clean()
    
    # Ensure correct extension
    output_path = Path(output_path)
    if format == "vtk":
        if output_path.suffix != ".vtk":
            output_path = output_path.with_suffix(".vtk")
    elif format == "stl":
        if output_path.suffix != ".stl":
            output_path = output_path.with_suffix(".stl")
    elif format == "vtp":
        if output_path.suffix != ".vtp":
            output_path = output_path.with_suffix(".vtp")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Write mesh
    mesh.save(str(output_path))
    
    return output_path

