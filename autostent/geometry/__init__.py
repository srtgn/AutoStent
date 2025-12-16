"""
Spline-based Geometry Generation Module

Provides parametric spline representations for stent geometries with:
- Control point manipulation
- Curvature constraints
- Manufacturability validation
- Export to 4C-compatible mesh formats
"""

from .spline_stent import SplineStentGeometry, StentGeometryParameters
from .mesh_export import export_to_4c_mesh

__all__ = [
    "SplineStentGeometry",
    "StentGeometryParameters",
    "export_to_4c_mesh",
]


