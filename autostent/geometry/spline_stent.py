"""
Spline-based Stent Geometry Generator

Implements parametric B-spline representations for stent strut geometries.
Provides control over design parameters while maintaining geometric validity.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.interpolate import BSpline, make_interp_spline
from pathlib import Path


@dataclass
class StentGeometryParameters:
    """Parameters defining a stent geometry design."""
    
    # Core dimensions
    diameter: float  # mm - inner diameter of deployed stent
    length: float  # mm - axial length
    
    # Strut geometry
    strut_thickness: float  # mm - radial thickness
    strut_width: float  # mm - circumferential width
    num_struts: int  # number of struts around circumference
    
    # Crown geometry (connecting struts)
    crown_height: float  # mm - peak height of crown
    crown_radius: float  # mm - radius of curvature at crown
    
    # Spline control
    num_control_points: int = 8  # control points per strut
    spline_degree: int = 3  # B-spline degree
    
    def validate(self) -> List[str]:
        """Validate parameter constraints."""
        errors = []
        if self.diameter <= 0:
            errors.append("diameter must be positive")
        if self.length <= 0:
            errors.append("length must be positive")
        if self.strut_thickness <= 0:
            errors.append("strut_thickness must be positive")
        if self.strut_width <= 0:
            errors.append("strut_width must be positive")
        if self.num_struts < 4:
            errors.append("num_struts must be >= 4")
        if self.crown_height < 0:
            errors.append("crown_height must be non-negative")
        if self.crown_radius <= 0:
            errors.append("crown_radius must be positive")
        return errors


class SplineStentGeometry:
    """
    Parametric spline-based stent geometry generator.
    
    Uses B-splines to represent strut centerlines, allowing smooth
    parametric control while maintaining geometric constraints.
    """
    
    def __init__(self, parameters: StentGeometryParameters):
        """
        Initialize geometry generator.
        
        Args:
            parameters: Stent geometry parameters
        """
        errors = parameters.validate()
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")
        
        self.params = parameters
        self._control_points = None
        self._spline_curves = None
        
    def generate_control_points(self) -> np.ndarray:
        """
        Generate control points for spline representation.
        
        Returns:
            Array of shape (num_struts, num_control_points, 3) with
            control points in cylindrical coordinates (r, theta, z)
        """
        num_struts = self.params.num_struts
        num_cp = self.params.num_control_points
        
        # Angular spacing between struts
        dtheta = 2 * np.pi / num_struts
        
        # Generate control points for each strut
        control_points = []
        
        for i in range(num_struts):
            theta_base = i * dtheta
            strut_cp = []
            
            # Create control points along strut with crown variation
            for j in range(num_cp):
                z = (j / (num_cp - 1)) * self.params.length
                
                # Radial position varies with crown geometry
                # Crown at midpoint (z = length/2)
                z_norm = (z - self.params.length / 2) / (self.params.length / 2)
                crown_factor = np.exp(-(z_norm ** 2) / (2 * (self.params.crown_radius ** 2)))
                
                r = (self.params.diameter / 2) + self.params.strut_thickness / 2
                r += self.params.crown_height * crown_factor
                
                # Angular position (slight variation for strut width)
                theta = theta_base + (j / (num_cp - 1) - 0.5) * (self.params.strut_width / r)
                
                strut_cp.append([r, theta, z])
            
            control_points.append(strut_cp)
        
        self._control_points = np.array(control_points)
        return self._control_points
    
    def generate_spline_curves(self, num_points: int = 100) -> List[BSpline]:
        """
        Generate B-spline curves from control points.
        
        Args:
            num_points: Number of points to sample along each curve
            
        Returns:
            List of BSpline objects, one per strut
        """
        if self._control_points is None:
            self.generate_control_points()
        
        splines = []
        
        for strut_cp in self._control_points:
            # Convert cylindrical to Cartesian
            r = strut_cp[:, 0]
            theta = strut_cp[:, 1]
            z = strut_cp[:, 2]
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Create spline in 3D space
            points = np.column_stack([x, y, z])
            
            # Parameterize by arc length
            t = np.linspace(0, 1, len(points))
            
            # Create B-spline
            spline = make_interp_spline(t, points, k=self.params.spline_degree)
            splines.append(spline)
        
        self._spline_curves = splines
        return splines
    
    def sample_points(self, num_points_per_strut: int = 50) -> np.ndarray:
        """
        Sample points from spline curves.
        
        Args:
            num_points_per_strut: Points to sample along each strut
            
        Returns:
            Array of shape (num_struts, num_points_per_strut, 3) with
            Cartesian coordinates (x, y, z)
        """
        if self._spline_curves is None:
            self.generate_spline_curves()
        
        all_points = []
        t_samples = np.linspace(0, 1, num_points_per_strut)
        
        for spline in self._spline_curves:
            points = spline(t_samples)
            all_points.append(points)
        
        return np.array(all_points)
    
    def get_geometry_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of geometry.
        
        Returns:
            Tuple of (min_bounds, max_bounds) arrays of shape (3,)
        """
        points = self.sample_points()
        all_points = points.reshape(-1, 3)
        
        min_bounds = np.min(all_points, axis=0)
        max_bounds = np.max(all_points, axis=0)
        
        return min_bounds, max_bounds
    
    def mutate_control_points(
        self,
        mutation_scale: float = 0.1,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Mutate control points for RL exploration.
        
        Args:
            mutation_scale: Maximum relative change (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            Mutated control points array
        """
        if seed is not None:
            np.random.seed(seed)
        
        if self._control_points is None:
            self.generate_control_points()
        
        mutated = self._control_points.copy()
        
        # Apply Gaussian noise scaled by parameter ranges
        for i in range(len(mutated)):
            for j in range(len(mutated[i])):
                # Mutate radial position
                r_range = self.params.strut_thickness
                mutated[i, j, 0] += np.random.normal(0, mutation_scale * r_range)
                
                # Mutate angular position
                theta_range = 2 * np.pi / self.params.num_struts
                mutated[i, j, 1] += np.random.normal(0, mutation_scale * theta_range)
                
                # Mutate axial position
                z_range = self.params.length
                mutated[i, j, 2] += np.random.normal(0, mutation_scale * z_range)
        
        # Enforce constraints
        mutated[:, :, 0] = np.clip(
            mutated[:, :, 0],
            self.params.diameter / 2,
            self.params.diameter / 2 + 2 * self.params.strut_thickness
        )
        
        self._control_points = mutated
        return mutated


