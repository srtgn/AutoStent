"""
Stent Mesh Generation for 4C FEM Simulations

Creates hex8 finite element meshes of cylindrical stents from geometric parameters.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class StentGeometry:
    """Stent geometric parameters."""
    diameter: float  # mm
    length: float  # mm
    strut_thickness: float  # mm
    num_struts: int
    crown_height: float  # mm


def generate_cylindrical_stent_mesh(
    geometry: StentGeometry,
    n_circumferential_per_strut: int = 4,
    n_radial: int = 2,
    aspect_ratio: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Generate hex8 mesh for a simplified cylindrical stent.
    
    Args:
        geometry: Stent geometric parameters
        n_circumferential_per_strut: Elements per strut in circumferential direction
        n_radial: Elements through thickness
        aspect_ratio: Target aspect ratio for mesh elements
        
    Returns:
        nodes: (N, 3) array of node coordinates [x, y, z]
        elements: (M, 8) array of hex8 element connectivity (0-indexed)
        fixed_nodes: List of node IDs to fix (boundary condition)
        loaded_nodes: List of node IDs to apply pressure
    """
    
    radius = geometry.diameter / 2.0
    length = geometry.length
    thickness = geometry.strut_thickness
    
    # Mesh discretization
    n_circumferential = geometry.num_struts * n_circumferential_per_strut
    element_length_target = thickness * aspect_ratio
    n_axial = max(2, int(np.ceil(length / element_length_target)))
    
    print(f"Mesh: {n_circumferential} x {n_axial} x {n_radial} = {n_circumferential * n_axial * n_radial} elements")
    
    # Generate nodes
    nodes = []
    node_map = {}  # (i, j, k) -> node_id
    
    for i in range(n_axial + 1):
        z = i * length / n_axial
        for j in range(n_circumferential):
            theta = 2.0 * np.pi * j / n_circumferential
            for k in range(n_radial + 1):
                r = radius + k * thickness / n_radial
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                node_id = len(nodes)
                nodes.append([x, y, z])
                node_map[(i, j, k)] = node_id
    
    nodes = np.array(nodes)
    
    # Generate hex8 elements
    elements = []
    for i in range(n_axial):
        for j in range(n_circumferential):
            j_next = (j + 1) % n_circumferential
            for k in range(n_radial):
                # Hex8 node ordering (VTK convention)
                n0 = node_map[(i, j, k)]
                n1 = node_map[(i, j_next, k)]
                n2 = node_map[(i, j_next, k + 1)]
                n3 = node_map[(i, j, k + 1)]
                n4 = node_map[(i + 1, j, k)]
                n5 = node_map[(i + 1, j_next, k)]
                n6 = node_map[(i + 1, j_next, k + 1)]
                n7 = node_map[(i + 1, j, k + 1)]
                elements.append([n0, n1, n2, n3, n4, n5, n6, n7])
    
    elements = np.array(elements, dtype=int)
    
    # Boundary conditions
    # Fix one end (z=0)
    fixed_nodes = [node_map[(0, j, k)] 
                   for j in range(n_circumferential) 
                   for k in range(n_radial + 1)]
    
    # Apply pressure on outer surface (max radius)
    loaded_nodes = [node_map[(i, j, n_radial)] 
                    for i in range(n_axial + 1) 
                    for j in range(n_circumferential)]
    
    return nodes, elements, fixed_nodes, loaded_nodes


def write_4c_geometry(
    nodes: np.ndarray,
    elements: np.ndarray,
    fixed_nodes: List[int],
    loaded_nodes: List[int],
    output_path: str
):
    """
    Write mesh geometry in 4C YAML format.
    
    Args:
        nodes: Node coordinates
        elements: Element connectivity
        fixed_nodes: Nodes to fix
        loaded_nodes: Nodes to load
        output_path: Output file path
    """
    
    with open(output_path, 'w') as f:
        f.write("# 4C Stent Geometry\n")
        f.write("GEOMETRY:\n")
        f.write("  NODES:\n")
        for i, node in enumerate(nodes):
            f.write(f"    - ID: {i+1}\n")
            f.write(f"      COORDS: [{node[0]:.6f}, {node[1]:.6f}, {node[2]:.6f}]\n")
        
        f.write("\n  ELEMENTS:\n")
        for i, elem in enumerate(elements):
            # 4C uses 1-indexed nodes
            node_ids = ", ".join(str(n+1) for n in elem)
            f.write(f"    - ID: {i+1}\n")
            f.write(f"      TYPE: hex8\n")
            f.write(f"      NODES: [{node_ids}]\n")
        
        f.write("\n  NODE_SETS:\n")
        f.write("    - ID: 1\n")
        f.write("      NAME: fixed_end\n")
        fixed_ids = ", ".join(str(n+1) for n in fixed_nodes)
        f.write(f"      NODES: [{fixed_ids}]\n")
        
        f.write("    - ID: 2\n")
        f.write("      NAME: loaded_surface\n")
        loaded_ids = ", ".join(str(n+1) for n in loaded_nodes)
        f.write(f"      NODES: [{loaded_ids}]\n")


if __name__ == "__main__":
    # Test mesh generation
    geometry = StentGeometry(
        diameter=10.0,
        length=20.0,
        strut_thickness=0.12,
        num_struts=12,
        crown_height=1.0
    )
    
    nodes, elements, fixed, loaded = generate_cylindrical_stent_mesh(geometry)
    
    print(f"Generated mesh:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Elements: {len(elements)}")
    print(f"  Fixed nodes: {len(fixed)}")
    print(f"  Loaded nodes: {len(loaded)}")
    
    write_4c_geometry(nodes, elements, fixed, loaded, "test_stent_mesh.yaml")
    print("Wrote test mesh to test_stent_mesh.yaml")
