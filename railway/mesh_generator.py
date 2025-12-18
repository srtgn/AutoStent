"""
Stent Mesh Generation for 4C FEM Simulations

Creates hex8 finite element meshes of cylindrical stents from geometric parameters.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from pathlib import Path

# Try to import pyvista for VTU file writing
try:
    import pyvista as pv
    PV_AVAILABLE = True
except ImportError:
    PV_AVAILABLE = False
    print("WARNING: pyvista not available, cannot write VTU files")


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
                # Mapping: radial(k) -> ξ, circumferential(j) -> η, axial(i) -> ζ
                # This matches the cube mesh ordering where:
                #   n0->n1 is +ξ (+r, radially outward)
                #   n0->n3 is +η (+θ, circumferentially)  
                #   n0->n4 is +ζ (+z, axially)
                #
                # For right-handed cylindrical coords (r, θ, z), this gives positive Jacobian
                
                n0 = node_map[(i, j, k)]              # lower z, theta_j, inner r
                n1 = node_map[(i, j, k + 1)]          # lower z, theta_j, outer r
                n2 = node_map[(i, j_next, k + 1)]     # lower z, theta_j+1, outer r
                n3 = node_map[(i, j_next, k)]         # lower z, theta_j+1, inner r
                n4 = node_map[(i + 1, j, k)]          # upper z, theta_j, inner r
                n5 = node_map[(i + 1, j, k + 1)]      # upper z, theta_j, outer r
                n6 = node_map[(i + 1, j_next, k + 1)] # upper z, theta_j+1, outer r
                n7 = node_map[(i + 1, j_next, k)]     # upper z, theta_j+1, inner r
                
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
    
    # Test VTU writing
    if PV_AVAILABLE:
        vtu_path = write_vtu_file(
            nodes, elements, "test_stent_mesh.vtu",
            fixed_nodes=np.array(fixed_nodes),
            loaded_nodes=np.array(loaded_nodes)
        )
        print(f"Wrote VTU file to {vtu_path} with block_id and point_sets")


def write_vtu_file(
    nodes: np.ndarray,
    elements: np.ndarray,
    output_path: str,
    fixed_nodes: np.ndarray = None,
    loaded_nodes: np.ndarray = None
) -> Path:
    """
    Write mesh as VTU (VTK unstructured) file for 4C.
    
    4C requires:
    - Exactly one integer-typed cell-array `block_id` defining element blocks
    - Optional integer-type point-arrays 'point_set_#' for node sets
    
    Args:
        nodes: (N, 3) array of node coordinates
        elements: (M, 8) array of hex8 element connectivity (0-indexed)
        output_path: Output VTU file path
        fixed_nodes: Optional array of fixed node indices (0-indexed)
        loaded_nodes: Optional array of loaded node indices (0-indexed)
        
    Returns:
        Path to written VTU file
    """
    if not PV_AVAILABLE:
        raise ImportError("pyvista not available, cannot write VTU files")
    
    # Create unstructured grid
    # PyVista expects 0-indexed connectivity
    # Cell type: VTK_HEXAHEDRON = 12
    cell_array = []
    for elem in elements:
        cell_array.append(8)  # Number of points in hex8
        cell_array.extend(elem.tolist())
    
    # Create grid
    grid = pv.UnstructuredGrid(cell_array, [12] * len(elements), nodes)
    
    # Add required cell array: block_id (all elements in block 1)
    block_ids = np.ones(len(elements), dtype=np.int32)
    grid.cell_data['block_id'] = block_ids
    
    # Add point set arrays for boundary conditions
    if fixed_nodes is not None and len(fixed_nodes) > 0:
        # point_set_1: fixed nodes (value=1 if in set, 0 otherwise)
        point_set_1 = np.zeros(len(nodes), dtype=np.int32)
        point_set_1[fixed_nodes] = 1
        grid.point_data['point_set_1'] = point_set_1
    
    if loaded_nodes is not None and len(loaded_nodes) > 0:
        # point_set_2: loaded nodes (value=1 if in set, 0 otherwise)
        point_set_2 = np.zeros(len(nodes), dtype=np.int32)
        point_set_2[loaded_nodes] = 1
        grid.point_data['point_set_2'] = point_set_2
    
    # Write VTU file
    output_path_obj = Path(output_path)
    grid.save(str(output_path_obj))
    
    return output_path_obj
