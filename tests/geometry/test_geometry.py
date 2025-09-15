# tests/test_geometry.py

import pytest
import numpy as np
from src.holographic_tn.geometry import HyperbolicBuilding, PoincareIsometry

# --- FIX: Added the missing pytest fixture ---
@pytest.fixture(scope="module")
def simple_building():
    """Pytest fixture to build a standard 2x2 grid building for testing."""
    building = HyperbolicBuilding(p=4, v=4)
    building.construct_2x2_grid(initial_coords=np.array([0.0, 0.0]))
    return building


def test_generative_exploration():
    """
    Tests that the generative algorithm explores the building to a
    specified radius.
    """
    building = HyperbolicBuilding(p=4, v=4)

    # --- FIX: Initialize isometries with non-trivial geometric values ---
    dist = 0.4  # A non-zero distance for translations
    g_x = PoincareIsometry("X+", z0=(dist, 0))
    g_y = PoincareIsometry("Y+", z0=(0, dist))
    # Create true inverses using the .get_inverse() method
    g_x_inv = g_x.get_inverse("X-")
    g_y_inv = g_y.get_inverse("Y-")

    # Pairings now define a generator and its inverse for each pairing
    # Format: (edge1, edge2, generator, inverse_generator)
    side_pairings = [
        (0, 2, g_y_inv, g_y),  # bottom/top
        (1, 3, g_x, g_x_inv),  # right/left
        (2, 0, g_y, g_y_inv),  # top/bottom
        (3, 1, g_x_inv, g_x),  # left/right
    ]

    # Explore to a radius of 1 tile from the origin
    building.construct_building(side_pairings, max_radius=1)

    faces = [n for n, attr in building.simplicial_complex.nodes(data=True)
             if attr.get('type') == 'face']
    adj_edges = [e for e in building.simplicial_complex.edges(data=True)
                 if e[2].get('type') == 'adjacency']

    # A radius-1 build should create the central face + 4 neighbors = 5 faces.
    assert len(faces) == 5
    # The 4 neighbors are connected to the central face, creating 4 edges.
    assert len(adj_edges) == 4


def test_building_construction(simple_building):
    """Tests that the building is constructed with the correct topology."""
    # A 2x2 grid should have 4 faces
    faces = [n for n, attr in simple_building.simplicial_complex.nodes(data=True) if attr.get('type') == 'face']
    assert len(faces) == 4

    # It should have 4 internal adjacency edges
    adj_edges = [e for e in simple_building.simplicial_complex.edges(data=True) if e[2].get('type') == 'adjacency']
    assert len(adj_edges) == 4

    # Adjacency info should be correctly populated
    assert len(simple_building.adjacency_info) == 8  # 4 pairs, two-way map


def test_find_geodesic_a_star(simple_building):
    """Tests the A* algorithm for finding shortest paths."""
    # The path between a node and itself is just that node
    path_to_self = simple_building.find_geodesic_a_star((2, 0), (2, 0))
    assert path_to_self == [(2, 0)]

    # Test a known path, e.g., a diagonal path in the 2x2 grid
    # Let's assume face 3 is diagonally opposite to face 0
    start_face = (2, 0)
    end_face = (2, 3)
    path = simple_building.find_geodesic_a_star(start_face, end_face)

    # The path should exist and have a length of 3 (start -> middle -> end)
    assert path is not None
    assert len(path) == 3
    assert path[0] == start_face
    assert path[-1] == end_face


def test_identify_gromov_boundary(simple_building):
    """Tests the structure of the Gromov boundary identification output."""
    origin = (2, 0)
    boundary_points = simple_building.identify_gromov_boundary(origin, num_rays=4)

    # The function should return a list of clusters
    assert isinstance(boundary_points, list)
    # For a 2x2 grid, we expect rays to go in a few distinct directions
    # The exact number of clusters depends on the random targets, but it should be > 0
    assert len(boundary_points) > 0
    # Each cluster should be a list of ray paths
    assert isinstance(boundary_points[0], list)
    assert isinstance(boundary_points[0][0], list)