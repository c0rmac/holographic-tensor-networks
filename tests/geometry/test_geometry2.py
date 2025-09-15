# tests/test_geometry.py

import pytest
from collections import deque
import numpy as np

# - FIX -# Add necessary imports
from src.holographic_tn.geometry import HyperbolicBuilding, PoincareIsometry
from src.holographic_tn.physics import _partition_network, build_network_from_building


# - FIX -# Fixture is updated to create a non-trivial geometry and also return the tensor network
@pytest.fixture
def radius2_setup():
    """
    Builds a non-trivial radius-2 patch and the corresponding tensor network.
    """
    building = HyperbolicBuilding(p=4, v=4)
    # Use non-trivial isometries to generate a real geometric patch
    dist = 0.4
    gx = PoincareIsometry("X+", z0=(dist, 0))
    gy = PoincareIsometry("Y+", z0=(0, dist))
    gx_inv = gx.get_inverse("X-")
    gy_inv = gy.get_inverse("Y-")

    side_pairings = [
        (0, 2, gy_inv, gy),
        (1, 3, gx, gx_inv),
        (2, 0, gy, gy_inv),
        (3, 1, gx_inv, gx),
    ]
    building.construct_building(side_pairings, max_radius=2)

    # Also build the tensor network, as it's needed for the partition test
    tn = build_network_from_building(building, bond_dim=2)

    return building, tn


def test_face_neighbors_are_symmetric(radius2_setup):
    b, _ = radius2_setup
    # for every face, each neighbor must list it back
    for face_id in b.word_map.values():
        neigh = b._get_face_neighbors(face_id)
        for n in neigh:
            assert face_id in b._get_face_neighbors(n), (
                f"{face_id} ↔ {n} connectivity is not mutual"
            )


# - FIX -# This test has been completely rewritten for the new _partition_network API
def test_bfs_partition_defines_ket_correctly(radius2_setup):
    """
    Tests that _partition_network correctly identifies the 'ket' region,
    which should include the boundary and the geodesic wall.
    """
    building, tn = radius2_setup

    # 1) Pick two boundary faces at radius=2
    radius2_words = [w for w in building.word_map if w.count(";") == 1]
    start_word, end_word = radius2_words[0], radius2_words[-1]
    start_id = building.word_map[start_word]
    end_id = building.word_map[end_word]

    # 2) Find the real geodesic between them to use as the wall
    geodesic_path = building.find_geodesic_a_star(start_id, end_id)
    assert geodesic_path is not None, "A valid geodesic path should exist."

    # 3) Define the boundary region for the function call
    boundary_faces = {start_id, end_id}

    # 4) Run the partition function with the correct arguments
    ket_tags, cut_inds = _partition_network(
        tn=tn,
        building=building,
        boundary_region_faces=boundary_faces,
        geodesic_path_faces=geodesic_path,
    )

    # --- Assertions based on the new logic ---
    # The ket is defined as the union of the interior BFS-discovered region
    # AND the geodesic wall itself.

    # A) Assert that the tags for the geodesic faces are in the final ket_tags
    for face_id in geodesic_path:
        assert str(face_id) in ket_tags, \
            f"Geodesic face {face_id} is missing from the ket."

    # B) Assert that the original boundary faces are also in the ket
    for face_id in boundary_faces:
        assert str(face_id) in ket_tags, \
            f"Boundary face {face_id} is missing from the ket."

    # C) The number of cut indices should be non-zero for a non-trivial partition
    assert len(cut_inds) > 0, "The partition should result in a non-empty cut."