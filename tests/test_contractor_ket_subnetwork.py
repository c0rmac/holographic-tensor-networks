# tests/test_contractor_ket_subnetwork.py

import pytest
# - FIX -# Import quimb and numpy
import quimb.tensor as qtn
import numpy as np

from src.holographic_tn.geometry import HyperbolicBuilding, PoincareIsometry
# - FIX -# Import the correct, refactored functions from physics.py
from src.holographic_tn.physics import (
    build_network_from_building,
    _find_boundary_endpoints,
    _partition_network,
)


# - FIX -# This fixture is rewritten to use the corrected quimb API
@pytest.fixture
def radius2_setup():
    """Builds a radius-2 patch, TN, and identifies a boundary region."""
    # Build a small hyperbolic patch at radius=2
    building = HyperbolicBuilding(p=4, v=4)
    # Use non-trivial isometries for a valid geometry
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

    # Build the tensor network at bond_dim=2
    tn = build_network_from_building(building, bond_dim=2)

    # Pick two boundary faces at radius=2 and collect dangling edges
    radius2_words = [w for w in building.word_map if w.count(";") == 1]
    face_A = building.word_map[radius2_words[0]]
    face_B = building.word_map[radius2_words[-1]]

    region_inds = []
    for fid in (face_A, face_B):
        tensor = tn.select(str(fid)).tensors[0]
        dangling = list(set(tensor.inds) & set(tn.outer_inds()))
        region_inds.extend(dangling)

    # Determine the start/end faces and the bulk geodesic
    start, end = _find_boundary_endpoints(tn, building, region_inds)
    geodesic = building.find_geodesic_a_star(start, end)

    return building, tn, region_inds, start, end, geodesic


# - FIX -# This test is rewritten to use the corrected quimb API
def test_ket_subnetwork_contraction_preserves_boundary(radius2_setup):
    """
    Tests that contracting the 'ket' subnetwork produces a single tensor
    with the correct boundary indices and shape.
    """
    building, tn, region_inds, start, end, geodesic = radius2_setup

    # Partition the network to get the tags for the 'ket'
    boundary_faces = {start, end}
    ket_tags, cut_inds = _partition_network(
        tn=tn,
        building=building,
        boundary_region_faces=boundary_faces,
        geodesic_path_faces=geodesic,
    )

    # Create the 'ket' sub-network
    ket_tn = tn.select(ket_tags, which='any')

    # The expected output indices are all the outer indices of this sub-network
    output_inds_expected = ket_tn.outer_inds()

    # Perform the contraction of the ket, keeping its boundary open
    contracted_ket_tensor = ket_tn.contract(all, output_inds=output_inds_expected)

    # 1) Assert that the final tensor has exactly the indices we requested
    assert set(contracted_ket_tensor.inds) == set(output_inds_expected)

    # 2) Assert that the resulting tensor's shape matches the dimensions of the indices
    expected_shape = tuple(tn.ind_size(ix) for ix in output_inds_expected)
    assert contracted_ket_tensor.shape == expected_shape