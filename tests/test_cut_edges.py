# tests/test_cut_edges.py

import pytest
from collections import defaultdict

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
    # 1) Build radius-2 hyperbolic patch
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

    # 2) Create the TN at bond_dim=2
    tn = build_network_from_building(building, bond_dim=2)

    # 3) Pick two boundary faces at radius=2
    radius2_words = [w for w in building.word_map if w.count(";") == 1]
    fA = building.word_map[radius2_words[0]]
    fB = building.word_map[radius2_words[-1]]

    # 4) Collect dangling edges on those faces
    region_inds = []
    for fid in (fA, fB):
        tensor = tn.select(str(fid)).tensors[0]
        dangling = list(set(tensor.inds) & set(tn.outer_inds()))
        region_inds.extend(dangling)

    # 5) Find boundary endpoints and the geodesic
    start, end = _find_boundary_endpoints(tn, building, region_inds)
    geodesic = building.find_geodesic_a_star(start, end)

    return building, tn, region_inds, start, end, geodesic


# In test_cut_edges.py

def test_cut_inds_match_partition_frontier(radius2_setup):
    """
    Tests that the `cut_inds` returned by `_partition_network` correctly
    identify the set of internal indices that cross the partition boundary.
    """
    building, tn, region_inds, start, end, geodesic = radius2_setup

    # Partition the network into a 'ket' and an 'environment'
    boundary_faces = {start, end}
    # This returns ALL indices on the ket's boundary (physical + environmental)
    ket_tags, ket_boundary_inds = _partition_network(
        tn=tn,
        building=building,
        boundary_region_faces=boundary_faces,
        geodesic_path_faces=geodesic,
    )

    # Get the set of tensor IDs that belong to the ket
    ket_tids = tn._get_tids_from_tags(ket_tags, which='any')

    # --- FIX: Manually build the expected *environmental* cut from scratch ---
    # An index is part of the environmental cut if it's an internal bond that
    # connects a tensor inside the ket to a tensor outside the ket.
    expected_environmental_cut = set()
    for ind in tn.inner_inds():
        tids_on_ind = list(tn.ind_map[ind])
        tid1, tid2 = tids_on_ind[0], tids_on_ind[1]

        # Use XOR (^) to check if one is in the ket and the other is not
        if (tid1 in ket_tids) ^ (tid2 in ket_tids):
            expected_environmental_cut.add(ind)

    # --- FIX: Filter the function's result to get just the environmental part ---
    # The true environmental cut is the set of the ket's boundary indices
    # that are NOT physical legs of the overall network.
    environmental_cut_from_func = {
        ind for ind in ket_boundary_inds if ind in tn.inner_inds()
    }

    # 3) The environmental parts must match exactly
    assert environmental_cut_from_func == expected_environmental_cut, (
        f"Computed environmental cut ({len(environmental_cut_from_func)}) "
        f"!= expected internal frontier ({len(expected_environmental_cut)})"
    )