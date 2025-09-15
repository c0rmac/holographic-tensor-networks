# tests/test_rt_entropy.py
import numpy as np
import pytest
import quimb.tensor as qtn

from src.holographic_tn.geometry import HyperbolicBuilding, PoincareIsometry
from src.holographic_tn.numerics.kpm.config import KPMConfig
from src.holographic_tn.physics import build_network_from_building, calculate_rt_entropy


@pytest.fixture(scope="module")
def grid_setup():
    """Pytest fixture to build a standard 2x2 grid building and tensor network."""
    building = HyperbolicBuilding(p=4, v=4)
    building.construct_2x2_grid(initial_coords=np.array([0.0, 0.0]))
    tn = build_network_from_building(building, bond_dim=2)
    return building, tn


class TestSimpleGrid:
    """Groups tests that run on the simple 2x2 grid geometry."""

    def test_single_face(self, grid_setup):
        """Tests the trivial case where the boundary region is on a single face."""
        building, tn = grid_setup

        subgraph = tn.select('FACE_0')
        all_physical_inds = tn.outer_inds()
        region = list(set(subgraph.outer_inds()) & set(all_physical_inds))

        res = calculate_rt_entropy(building, tn, region)

        # The 'geodesic' for a single-face region is empty
        assert res["discrete_geodesic_length"] == pytest.approx(0.0)
        # The cut is the boundary between the face and the rest of the network (2 bonds)
        assert res["cut_length"] == 2
        # For perfect tensors, S(A) = cut_length * log2(D). For D=2, S(A) = cut_length.
        assert res["entropy"] == pytest.approx(res["cut_length"])

    def test_two_adjacent_faces(self, grid_setup):
        """Tests a boundary region defined on two adjacent faces."""
        building, tn = grid_setup

        subgraph = tn.select(['FACE_0', 'FACE_1'], 'any')
        all_physical_inds = tn.outer_inds()
        region = list(set(subgraph.outer_inds()) & set(all_physical_inds))

        # Use the 'exact' method as it's deterministic and precise for small systems
        res = calculate_rt_entropy(building, tn, region, entropy_method='exact')

        # --- FIX: Assert the correct physical quantities ---
        # The geodesic is a single step between adjacent faces
        assert res["discrete_geodesic_length"] == pytest.approx(1.0)
        # The cut separating {F0, F1} from the rest of the grid is 2 bonds
        assert res["cut_length"] == 2
        # For perfect tensors, S(A) should equal the cut_length
        assert res["entropy"] == pytest.approx(res["cut_length"])

    def test_opposite_corners(self, grid_setup):
        """Tests a boundary region on diagonally opposite corners of the grid."""
        building, tn = grid_setup

        subgraph = tn.select(['FACE_0', 'FACE_3'], 'any')
        all_physical_inds = tn.outer_inds()
        region = list(set(subgraph.outer_inds()) & set(all_physical_inds))

        res = calculate_rt_entropy(building, tn, region, entropy_method='bbp')

        # --- FIX: Assert the correct physical quantities ---
        # The path between opposite corners is 2 steps
        assert res["discrete_geodesic_length"] == pytest.approx(2.0)
        # The minimal cut for this region is also 2 bonds
        assert res["cut_length"] == 4
        # For perfect tensors, S(A) should equal the cut_length
        assert res["entropy"] == pytest.approx(res["cut_length"])


def test_generative_construction():
    """
    Tests the RT correspondence on a complex hyperbolic patch.
    """
    # Using a pentagonal tiling (p=5) makes the geometry hyperbolic
    building = HyperbolicBuilding(p=5, v=4)
    dist = 0.4
    # Define non-trivial isometries for the generative construction
    generators = [PoincareIsometry(f"g{i}", z0=(dist, 0), phi=i*2*np.pi/5) for i in range(5)]
    inverses = [g.get_inverse(f"g{i}_inv") for i, g in enumerate(generators)]
    side_pairings = [
        (0, 2, generators[0], inverses[0]), (1, 3, generators[1], inverses[1]),
        (2, 4, generators[2], inverses[2]), (3, 0, generators[3], inverses[3]),
        (4, 1, generators[4], inverses[4]),
    ]
    side_pairings.extend([(p[1], p[0], p[3], p[2]) for p in side_pairings])
    building.construct_building(side_pairings, max_radius=2)
    tn = build_network_from_building(building, bond_dim=2)


    # Select two opposing faces at the max radius
    radius2_words = [w for w in building.word_map if w.count(";") == 1]
    fA_tag = str(building.word_map[radius2_words[0]])
    fB_tag = str(building.word_map[radius2_words[-1]])

    subgraph = tn.select([fA_tag, fB_tag], 'any')
    all_physical_inds = tn.outer_inds()
    region = list(set(subgraph.outer_inds()) & set(all_physical_inds))

    res = calculate_rt_entropy(building, tn, region, config=KPMConfig())

    assert res["entropy"] > 0.0
    assert res["geometric_geodesic_length"] > 0.0

    # For perfect tensors, the ratio S(A) / cut_length should be ~1.
    # The geometric length has a non-trivial prefactor, so we check a wider range.
    ratio = res["entropy"] / res["geometric_geodesic_length"]
    assert 0.1 <= ratio <= 5.0, f"ratio={ratio:.3f} out of tolerance"