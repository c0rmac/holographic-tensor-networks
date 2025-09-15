# tests/test_tensor.py

import pytest
import quimb.tensor as qtn
import numpy as np

# - FIX -# Add import for HyperbolicBuilding
from src.holographic_tn.geometry import HyperbolicBuilding
from src.holographic_tn.physics import build_network_from_building
from src.holographic_tn.tensor import get_perfect_tensor_of_rank


# - FIX -# The import for the fixture is no longer needed
# from tests.conftest import simple_building


def test_get_perfect_tensor_of_rank():
    """Tests the placeholder tensor generation function."""
    rank = 6
    bond_dim = 2
    tensor = get_perfect_tensor_of_rank(rank, bond_dim)

    assert tensor.ndim == rank
    assert tensor.shape == (bond_dim,) * rank


def test_build_network_from_building():
    """Tests the creation of a tensor network from a building."""
    # - FIX -# Remove the fixture and create the building object inside the test
    building = HyperbolicBuilding(p=4, v=4)
    building.construct_2x2_grid(initial_coords=np.array([0.0, 0.0]))

    tn = build_network_from_building(building, bond_dim=2)

    # Number of tensor nodes should match the number of faces
    assert tn.num_tensors == 4

    # In a 2x2 grid of 4 squares, there are 4 internal adjacencies.
    assert len(tn.inner_inds()) == 4

    # There are 8 dangling edges on the perimeter (2 per face, on the outer sides).
    assert len(tn.outer_inds()) == 8