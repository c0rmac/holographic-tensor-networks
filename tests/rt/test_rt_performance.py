import time
import pytest
import numpy as np
import tracemalloc

#- FIX -# Add quimb import
import quimb.tensor as qtn

from src.holographic_tn.geometry import HyperbolicBuilding, PoincareIsometry
from src.holographic_tn.numerics.kpm.config import KPMConfig
#- FIX -# Import the correct function from physics.py
from src.holographic_tn.physics   import build_network_from_building, calculate_rt_entropy

# -- Helper to build a hyperbolic patch and TN once per (radius, bond_dim) --
def setup_network(radius: int, bond_dim: int):
    """Constructs hyperbolic building and holographic TN for given radius."""
    building = HyperbolicBuilding(p=4, v=4)
    # Define X/Y isometries
    g_x, g_x_inv = PoincareIsometry("X+"), PoincareIsometry("X-")
    g_y, g_y_inv = PoincareIsometry("Y+"), PoincareIsometry("Y-")
    side_pairings = [
        (0, 2, g_y_inv, g_y),
        (1, 3, g_x,     g_x_inv),
        (2, 0, g_y,     g_y_inv),
        (3, 1, g_x_inv, g_x),
    ]
    building.construct_building(side_pairings, max_radius=radius)

    #- FIX -# Call the renamed function and use 'tn' as the variable name
    tn = build_network_from_building(building, bond_dim=bond_dim)

    # Pick two distinct boundary faces at max radius
    words = [w for w in building.word_map if w.count(";") == radius - 1]
    face_A = building.word_map[words[0]]
    face_B = building.word_map[words[-1]]

    # Collect all dangling edges from both faces
    region_inds = []
    for f in (face_A, face_B):
        #- FIX -# Use correct quimb methods to select tensor and get dangling indices
        tensor = tn.select(str(f)).tensors[0]
        dangling = list(set(tensor.inds) & set(tn.outer_inds()))
        region_inds.extend(dangling)

    #- FIX -# Return corrected variables
    return building, tn, region_inds


@pytest.mark.slow
@pytest.mark.parametrize("radius,bond_dim", [
    (2, 2),
    (3, 2),
    # add more combos if desired
])
def test_rt_entropy_performance(radius, bond_dim):
    """
    Measures runtime and peak memory of calculate_rt_entropy for
    various radius/bond_dim settings.
    """
    #- FIX -# Unpack corrected variable names
    building, tn, region_inds = setup_network(radius, bond_dim)

    #- FIX -# Replace incorrect edge counting with direct quimb properties
    dangling_inds = tn.outer_inds()
    internal_inds = tn.inner_inds()

    print("dangling legs:", len(dangling_inds))
    print("internal legs:", len(internal_inds))
    print("region size:", len(region_inds))

    num_tensors = tn.num_tensors

    # Start wall-clock timer and memory tracker
    tracemalloc.start()
    t0 = time.perf_counter()

    #- FIX -# Pass correct variables and specify the memory-efficient KPM method
    results = calculate_rt_entropy(building, tn, region_inds, config=KPMConfig())

    duration = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Basic sanity checks
    assert results["entropy"] >= 0.0
    assert results["discrete_geodesic_length"] > 0.0

    # cut_length must be positive
    assert results["cut_length"] > 0, "RT cut should sever at least one edge"

    # entropy bound still holds
    max_ent = results["cut_length"] * np.log2(bond_dim)
    assert results["entropy"] <= max_ent + 1e-1 # Looser tolerance for KPM

    # Report to the console for manual inspection / CI logs
    #- FIX -# Update variable names in print statement
    print(
        f"\n[Performance] radius={radius}, bond_dim={bond_dim}, "
        f"tensors={num_tensors}, time={duration:.2f}s, peak_mem={peak/1e6:.1f}MB"
    )

    # Optional: enforce thresholds (uncomment & tune)
    # assert duration < 30.0, "Runtime exceeded 30s"
    # assert peak < 2e9,      "Peak memory exceeded 2 GB"