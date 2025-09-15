import numpy as np
import pytest
import random

from src.holographic_tn.geometry import HyperbolicBuilding, PoincareIsometry
from src.holographic_tn.tensor    import create_holographic_tn
from src.holographic_tn.physics   import calculate_rt_entropy

def build_radius2_patch():
    b = HyperbolicBuilding(p=4, v=4)
    g_x, g_x_inv = PoincareIsometry("X+"), PoincareIsometry("X-")
    g_y, g_y_inv = PoincareIsometry("Y+"), PoincareIsometry("Y-")
    side_pairings = [
        (0, 2, g_y_inv, g_y),
        (1, 3, g_x,     g_x_inv),
        (2, 0, g_y,     g_y_inv),
        (3, 1, g_x_inv, g_x),
    ]
    b.construct_building(side_pairings, max_radius=2)
    return b

def pick_radius2_region(building, all_nodes):
    """
    Given a HyperbolicBuilding and its holographic TN (all_nodes),
    return the list of dangling edges on two radius‐2 faces.
    """
    # 1) Find all the radius‐2 words in the building’s word_map
    radius2_words = [w for w in building.word_map if w.count(";") == 1]
    if len(radius2_words) < 2:
        raise RuntimeError("Expected at least two radius-2 faces")

    # 2) Pick the first and last of those faces
    fA = building.word_map[radius2_words[0]]
    fB = building.word_map[radius2_words[-1]]

    # 3) Gather dangling edges from each face’s node in the TN
    region = []
    for face_id in (fA, fB):
        node = all_nodes[face_id]
        region.extend(e for e in node.edges if e.is_dangling())

    return region

@pytest.mark.slow
def test_ratio_large_d_limit_radius2():
    """
    Ensemble‐smoothed |S/L - 1| at each D should decrease with D.
    We sample 5 independent RNG seeds per bond‐dim.
    """
    import tensornetwork as tn
    # switch to TF backend
    tn.set_default_backend("tensorflow")

    building = build_radius2_patch()
    Ds       = [2, 4, 5]
    seed_list= [0, 13, 42, 99, 123]  # 5 seeds
    mean_devs = []

    for D in Ds:
        devs = []
        for seed in seed_list:
            np.random.seed(seed)
            random.seed(seed)

            all_nodes = create_holographic_tn(building, bond_dim=D)
            region    = pick_radius2_region(building, all_nodes)
            res       = calculate_rt_entropy(building, all_nodes, region)

            S, L = res["entropy"], res["discrete_geodesic_length"]
            assert L > 0
            devs.append(abs(S/L - 1.0))

        mean_dev = float(np.mean(devs))
        std_dev  = float(np.std(devs))
        mean_devs.append(mean_dev)
        print(f"D={D}: ⟨|S/L-1|⟩={mean_dev:.3f} ±{std_dev:.3f}")

    # Now assert the smoothed deviations decrease (D=2→8)
    assert mean_devs[2] < mean_devs[0], (
        f"Mean dev at D=8 ({mean_devs[2]:.3f}) !< at D=2 ({mean_devs[0]:.3f})"
    )

    # Optionally also check D=4 < D=2
    assert mean_devs[1] < mean_devs[0], (
        f"Mean dev at D=4 ({mean_devs[1]:.3f}) !< at D=2 ({mean_devs[0]:.3f})"
    )

    # And require the final mean deviation is below some threshold
    assert mean_devs[2] < 0.2, (
        f"Smoothed |S/L-1| at D=8 is too large: {mean_devs[2]:.3f}"
    )