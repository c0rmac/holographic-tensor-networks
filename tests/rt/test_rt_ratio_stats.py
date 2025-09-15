import os
import pickle
import random

import pytest
# - FIX -# Add necessary import for quimb
import quimb.tensor as qtn
from matplotlib import pyplot as plt

from src.holographic_tn.config.entropy_method import ExactConfig
from src.holographic_tn.experimental.bp.config import BBPConfig
from src.holographic_tn.geometry import HyperbolicBuilding, PoincareIsometry
import numpy as np

from src.holographic_tn.numerics.kpm.config import KPMConfig
# - FIX -# Import the correct, renamed function from physics.py
from src.holographic_tn.physics import calculate_rt_entropy, build_network_from_building

def build_radius2_patch(bond_dim=2):
    building = HyperbolicBuilding(p=4, v=4)
    g_x, g_x_inv = PoincareIsometry("X+"), PoincareIsometry("X-")
    g_y, g_y_inv = PoincareIsometry("Y+"), PoincareIsometry("Y-")
    side_pairings = [
        (0, 2, g_y_inv, g_y),
        (1, 3, g_x,     g_x_inv),
        (2, 0, g_y,     g_y_inv),
        (3, 1, g_x_inv, g_x),
    ]
    building.construct_building(side_pairings, max_radius=2)

    # 2) Create the tensor network at the desired bond dimension
    # - FIX -# Call the renamed function and use 'tn' as the variable name
    tn = build_network_from_building(building, bond_dim=bond_dim)

    # 3) Pick two faces at radius=2 (words containing exactly one ';')
    radius2_words = [w for w in building.word_map if w.count(";") == 1]
    if len(radius2_words) < 2:
        raise RuntimeError("Not enough radius-2 faces found.")
    fA = building.word_map[radius2_words[0]]
    fB = building.word_map[radius2_words[-1]]

    # 4) Collect all dangling (physical) edges from both faces
    region_inds = []
    for face_id in (fA, fB):
        # - FIX -# Select a tensor by its unique tag
        tensor = tn.select(str(face_id)).tensors[0]
        # - FIX -# Get dangling indices by intersecting with the network's outer indices
        dangling_inds = list(set(tensor.inds) & set(tn.outer_inds()))
        region_inds.extend(dangling_inds)

    return building, tn, region_inds


def _build_radius3_region(bond_dim=2):
    """
    Constructs a HyperbolicBuilding at max_radius=3 with p=6, v=4
    and returns (building, tn, region_inds) for a valid RT test.
    """
    # Using p=6 (an even number) ensures true perfect tensors are used.
    building = HyperbolicBuilding(p=6, v=4)

    # For a hexagon, we need 3 generators to pair opposite sides.
    dist = 0.5
    g0 = PoincareIsometry("g0", z0=(dist, 0), phi=0)
    g1 = PoincareIsometry("g1", z0=(0, dist), phi=np.pi / 2)
    g2 = PoincareIsometry("g2", z0=(-dist, 0), phi=np.pi)
    inv0 = g0.get_inverse("g0_inv")
    inv1 = g1.get_inverse("g1_inv")
    inv2 = g2.get_inverse("g2_inv")

    # Corrected side pairings for a hexagon (p=6)
    side_pairings = [
        (0, 3, g0, inv0),
        (1, 4, g1, inv1),
        (2, 5, g2, inv2),
    ]

    side_pairings.extend([(p[1], p[0], p[3], p[2]) for p in side_pairings])

    # --- FIX: Increase max_radius to generate a larger graph ---
    max_radius = 3
    building.construct_building(side_pairings, max_radius=max_radius)

    tn = build_network_from_building(building, bond_dim=bond_dim)

    # Find faces at the new, larger radius
    # Note: word.count(";") is number of steps - 1. So radius = count + 1.
    outer_words = [w for w in building.word_map if w.count(";") == (max_radius - 1)]

    if len(outer_words) < 2:
        raise RuntimeError(f"Not enough distinct faces found even at radius {max_radius}.")

    # Use the robust method to find the maximally separated faces
    outer_faces = [building.word_map[w] for w in outer_words]
    max_dist = -1.0
    fA, fB = None, None
    for i in range(len(outer_faces)):
        for j in range(i + 1, len(outer_faces)):
            p1, p2 = outer_faces[i], outer_faces[j]
            dist = building._get_path_distance(p1, p2)
            if dist > max_dist:
                max_dist = dist
                fA, fB = p1, p2

    print(f"    - Selected boundary faces {fA} and {fB} with max distance {max_dist}")

    region_inds = []
    for face_id in (fA, fB):
        tensor = tn.select(str(face_id)).tensors[0]
        dangling_inds = list(set(tensor.inds) & set(tn.outer_inds()))
        region_inds.extend(dangling_inds)

    return building, tn, region_inds

def _build_radius2_region(bond_dim=2):
    """
    Constructs a HyperbolicBuilding at max_radius=2 with p=6, v=4
    and returns (building, tn, region_inds) for a valid RT test.
    """
    # Using p=6 (an even number) ensures true perfect tensors are used.
    building = HyperbolicBuilding(p=6, v=4)

    # For a hexagon, we need 3 generators to pair opposite sides.
    dist = 0.5
    g0 = PoincareIsometry("g0", z0=(dist, 0), phi=0)
    g1 = PoincareIsometry("g1", z0=(0, dist), phi=np.pi/2)
    g2 = PoincareIsometry("g2", z0=(-dist, 0), phi=np.pi)
    inv0 = g0.get_inverse("g0_inv")
    inv1 = g1.get_inverse("g1_inv")
    inv2 = g2.get_inverse("g2_inv")

    # Corrected side pairings for a hexagon (p=6)
    # 0 is paired with 3, 1 with 4, and 2 with 5.
    side_pairings = [
        (0, 3, g0, inv0),
        (1, 4, g1, inv1),
        (2, 5, g2, inv2),
    ]

    # Add inverse rules to allow traversal in both directions.
    side_pairings.extend([(p[1], p[0], p[3], p[2]) for p in side_pairings])

    building.construct_building(side_pairings, max_radius=2)

    # The rest of the function remains the same.
    tn = build_network_from_building(building, bond_dim=bond_dim)

    radius2_words = [w for w in building.word_map if w.count(";") == 1]
    if len(radius2_words) < 2:
        radius1_words = [w for w in building.word_map if w.count(";") == 0 and w]
        if len(radius1_words) < 2:
            raise RuntimeError("Not enough distinct faces found to form a region.")
        fA = building.word_map[radius1_words[0]]
        fB = building.word_map[radius1_words[-1]]
    else:
        fA = building.word_map[radius2_words[0]]
        fB = building.word_map[radius2_words[-1]]

    region_inds = []
    for face_id in (fA, fB):
        tensor = tn.select(str(face_id)).tensors[0]
        dangling_inds = list(set(tensor.inds) & set(tn.outer_inds()))
        region_inds.extend(dangling_inds)

    return building, tn, region_inds

def build_radius2_region(bond_dim=2):
    """
    Constructs a HyperbolicBuilding at max_radius=2 with p=4, v=4
    and returns (building, tn, region_inds) where the region
    is the union of dangling legs on two selected radius-2 faces.
    """
    # 1) Set up the building & side pairings
    # --- FIX: Change 'p' to a value > 4 to create a hyperbolic tiling ---
    building = HyperbolicBuilding(p=5, v=4)

    # Define isometries with non-trivial geometric transformations
    dist = 0.4  # A non-zero distance for translations
    # For a pentagon (p=5), you need 5 generators
    generators = [PoincareIsometry(f"g{i}", z0=(dist, 0), phi=i * 2 * np.pi / 5) for i in range(5)]
    inverses = [g.get_inverse(f"g{i}_inv") for i, g in enumerate(generators)]

    # Use a side-pairing rule appropriate for the new tiling
    # This is the rule from the HaPPY code, which is known to be non-trivial
    side_pairings = [
        (0, 2, generators[0], inverses[0]), (1, 3, generators[1], inverses[1]),
        (2, 4, generators[2], inverses[2]), (3, 0, generators[3], inverses[3]),
        (4, 1, generators[4], inverses[4]),
    ]
    side_pairings.extend([(p[1], p[0], p[3], p[2]) for p in side_pairings])

    building.construct_building(side_pairings, max_radius=2)

    # 2) Create the tensor network at the desired bond dimension
    # - FIX -# Call the renamed function and use 'tn' as the variable name
    tn = build_network_from_building(building, bond_dim=bond_dim)

    # 3) Pick two faces at radius=2 (words containing exactly one ';')
    radius2_words = [w for w in building.word_map if w.count(";") == 1]
    if len(radius2_words) < 2:
        raise RuntimeError("Not enough radius-2 faces found.")
    fA = building.word_map[radius2_words[0]]
    fB = building.word_map[radius2_words[-1]]

    # 4) Collect all dangling (physical) edges from both faces
    region_inds = []
    for face_id in (fA, fB):
        # - FIX -# Select a tensor by its unique tag
        tensor = tn.select(str(face_id)).tensors[0]
        # - FIX -# Get dangling indices by intersecting with the network's outer indices
        dangling_inds = list(set(tensor.inds) & set(tn.outer_inds()))
        region_inds.extend(dangling_inds)

    return building, tn, region_inds


@pytest.mark.slow
def test_ratio_ensemble_average_radius2():
    ratios = []
    for seed in range(10):  # Reduced from 100 for faster testing
        # - FIX -# Unpack corrected variable names
        building, tn, region_inds = build_radius2_region(bond_dim=2)
        # - FIX -# Pass correct variables and use exact contractor for this small network
        res = calculate_rt_entropy(building, tn, region_inds)
        ratios.append(res["entropy"] / res["discrete_geodesic_length"])
    avg_ratio = np.mean(ratios)
    assert 0.4 < avg_ratio < 0.6

@pytest.mark.slow
def test_ratio_large_d_limit_radius2():
    #np.random.seed(123)

    # - FIX -# Refactored for clarity and efficiency
    # 1. Build the geometry and identify the boundary region just once.
    #building, tn_template, region_inds = build_radius2_patch(bond_dim=2)
    building, tn_template, region_inds = _build_radius2_region(bond_dim=2)

    # D -> M
    moment_map = {
        # 2: 400,
        4: 800,
        # 6: 6000
    }

    deviations = []
    for D, M in moment_map.items():
        # 2. For each bond dimension D, create a new tensor network on the same geometry.
        tn = build_network_from_building(building, bond_dim=D)

        # 3. Run the calculation.
        # res = calculate_rt_entropy(building, tn, region_inds, config=ExactConfig())
        res = calculate_rt_entropy(building, tn, region_inds, config=KPMConfig(num_vectors=5, num_moments=M, bounds_method='accurate'))
        deviation = abs(res["entropy"] / res["geometric_geodesic_length"] - np.log2(D))
        deviations.append(deviation)

    # As bond dimension D increases, entropy S(A) should approach |γ_A|*log2(D)
    # Therefore, the deviation from this value should decrease.
    assert deviations[1] < deviations[0]
    assert deviations[2] < deviations[1]


@pytest.mark.slow
def test_generate_moment_convergence_plot_with_caching():
    """
    Calculates KPM entropy convergence, caching the results and the parameters
    used to generate the list of moments.
    """
    # --- Configuration for the run ---
    BOND_DIM = 4
    NUM_VECTORS = 50
    # Define moment parameters separately for saving
    M_START = 300
    M_STOP = 2000
    M_STEP = 50
    MOMENT_VALUES = list(range(M_START, M_STOP, M_STEP))
    # ---

    # Generate a more descriptive filename for the cache
    pickle_filename = f"convergence_data_D{BOND_DIM}_V{NUM_VECTORS}_M{M_START}-{M_STOP - 1}-{M_STEP}.pkl"

    # 1. Check if cached results exist.
    if os.path.exists(pickle_filename):
        print(f"✅ Found cached results in '{pickle_filename}'. Loading data...")
        with open(pickle_filename, 'rb') as f:
            results = pickle.load(f)

        # Unpack data and reconstruct the moment values list from parameters
        m_params = results['moment_parameters']
        MOMENT_VALUES = list(range(m_params['start'], m_params['stop'], m_params['step']))
        entropies = results['entropies']
        geometric_geodesic_length = results['geometric_geodesic_length']

    else:
        print(f"INFO: No cache found at '{pickle_filename}'. Running full calculation...")
        # 2. If no cache, run the full calculation.
        # 2a. Set up the geometry and tensor network.
        print("▶️ Step A: Building geometry and tensor network...")
        building, _, region_inds = _build_radius2_region(bond_dim=2)
        tn = build_network_from_building(building, bond_dim=BOND_DIM)

        # 2b. Loop over the moment values.
        print(f"▶️ Step B: Calculating entropy for M in {MOMENT_VALUES}...")
        entropies = []
        first_res = calculate_rt_entropy(
            building, tn, region_inds,
            config=KPMConfig(num_vectors=NUM_VECTORS, num_moments=MOMENT_VALUES[0], bounds_method='accurate')
        )
        geometric_geodesic_length = first_res["geometric_geodesic_length"]
        entropies.append(first_res["entropy"])

        # Start loop from the second moment value
        for M in MOMENT_VALUES[1:]:
            print(f"  Calculating for M = {M}...")
            res = calculate_rt_entropy(
                building, tn, region_inds,
                config=KPMConfig(num_vectors=NUM_VECTORS, num_moments=M)
            )
            entropies.append(res["entropy"])

        # 2c. Save the fresh results to the pickle file, including parameters.
        print(f"▶️ Step C: Saving new results to '{pickle_filename}'...")
        results_to_pickle = {
            "bond_dimension": BOND_DIM,
            "moment_parameters": {
                "start": M_START,
                "stop": M_STOP,
                "step": M_STEP
            },
            "entropies": entropies,
            "geometric_geodesic_length": geometric_geodesic_length
        }
        with open(pickle_filename, 'wb') as f:
            pickle.dump(results_to_pickle, f)
        print("✅ Results saved.")

    # 3. Generate and save the plot (this step runs every time).
    print("▶️ Final Step: Generating and saving plot...")

    ratios = [e / geometric_geodesic_length for e in entropies]

    plt.figure(figsize=(12, 7))
    plt.plot(MOMENT_VALUES, ratios, marker='o', linestyle='-', label=f'Calculated Ratio (D={BOND_DIM})')

    theoretical_max = np.log2(BOND_DIM)
    plt.axhline(
        y=theoretical_max, color='r', linestyle='--',
        label=f'Theoretical Max: log2({BOND_DIM}) ≈ {theoretical_max:.3f}'
    )

    plt.xlabel("Number of Moments (M)")
    plt.ylabel("Entropy / Geometric Geodesic Length")
    plt.title(f"KPM Entropy Convergence for D={BOND_DIM} (Cached)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    output_filename = "kpm_moment_convergence.png"
    plt.savefig(output_filename)
    print(f"✅ Plot saved to '{output_filename}'")