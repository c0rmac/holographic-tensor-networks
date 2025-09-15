# validation/rt_formula_validation.py

import numpy as np
import quimb.tensor as qtn

from src.holographic_tn.geometry import HyperbolicBuilding, PoincareIsometry
from src.holographic_tn.physics import calculate_rt_entropy, build_network_from_building


def main():
    """
    A self-verifying validation script to benchmark the library against the
    HaPPY code, with assertions to check each step.
    """
    print("--- HaPPY Code Benchmark Script ---")

    # --- 1. Construct the HaPPY Code Geometry ---
    print("\n[Phase 1: Geometry]")
    print("Building a single apartment with a pentagonal tiling (p=5)...")

    building = HyperbolicBuilding(p=5, v=4)

    # Note: Using placeholder isometries as the exact values are not needed for topology validation.
    generators = [PoincareIsometry(f"g{i}") for i in range(5)]
    inverses = [g.get_inverse(f"{g.name}_inv") for g in generators]

    side_pairings = [
        (0, 2, generators[0], inverses[0]), (1, 3, generators[1], inverses[1]),
        (2, 4, generators[2], inverses[2]), (3, 0, generators[3], inverses[3]),
        (4, 1, generators[4], inverses[4]),
    ]
    side_pairings.extend([(p[1], p[0], p[3], p[2]) for p in side_pairings])

    building.construct_building(side_pairings, max_radius=1)

    # --- ASSERTIONS for Geometry ---
    assert len(building.word_map) == 6, "Error: Incorrect number of faces generated."
    adj_edges = [e for e in building.simplicial_complex.edges(data=True) if e[2].get('type') == 'adjacency']
    assert len(adj_edges) == 5, "Error: Incorrect number of edges in the geometry."
    print(
        f"✅ Geometry validation passed: Built a patch with {len(building.word_map)} faces and {len(adj_edges)} connections.")

    # --- 2. Create the Holographic State ---
    print("\n[Phase 2: Quantum State]")
    tn = build_network_from_building(building, bond_dim=2)

    # --- ASSERTIONS for Tensor Network ---
    assert tn.num_tensors == len(building.word_map), "Error: Number of tensors does not match number of faces."

    sample_tensor = tn.tensors[0]

    assert len(sample_tensor.inds) == 5, "Error: Tensors have incorrect rank for a pentagonal tiling."
    print("✅ Tensor network validation passed.")

    # --- 3. Define a Boundary Region A ---
    print("\n[Phase 3: Boundary Region]")
    boundary_face_word = "g0"
    boundary_face_id = building.word_map.get(boundary_face_word)

    assert boundary_face_id is not None, f"Error: Could not find boundary face '{boundary_face_word}'."
    print(f"Defining boundary region A on face {boundary_face_id}...")

    boundary_tensor = tn.select(str(boundary_face_id)).tensors[0]

    # --- FIX IS HERE ---
    # Find the intersection of the tensor's indices and the network's outer indices.
    boundary_region_inds_A = list(set(boundary_tensor.inds) & set(tn.outer_inds()))

    # --- ASSERTION for Boundary Region ---
    assert len(boundary_region_inds_A) == 4, "Error: Boundary face has incorrect number of dangling edges."
    print(f"✅ Boundary region validation passed: Region A defined by {len(boundary_region_inds_A)} indices.")

    # --- 4. Perform the Holographic Calculation ---
    print("\n[Phase 4: Holographic Calculation]")
    # Using the exact contractor for this small validation network.
    results = calculate_rt_entropy(building, tn, boundary_region_inds_A)

    # --- 5. Output for Benchmarking & Final Assertions ---
    print("\n--- BENCHMARK RESULTS ---")
    entropy = results["entropy"]
    cut_length = results["cut_length"]

    print(f"\n  ▶️  Quantum Calculation: Entanglement Entropy S(A) = {entropy:.4f} bits")
    print(f"  ▶️  Geometric Ground Truth: Minimal Cut Length |γ_A| = {cut_length}")

    # --- Ground Truth Assertion ---
    # The RT formula for perfect tensors predicts S_A == |γ_A| * log2(D).
    # For bond_dim D=2, log2(D)=1, so S_A == |γ_A|.
    # The tensors created by get_perfect_tensor_of_rank are true perfect tensors.
    assert np.isclose(entropy, cut_length), "Validation Failed: Entropy does not match the cut length."

    print("\n✅ Ground truth validation passed: S_A ≈ |γ_A|, as predicted by the Ryu-Takayanagi formula.")


if __name__ == "__main__":
    main()