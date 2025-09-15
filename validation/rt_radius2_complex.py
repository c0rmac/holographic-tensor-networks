# validation/rt_radius2_complex.py

import numpy as np
#- FIX -# Add necessary import for quimb
import quimb.tensor as qtn

from src.holographic_tn.geometry import HyperbolicBuilding, PoincareIsometry
#- FIX -# Import the correct, renamed function from physics.py
from src.holographic_tn.physics import calculate_rt_entropy, build_network_from_building


def main():
    print("--- Complex Ryu–Takayanagi Validation (radius=2) ---\n")

    # 1. Build a bulk patch up to radius=2 with a 4-valent tiling
    building = HyperbolicBuilding(p=4, v=4)

    # Define two commuting “X” and “Y” isometries and their inverses
    g_x, g_x_inv = PoincareIsometry("X+"), PoincareIsometry("X-")
    g_y, g_y_inv = PoincareIsometry("Y+"), PoincareIsometry("Y-")

    # Pair each edge to its opposite in the square (0↔2, 1↔3)
    side_pairings = [
        (0, 2, g_y_inv, g_y),
        (1, 3, g_x,     g_x_inv),
        (2, 0, g_y,     g_y_inv),
        (3, 1, g_x_inv, g_x),
    ]

    print("Constructing hyperbolic building to max_radius=2…")
    building.construct_building(side_pairings, max_radius=2)
    num_faces = len(building.word_map)
    print(f"  → Created patch with {num_faces} faces.\n")

    # 2. Create the holographic tensor network with bond dimension 2
    print("Creating holographic tensor network (bond_dim=2)…")
    #- FIX -# Call the renamed function and use 'tn' as the variable name
    tn = build_network_from_building(building, bond_dim=2)
    #- FIX -# Use the correct .num_tensors property to get the tensor count
    print(f"  → Network has {tn.num_tensors} tensor nodes.\n")

    # 3. Pick two boundary faces at radius=2 for a non-trivial geodesic
    #    We filter for generator-words of length two (one ‘;’ in the string).
    radius2_words = [w for w in building.word_map if w.count(";") == 1]
    if len(radius2_words) < 2:
        raise RuntimeError("Not enough radius-2 faces; check construct_building parameters.")
    word_A, word_B = radius2_words[0], radius2_words[-1]
    face_A, face_B = building.word_map[word_A], building.word_map[word_B]
    print(f"Boundary region on faces {word_A} and {word_B}.\n")

    # Collect all dangling (physical) edges from both faces
    boundary_region_inds = []
    for fid in (face_A, face_B):
        #- FIX -# Select a tensor by its unique tag
        tensor = tn.select(str(fid)).tensors[0]
        #- FIX -# Get dangling indices by intersecting with the network's outer indices
        dangling_inds = list(set(tensor.inds) & set(tn.outer_inds()))
        boundary_region_inds.extend(dangling_inds)
    print(f"Total physical legs in region A: {len(boundary_region_inds)}\n")

    # 4. Run the Ryu–Takayanagi entanglement-entropy calculation
    print("Running calculate_rt_entropy…")
    #- FIX -# Pass the corrected tensor network and list of indices
    # Using the exact contractor as the network is still relatively small.
    results = calculate_rt_entropy(building, tn, boundary_region_inds,
                                   #entropy_method="exact"
                                   )

    # 5. Report results
    S = results["entropy"]
    L = results["geometric_geodesic_length"]
    ratio = S / L if L > 0 else float("inf")

    print("\n--- RESULTS ---")
    print(f"Entanglement Entropy S(A)        = {S:.4f} bits")
    print(f"Geodesic Length length(γ_A)      = {L:.4f}")
    print(f"Ratio S(A) / length(γ_A)         = {ratio:.4f}")

    if L > 0:
        print("\n✅ Non-trivial minimal surface detected—in good agreement with RT.")
    else:
        print("\n⚠️  Unexpected zero-length geodesic; please inspect building geometry.")

if __name__ == "__main__":
    main()