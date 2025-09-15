# validation/rt_nontrivial.py

import numpy as np
# - FIX -# Add necessary import for quimb
import quimb.tensor as qtn

from src.holographic_tn.geometry import HyperbolicBuilding
# - FIX -# Import the correct, renamed function from physics.py
from src.holographic_tn.physics import calculate_rt_entropy, build_network_from_building


def main():
    print("--- Non-trivial Ryu-Takayanagi Validation ---\n")

    # 1. Build a simple 2×2 patch of the bulk
    building = HyperbolicBuilding(p=4, v=4)
    building.construct_2x2_grid(initial_coords=np.array([0.0, 0.0]))
    face_ids = list(building.simplicial_complex.nodes())

    print(f"Constructed 2×2 grid with faces: {face_ids}\n")

    # 2. Create the holographic tensor network
    # - FIX -# Call the renamed function and use 'tn' as the variable name
    tn = build_network_from_building(building, bond_dim=2)
    print("Tensor network created with bond dimension 2.\n")

    # 3. Pick two opposite-corner faces for region A
    #    e.g. face_ids[0] = bottom-left, face_ids[3] = top-right
    f0, f3 = face_ids[0], face_ids[3]

    # - FIX -# Select tensors by tag and find dangling indices using the correct quimb methods
    t0 = tn.select(str(f0)).tensors[0]
    t3 = tn.select(str(f3)).tensors[0]

    inds_f0 = list(set(t0.inds) & set(tn.outer_inds()))
    inds_f3 = list(set(t3.inds) & set(tn.outer_inds()))

    boundary_region_inds = inds_f0 + inds_f3

    print(f"Boundary region A on faces {f0} & {f3}, total legs = {len(boundary_region_inds)}\n")

    # 4. Run the RT entropy calculation
    # - FIX -# Update the function call to use the correct parameters for selecting the BP contractor
    results = calculate_rt_entropy(building, tn, boundary_region_inds)

    # 5. Print & verify
    S = results["entropy"]
    L = results["geometric_geodesic_length"]
    R = S / L if L > 0 else float("inf")

    print("\n--- RESULTS ---")
    print(f"Entanglement Entropy S(A)        = {S:.4f} bits")
    print(f"Geodesic Length length(γ_A)      = {L:.4f}")
    print(f"Ratio S(A) / length(γ_A)         = {R:.4f}")

    if L > 0:
        print("\nValidation successful: non-trivial minimal surface detected.")
    else:
        print("\nUnexpected zero-length geodesic. Check face ordering.")


if __name__ == "__main__":
    main()