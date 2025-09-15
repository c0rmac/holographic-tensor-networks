# symmetry_analysis_from_building.py
import numpy as np
from src.holographic_tn.tensor import get_perfect_tensor_of_rank
from src.holographic_tn.geometry import HyperbolicBuilding


# =========================================================================== #
#                      Symmetry Checking Function                             #
# =========================================================================== #

def check_z2_symmetry(tensor: np.ndarray) -> bool:
    """
    Checks if a tensor has a Z2 symmetry by testing for charge conservation.
    """
    bond_dim = tensor.shape[0]
    charges = np.arange(bond_dim) % 2  # |0> -> charge 0; |1> -> charge 1

    print(f"▶️ Numerically testing tensor of shape {tensor.shape} for Z2 symmetry...")

    for indices in np.ndindex(tensor.shape):
        if not np.isclose(tensor[indices], 0.0):
            total_charge = sum(charges[site_idx] for site_idx in indices)
            if total_charge % 2 != 0:
                print(f"❌ Symmetry VIOLATED at index {indices}!")
                return False

    print("✅ SUCCESS: Tensor respects the Z2 symmetry.")
    return True


# =========================================================================== #
#                                Main Analysis                                #
# =========================================================================== #

def main():
    """
    Constructs a HyperbolicBuilding and analyzes its corresponding tensor.
    """
    # 1. Construct the specific HyperbolicBuilding object
    print("--- Constructing the HyperbolicBuilding ---")
    building = HyperbolicBuilding(p=6, v=2)
    layers = 3
    building.construct_tiling(layers=layers)
    print("Building constructed.")

    # 2. Extract the 'p' parameter from the building object
    # This determines the rank of the tensor used in the simulation.
    rank_to_test = building.p
    bond_dim = 2  # Assuming qubit-based tensors

    print(f"\n--- Analyzing Tensors for the Constructed {{{building.p},{building.v * 2}}} Tiling ---")

    # 3. Generate and test the corresponding perfect tensor
    tensor_to_test = get_perfect_tensor_of_rank(rank=rank_to_test, bond_dim=bond_dim)
    has_symmetry = check_z2_symmetry(tensor_to_test)

    # --- Conclusion ---
    print("\n--- CONCLUSION ---")
    if has_symmetry:
        print(f"The rank-{rank_to_test} tensor for this building has a Z2 symmetry.")
        print("You can confidently use block-sparse tensors to resolve memory issues.")
    else:
        print(f"The rank-{rank_to_test} tensor for this building does NOT have a Z2 symmetry.")


if __name__ == '__main__':
    main()