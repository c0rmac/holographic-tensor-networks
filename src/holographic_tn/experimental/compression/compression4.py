# In your compression.py or renormalization.py file

import quimb.tensor as qtn
import numpy as np
from typing import List, Tuple, Dict, Set


def _gilt_filter_bond(t1: qtn.Tensor, t2: qtn.Tensor,
                      env_tensors: List[qtn.Tensor], max_bond: int) -> Tuple[qtn.Tensor, qtn.Tensor]:
    """
    A 'pure' function that performs GILT filtering on two input tensors
    and returns the two new, modified tensors.
    """
    bond_ind = next(iter(t1.bonds(t2)))

    # 1. Compute the isometry from the environment
    env_tn = qtn.TensorNetwork(env_tensors)
    external_inds = list(set(env_tn.outer_inds()) - {bond_ind})
    E_tensor = env_tn.contract(all, output_inds=external_inds + [bond_ind])

    U, s, Vd = E_tensor.split(
        left_inds=external_inds, method='svds', max_bond=max_bond, get='arrays', absorb=None
    )
    isometry = Vd.conj().T

    # 2. Create the tensors that perform the truncation
    new_bond_ind = qtn.rand_uuid()
    V_tensor = qtn.Tensor(isometry, inds=(bond_ind, new_bond_ind))
    V_adj_tensor = qtn.Tensor(isometry.conj().T, inds=(new_bond_ind, bond_ind))

    # 3. Create the new tensors by contracting the projectors
    new_t1 = (t1 @ V_tensor).reindex_({new_bond_ind: bond_ind})
    new_t2 = (t2 @ V_adj_tensor).reindex_({new_bond_ind: bond_ind})

    return new_t1, new_t2


def run_gilt_tnr_step(
        tn: qtn.TensorNetwork,
        max_bond: int,
        max_rank: int = 5
) -> qtn.TensorNetwork:
    """
    Performs one full Gilt-TNR iteration by building a new network.
    """
    # --- STAGE 0: Find Plaquettes Robustly ---
    bonds = [tuple(bond) for bond in set(map(frozenset,
                                             (tids for tids in tn.ind_map.values() if len(tids) == 2)))]
    print(f"[INFO] Found {len(bonds)} unique bonds.")

    plaquettes = []
    processed_ids = set()

    for tid1, tid2 in bonds:
        if tid1 in processed_ids or tid2 in processed_ids:
            continue

        # Get neighbors of the two tensors on the bond (excluding each other)
        t1_other_neighbors = set(tn._get_neighbor_tids(tid1)) - {tid2}
        t2_other_neighbors = set(tn._get_neighbor_tids(tid2)) - {tid1}
        print(f"[DEBUG]   - t1's other neighbors: {t1_other_neighbors}")
        print(f"[DEBUG]   - t2's other neighbors: {t2_other_neighbors}")

        # Iterate through t1's neighbors to find the rest of the square
        found_plaquette_for_this_bond = False
        for tid3 in t1_other_neighbors:
            # Look for a t4 that is a neighbor of t3 AND t2
            tid3_neighbors = set(tn._get_neighbor_tids(tid3))
            completing_tids = tid3_neighbors.intersection(t2_other_neighbors)
            print(f"[DEBUG]     - Checking via t3={tid3}. Neighbors of t3: {tid3_neighbors}")
            print(f"[DEBUG]     - Intersection with t2's neighbors: {completing_tids}")

            # This intersection finds the fourth corner of the square
            completing_tids = tid3_neighbors.intersection(t2_other_neighbors)

            if completing_tids:
                tid4 = completing_tids.pop()
                p_ids = [tid1, tid2, tid3, tid4]
                plaquettes.append(p_ids)
                processed_ids.update(p_ids)
                print(f"[SUCCESS] Found plaquette: {p_ids}")
                found_plaquette_for_this_bond = True
                break  # Found the plaquette for this bond, move to the next

        if found_plaquette_for_this_bond:
            print(f"[INFO] Marking IDs {p_ids} as processed.")

    print(f"\n--- Plaquette search finished. Found {len(plaquettes)} plaquettes. ---")

    if not plaquettes:
        return tn

    # --- Build the list for the new network ---
    final_tensors = []
    plaquette_tensor_ids = {pid for p_ids in plaquettes for pid in p_ids}

    # 1. Add tensors that were not part of any processed plaquette
    for tid, tensor in tn.tensor_map.items():
        if tid not in plaquette_tensor_ids:
            final_tensors.append(tensor.copy())

    # 2. Process each plaquette and add the new coarse-grained tensor
    for p_ids in plaquettes:
        # Get the initial tensor objects for this plaquette
        tA, tB, tC, tD = [tn.tensor_map[pid].copy() for pid in p_ids]

        # STAGE 1: GILT Filtering (serially update the tensor objects)
        current_p_tensors = [tA, tB, tC, tD]
        tA, tB = _gilt_filter_bond(tA, tB, current_p_tensors, max_bond)

        current_p_tensors = [tA, tB, tC, tD]
        tA, tC = _gilt_filter_bond(tA, tC, current_p_tensors, max_bond)

        current_p_tensors = [tA, tB, tC, tD]
        tB, tD = _gilt_filter_bond(tB, tD, current_p_tensors, max_bond)

        current_p_tensors = [tA, tB, tC, tD]
        tC, tD = _gilt_filter_bond(tC, tD, current_p_tensors, max_bond)

        # STAGE 2: Coarse-Graining (contract the final filtered tensors)
        plaquette_tn = qtn.TensorNetwork([tA, tB, tC, tD])
        new_coarse_tensor = plaquette_tn.contract(all, optimize='auto-hq')

        # Conditionally fuse the tensor if its rank is too high
        if len(new_coarse_tensor.shape) > max_rank:
            # c. Find all unique neighboring tensors that are outside the plaquette
            outer_indices = plaquette_tn.outer_inds()
            neighbor_tids = set()
            for ix in outer_indices:
                # For each outer index, find the tensors it connects to and remove the ones
                # that are inside the current plaquette, leaving only external neighbors.
                neighbor_tids.update(set(tn.ind_map[ix]) - set(p_ids))

            # d. Build the fuse map based on these external neighbors
            fuse_map = {}
            for neighbor_id in neighbor_tids:
                neighbor_tensor = tn.tensor_map[neighbor_id]

                # The "limb" is the set of all indices shared between the plaquette
                # and this specific external neighbor.
                limb_inds = tuple(set(outer_indices) & set(neighbor_tensor.inds))

                if limb_inds:
                    # The name of the new fused leg is based on the neighbor's ID.
                    # This creates a unique, meaningful index for the new connection.
                    fuse_map[f'bond_to_{neighbor_id}'] = limb_inds

            # --- THE FIX ---
            # c. Now, handle the physical boundary legs
            remaining_indices = outer_indices - fused_indices
            if remaining_indices:
                # Group all remaining physical/boundary legs into a single new leg.
                fuse_map['phys_leg'] = tuple(remaining_indices)
            # --- END FIX ---

            # e. Fuse the limbs to create the new tensor
            if fuse_map:
                new_coarse_tensor = new_coarse_tensor.fuse(fuse_map, inplace=True)

        # The result is now a single, rank-4 tensor with the correct connectivity.
        final_tensors.append(new_coarse_tensor)

    # 3. Create the new, smaller tensor network from the final list
    return qtn.TensorNetwork(final_tensors)