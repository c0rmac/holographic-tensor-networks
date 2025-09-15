# src/holographic_tn/renormalization.py
import quimb.tensor as qtn
import numpy as np
from typing import List, Tuple, Dict, Set


def _gilt_filter_bond(tn, t1, t2, env_tensors: List[qtn.Tensor], max_bond):
    """
    Performs the GILT filtering sub-step on a single bond.
    This is the "splitting" stage.
    """
    # Fetch the tensor objects from the network
    #t1 = tn.tensor_map[t1_id]
    #t2 = tn.tensor_map[t2_id]

    bond_ind = next(iter(t1.bonds(t2)))

    # 1. Compute the isometry from the environment
    env_tn = qtn.TensorNetwork(env_tensors)
    external_inds = list(set(env_tn.outer_inds()) - {bond_ind})
    E_tensor = env_tn.contract(all, output_inds=external_inds + [bond_ind])

    U, s, Vd = E_tensor.split(
        left_inds=external_inds, method='svds', max_bond=max_bond, get='arrays', absorb=None
    )
    isometry = Vd.conj().T  # Shape: (D_old, D_new)

    # 2. Create the two tensors that perform the truncation
    new_bond_ind = qtn.rand_uuid()

    # V_tensor maps the old bond to the new, smaller bond
    V_tensor = qtn.Tensor(isometry, inds=(bond_ind, new_bond_ind))

    # V_adj_tensor maps the new bond back to the old bond's space
    V_adj_tensor = qtn.Tensor(isometry.conj().T, inds=(new_bond_ind, bond_ind))

    # 3. Absorb these tensors into t1 and t2
    # The rank of the tensors is preserved.
    new_t1 = t1 @ V_tensor
    new_t2 = t2 @ V_adj_tensor

    # The two new tensors are now connected by `new_bond_ind`.
    # We can relabel this new bond back to the original name for consistency.
    new_t1.reindex_({new_bond_ind: bond_ind}, inplace=True)
    new_t2.reindex_({new_bond_ind: bond_ind}, inplace=True)

    return new_t1, new_t2


def run_gilt_tnr_step(
        tn: qtn.TensorNetwork,
        max_bond: int
) -> qtn.TensorNetwork:
    """
    Performs one full Gilt-TNR iteration using a robust, bond-based
    method for finding plaquettes.
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

    total_error_sq = 0.0

    # 2. Process each plaquette and add the new coarse-grained tensor
    for p_ids_unordered in plaquettes:

        # --- Sort the plaquette into a canonical geometric order ---
        # This ensures the contractions are always applied symmetrically.
        tA_id = p_ids_unordered[0]
        neighbors_of_A = set(tn._get_neighbor_tids(tA_id))
        p_ids_set = set(p_ids_unordered)

        neighbors_in_plaquette = list(neighbors_of_A.intersection(p_ids_set - {tA_id}))
        tB_id = neighbors_in_plaquette[0]  # This is now 'tB' (e.g., top-right)
        tC_id = neighbors_in_plaquette[1]  # This is now 'tC' (e.g., bottom-left)

        # The last remaining tensor is the diagonal one, 'tD'
        tD_id = list(p_ids_set - {tA_id, tB_id, tC_id})[0]

        # Now we have a guaranteed order: A=top-left, B=top-right, etc.
        p_ids = [tA_id, tB_id, tC_id, tD_id]
        tA, tB, tC, tD = [tn.tensor_map[pid].copy() for pid in p_ids]

        # --- STAGE 1: GILT Filtering (serially update the tensor objects) ---
        # (This stage is optional for basic TRG but included for the full Gilt-TNR)
        current_p_tensors = [tA, tB, tC, tD]
        tA, tB = _gilt_filter_bond(tn, tA, tB, current_p_tensors, max_bond)

        current_p_tensors = [tA, tB, tC, tD]
        tA, tC = _gilt_filter_bond(tn, tA, tC, current_p_tensors, max_bond)

        current_p_tensors = [tA, tB, tC, tD]
        tB, tD = _gilt_filter_bond(tn, tB, tD, current_p_tensors, max_bond)

        current_p_tensors = [tA, tB, tC, tD]
        tC, tD = _gilt_filter_bond(tn, tC, tD, current_p_tensors, max_bond)

        # --- STAGE 2: TRG COARSE-GRAINING (Definitive Version) ---

        plaquette_error_sq = 0.0  # Accumulator for this specific plaquette

        def split_and_track_error(t, **kwargs):
            """Helper function to perform a split and accumulate the squared error."""
            nonlocal plaquette_error_sq
            # Use ret_info=True to get a dict with truncation info
            L, R = t.split(get='tensors', absorb='both', **kwargs)
            # 'discarded_sum_sq' is the sum of the squares of discarded singular values
            # plaquette_error_sq += info['discarded_sum_sq']
            return L, R

        # 1. Decompose all four tensors, tracking the error from each split
        A_L, A_R = split_and_track_error(tA, left_inds=tA.bonds(tC), max_bond=max_bond)
        A_U, A_D = split_and_track_error(tA, left_inds=tA.bonds(tB), max_bond=max_bond)

        B_L, B_R = split_and_track_error(tB, left_inds=tB.bonds(tA), max_bond=max_bond)
        B_U, B_D = split_and_track_error(tB, left_inds=tB.bonds(tD), max_bond=max_bond)

        C_L, C_R = split_and_track_error(tC, left_inds=tC.bonds(tD), max_bond=max_bond)
        C_U, C_D = split_and_track_error(tC, left_inds=tC.bonds(tA), max_bond=max_bond)

        D_L, D_R = split_and_track_error(tD, left_inds=tD.bonds(tC), max_bond=max_bond)
        D_U, D_D = split_and_track_error(tD, left_inds=tD.bonds(tB), max_bond=max_bond)

        # 2. Recombine the factors corresponding to the four internal bonds.
        T_horiz_up = A_R @ B_L
        T_horiz_down = C_R @ D_L
        T_vert_left = A_D @ C_U
        T_vert_right = B_D @ D_U

        # 3. Contract the four intermediate tensors to form the final rank-4 tensor.
        temp_tn = qtn.TensorNetwork([T_horiz_up, T_horiz_down, T_vert_left, T_vert_right])
        new_coarse_tensor = temp_tn.contract(all, optimize='auto-hq')

        final_tensors.append(new_coarse_tensor)

        total_error_sq += plaquette_error_sq

    return qtn.TensorNetwork(final_tensors)

def _run_gilt_tnr_step(
        tn: qtn.TensorNetwork,
        max_bond: int,
        ket_tags: Set[str]
) -> qtn.TensorNetwork:
    """
    Performs one full, two-stage Gilt-TNR iteration on the ket network.

    Args:
        tn: The tensor network (just the ket part).
        max_bond: The maximum bond dimension.
        ket_tags: Tags identifying all tensors in the network.

    Returns:
        The new, coarse-grained tensor network.
    """
    tn_filtered = tn.copy()

    # === STAGE 1: GILT FILTERING ("Splitting") ===
    # This stage modifies the tensors in-place to remove local correlations.

    # Identify plaquettes to operate on (e.g., checkerboard pattern)
    plaquettes = []
    processed_tids = set()
    plaquette_tids = set()

    for tid, t_corner in tn_filtered.select(ket_tags, which='any').tensor_map.items():
        if tid in processed_tids:
            continue

        # neighbors = tn_filtered.neighbors(t_corner.tid)
        neighbors = list(tn_filtered._get_neighbor_tids(tid))
        if len(neighbors) < 2: continue

        # Simple heuristic to find a plaquette
        t_right_id = neighbors[0]
        t_down_id = neighbors[1]

        common_neighbors = set(tn_filtered._get_neighbor_tids(t_right_id)).intersection(
            set(tn_filtered._get_neighbor_tids(t_down_id)))
        common_neighbors.discard(tid)
        if not common_neighbors: continue
        t_diag_id = common_neighbors.pop()

        # Collect the IDs and tensor objects for the plaquette
        plaquette_ids = [tid, t_right_id, t_down_id, t_diag_id]
        p_tensors = [tn_filtered.tensor_map[pid] for pid in plaquette_ids]

        plaquettes.append({'ids': plaquette_ids, 'tensors': p_tensors})
        processed_tids.update(plaquette_ids)

        plaquette_tids.update(set(plaquette_ids))

    # === Build the list of tensors for the new network ===
    final_tensors = []

    # 1. Add all tensors that are NOT part of any plaquette
    for tid, tensor in tn.tensor_map.items():
        if tid not in plaquette_tids:
            final_tensors.append(tensor.copy())

    # 2. Process each plaquette and add the new coarse-grained tensor
    for p_ids in plaquettes:
        # Get the initial tensor objects for this plaquette
        tA, tB, tC, tD = [tn.tensor_map[pid].copy() for pid in p_ids['ids']]
        tA_id, tB_id, tC_id, tD_id = [pid for pid in p_ids['ids']]

        # STAGE 1: GILT Filtering (serially update the tensor objects)
        current_p_tensors = [tA, tB, tC, tD]
        tA, tB = _gilt_filter_bond(tn, tA, tB, current_p_tensors, max_bond)

        current_p_tensors = [tA, tB, tC, tD]
        tA, tC = _gilt_filter_bond(tn, tA, tC, current_p_tensors, max_bond)

        current_p_tensors = [tA, tB, tC, tD]
        tB, tD = _gilt_filter_bond(tn, tB, tD, current_p_tensors, max_bond)

        current_p_tensors = [tA, tB, tC, tD]
        tC, tD = _gilt_filter_bond(tn, tC, tD, current_p_tensors, max_bond)

        # STAGE 2: Coarse-Graining (contract the final filtered tensors)
        new_coarse_tensor = qtn.TensorNetwork([tA, tB, tC, tD]).contract(all)

        # --- FIX START ---
        # To ensure the new tensor is found in the next iteration,
        # we copy the persistent tags from the original corner tensor.
        original_corner_tensor = tn.tensor_map[tA_id]
        for tag in original_corner_tensor.tags:
            if not tag.startswith('__'):  # Avoids copying temporary/internal tags
                new_coarse_tensor.add_tag(tag)
        # --- FIX END ---

        final_tensors.append(new_coarse_tensor)

    # 3. Create the new, smaller tensor network from the final list
    return qtn.TensorNetwork(final_tensors)