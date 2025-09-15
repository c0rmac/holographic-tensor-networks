import quimb.tensor as qtn
import numpy as np
from typing import List, Tuple, Dict, Set

from src.holographic_tn.geometry import HyperbolicBuilding


# This helper function performs the GILT filtering on a single bond.
def _gilt_filter_bond(t1: qtn.Tensor, t2: qtn.Tensor,
                      env_tensors: List[qtn.Tensor], max_bond: int) -> Tuple[qtn.Tensor, qtn.Tensor]:
    """
    A 'pure' function that performs GILT filtering on two input tensors
    and returns the two new, modified tensors.
    """
    bond_ind = next(iter(t1.bonds(t2)))

    env_tn = qtn.TensorNetwork(env_tensors)
    external_inds = list(env_tn.outer_inds() - {bond_ind})
    E_tensor = env_tn.contract(all, output_inds=external_inds + [bond_ind])

    U, s, Vd = E_tensor.split(
        left_inds=external_inds, method='svds', max_bond=max_bond, get='arrays'
    )
    isometry = Vd.conj().T

    new_bond_ind = qtn.rand_uuid()
    V_tensor = qtn.Tensor(isometry, inds=(bond_ind, new_bond_ind))
    V_adj_tensor = qtn.Tensor(isometry.conj().T, inds=(new_bond_ind, bond_ind))

    new_t1 = (t1 @ V_tensor).reindex_({new_bond_ind: bond_ind})
    new_t2 = (t2 @ V_adj_tensor).reindex_({new_bond_ind: bond_ind})

    return new_t1, new_t2


# This is the main function that executes one full coarse-graining step.
def run_pentagon_tnr_step(
        tn: qtn.TensorNetwork,
        building: HyperbolicBuilding,
        max_bond: int,
        level: int = 0
) -> qtn.TensorNetwork:
    """
    Performs one full coarse-graining step on a {5,q} tiling using
    pre-computed geometric metadata from the HyperbolicBuilding object.
    """
    # 1. Get the geometric blueprint for all plaquettes from the building
    all_pentagons_geom_ids = building.get_plaquettes(level=level)
    plaquette_map, coarse_bonds = building.get_coarse_grained_network_info(level=level)

    # 2. Filter for plaquettes that are fully contained in the current network (tn)
    current_network_tags = tn.tags
    relevant_pentagons = []
    for p_geom_ids in all_pentagons_geom_ids:
        p_tags = {str(gid) for gid in p_geom_ids}
        if p_tags.issubset(current_network_tags):
            relevant_pentagons.append(p_geom_ids)

    if not relevant_pentagons:
        print("[INFO] No complete plaquettes found in the current network. Halting compression.")
        return tn

    # 3. Build the list of tensors for the new, smaller network
    final_tensors = []
    all_plaquette_tags = {str(gid) for p_gids in relevant_pentagons for gid in p_gids}

    # Add any tensors that are not part of a plaquette to be processed
    for tensor in tn:
        if not (tensor.tags & all_plaquette_tags):
            final_tensors.append(tensor.copy())

    # 4. Process each relevant pentagon
    for p_geom_ids in relevant_pentagons:
        # a. Find the tensors for this pentagon using their persistent tags
        p_tags = [str(gid) for gid in p_geom_ids]
        tensors_unordered = tn.select_tensors(p_tags)
        tag_to_tensor_map = {next(iter(t.tags)): t for t in tensors_unordered}
        pentagon_tensors = [tag_to_tensor_map[tag] for tag in p_tags]

        # b. (Optional) Perform GILT filtering on the 5 bonds of the pentagon
        #    This would involve serially calling a helper function like `_gilt_filter_bond`

        # c. Coarse-grain the pentagon using the contract-and-fuse method
        plaquette_tn = qtn.TensorNetwork(pentagon_tensors)
        T_plaquette = plaquette_tn.contract(all, optimize='auto-hq')

        # d. Build the fuse map with the correct, global bond names from the building
        p_key = frozenset(p_geom_ids)
        new_tid = plaquette_map[p_key]
        fuse_map = {}
        outer_indices = plaquette_tn.outer_inds()

        for other_p_geom_ids in relevant_pentagons:
            if p_key == frozenset(other_p_geom_ids): continue

            shared_geom_ids = set(p_geom_ids).intersection(set(other_p_geom_ids))
            if len(shared_geom_ids) >= 2:  # Adjacent plaquettes share an edge
                other_p_key = frozenset(other_p_geom_ids)
                other_new_tid = plaquette_map.get(other_p_key)

                if other_new_tid:
                    new_bond_name = coarse_bonds.get((new_tid, other_new_tid))
                    if new_bond_name:
                        # The indices on the shared tensors that are also outer indices form the limb
                        shared_tags = {str(gid) for gid in shared_geom_ids}
                        limb_tensors = plaquette_tn.select_tensors(shared_tags)
                        limb_inds = tuple(qtn.TensorNetwork(limb_tensors).outer_inds() & outer_indices)
                        if limb_inds:
                            fuse_map[new_bond_name] = limb_inds

        # Fuse the external limbs into new, correctly named legs
        T_plaquette.fuse_(fuse_map, inplace=True)

        # Add a persistent tag to the new tensor for the next iteration
        T_plaquette.add_tag(str(p_geom_ids[0]))

        final_tensors.append(T_plaquette)

    # 5. Return the new, smaller, and correctly connected tensor network
    return qtn.TensorNetwork(final_tensors)


def compress_network_bonds(tn: qtn.TensorNetwork, max_bond: int):
    """
    Compresses a tensor network by iteratively truncating its internal bonds.
    This is the robust method for simplifying a network on an arbitrary graph.

    Args:
        tn: The tensor network to compress, modified in-place.
        max_bond: The maximum bond dimension to keep for any bond.
    """
    # Find all internal bonds with a dimension greater than max_bond
    bonds_to_compress = []
    for ix, tids in tn.ind_map.items():
        if len(tids) == 2 and tn.ind_size(ix) > max_bond:
            bonds_to_compress.append((ix, tids))

    if not bonds_to_compress:
        return

    for ix, (tid1, tid2) in bonds_to_compress:
        # Re-check in case a previous compression affected this bond
        if tid1 not in tn.tensor_map or tid2 not in tn.tensor_map:
            continue
        if tn.ind_size(ix) <= max_bond:
            continue

        t1, t2 = tn[tid1], tn[tid2]
        T_block = t1 @ t2

        left_inds = t1.inds_to(t2, 'exclusive')
        right_inds = t2.inds_to(t1, 'exclusive')

        new_t1, new_t2 = T_block.split(
            left_inds, right_inds, max_bond=max_bond, get='tensors'
        )

        # In-place update: remove old tensors and add new ones
        tn.tensor_map[tid1] = new_t1
        tn.tensor_map[tid2] = new_t2


import quimb.tensor as qtn
from typing import Tuple


def coarse_grain_step(tn: qtn.TensorNetwork) -> float:
    """
    Performs one step of coarse-graining and returns the entanglement entropy
    of the contracted bond as a measure of the step's impact.

    Args:
        tn: The tensor network to coarse-grain, modified in-place.

    Returns:
        The von Neumann entropy of the removed bond. Returns 0.0 if no
        bonds are contracted.
    """
    inner_bonds = tn.inner_inds()

    if not inner_bonds:
        print("[INFO] No internal bonds left to contract.")
        return 0.0

    bond_to_contract = next(iter(inner_bonds))

    # --- Calculate Entanglement Entropy Before Contracting ---
    tid1, tid2 = tn.ind_map[bond_to_contract]
    t1, t2 = tn.tensor_map[tid1], tn.tensor_map[tid2]

    # --- THE FIX ---
    # 1. Correctly identify the external indices of t1.
    #    These are all of t1's indices EXCEPT the one(s) connecting it to t2.
    left_inds = tuple(set(t1.inds) - set(t1.bonds(t2)))

    # 2. Perform SVD across the t1-t2 partition to get the bond spectrum.
    u, s, v = (t1 @ t2).split(left_inds=left_inds, get='arrays', absorb=None)
    # --- END FIX ---

    # Normalize the squared singular values to get probabilities
    s_sq = s ** 2
    probs = s_sq / np.sum(s_sq)

    # Calculate von Neumann entropy (ignoring zero probabilities for log2)
    entropy = -np.sum(p * np.log2(p) for p in probs if p > 1e-12)
    # ---

    # Get the tags of the two tensors connected by this bond
    tag1 = next(iter(t1.tags))
    tag2 = next(iter(t2.tags))

    # Now, perform the contraction
    tn.contract_tags([tag1, tag2], inplace=True)

    return entropy