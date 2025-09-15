import quimb.tensor as qtn
import numpy as np


def coarse_grain_step(tn: qtn.TensorNetwork):
    """
    Performs one step of coarse-graining by finding an internal bond
    and contracting the two tensors across it. Modifies the network in-place.
    """
    inner_bonds = tn.inner_inds()
    if not inner_bonds:
        return

    bond_to_contract = next(iter(inner_bonds))
    tid1, tid2 = tn.ind_map[bond_to_contract]
    tag1 = next(iter(tn.tensor_map[tid1].tags))
    tag2 = next(iter(tn.tensor_map[tid2].tags))

    tn.contract_tags([tag1, tag2], inplace=True)


def compress_network_bonds(tn: qtn.TensorNetwork, max_bond: int):
    """
    Compresses a tensor network by truncating internal bonds larger than max_bond.
    """
    bonds_to_compress = [
        (ix, tids) for ix, tids in tn.ind_map.items()
        if len(tids) == 2 and tn.ind_size(ix) > max_bond
    ]

    for ix, (tid1, tid2) in bonds_to_compress:
        if tid1 not in tn.tensor_map or tid2 not in tn.tensor_map:
            continue

        t1, t2 = tn.tensor_map[tid1], tn.tensor_map[tid2]
        T_block = t1 @ t2

        left_inds = tuple(set(t1.inds) - set(t1.bonds(t2)))
        right_inds = tuple(set(t2.inds) - set(t2.bonds(t1)))

        new_t1, new_t2 = T_block.split(
            left_inds, right_inds, max_bond=max_bond, get='tensors'
        )

        tn[tid1] = new_t1
        tn[tid2] = new_t2


# This is the new main function you would call in a loop
def run_coarse_grain_and_compress_step(tn: qtn.TensorNetwork, max_bond: int):
    """
    Performs a single, size-controlled coarse-graining step.
    1. Contracts two tensors to reduce the tensor count by one.
    2. Compresses any new bonds that grew larger than max_bond.
    """
    if not tn.inner_inds():
        print("[INFO] Coarse-graining complete.")
        return

    # Step 1: Reduce tensor count
    coarse_grain_step(tn)

    # Step 2: Control the size of any new, large bonds
    compress_network_bonds(tn, max_bond)