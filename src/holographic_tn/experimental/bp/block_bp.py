#!/usr/bin/env python3
"""
block_bp.py

Block Belief Propagation (BlockBP) for arbitrary quimb.TensorNetwork.
Each tensor is its own "block". Messages are MPS on the shared boundary
indices. We split a contracted tensor into an MPS via MatrixProductState.from_dense.
"""

import numpy as np
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState
from typing import Any, Dict, Tuple, List, Set

def _split_tensor_to_mps(
    T: qtn.Tensor,
    boundary: Tuple[str, ...],
    chi: int
) -> MatrixProductState:
    """
    Flatten T on the legs in `boundary` → vector,
    build an MPS via from_dense(vec, dims=…),
    then rename each MPS site’s last index back to the real boundary name.
    """
    # 1) gather the physical dims in boundary-order
    dims = [T.ind_size(i) for i in boundary]
    vec  = T.data.reshape(-1)

    # 2) build & truncate the MPS
    mps = MatrixProductState.from_dense(vec, dims=dims, max_bond=chi)

    # 3) reindex dummy 'k0','k1',… → the REAL physical leg names
    for site_idx, site in enumerate(mps):
        old = site.inds[-1]
        site.reindex({old: boundary[site_idx]}, inplace=True)

    # 4) normalize & return
    mps.normalize()
    return mps


def _mps_to_tensor(
    mps: MatrixProductState,
    shared: Tuple[str, ...]
) -> qtn.Tensor:
    """
    Wrap the MPS into a tiny TensorNetwork and
    contract away every index except those in `shared`.
    """
    tn = qtn.TensorNetwork()
    for i, T in enumerate(mps):
        tn.add_tensor(T.copy(), f"site_{i}")
    # `contract(output_inds=…)` will trace out *all* other legs,
    # leaving exactly the shared physical ones.
    return tn.contract(output_inds=shared)


def init_block_messages(
    boundary_map: Dict[Tuple[Any,Any], Tuple[str,...]],
    tn:           qtn.TensorNetwork,
    chi_msg:      int
) -> Dict[Tuple[Any,Any], MatrixProductState]:
    """
    For each directed edge (A→B) that shares *at least one* physical leg,
    create a random Tensor on exactly those legs and split it to an MPS.
    """
    msgs: Dict[Tuple[Any,Any], MatrixProductState] = {}
    for (A, B), shared in boundary_map.items():
        # random data on the shared dims
        dims = [tn.ind_size(i) for i in shared]
        data = np.random.randn(*dims)
        T_rand = qtn.Tensor(data, inds=shared)

        # split → MPS over that exact boundary
        msgs[(A, B)] = _split_tensor_to_mps(T_rand, shared, chi_msg)

    return msgs

def update_block_message(
    A:        Any,
    B:        Any,
    blocks:   Dict[Any, List[Any]],
    adj:      Dict[Any, Set[Any]],
    tn:       qtn.TensorNetwork,
    messages: Dict[Tuple[Any,Any], MatrixProductState],
    chi_msg:  int,
    shared:   Tuple[str, ...],
) -> MatrixProductState:
    """
    Compute the new MPS on (A→B) by
      1) building the sub‐TN of block A plus all incoming msgs except from B,
      2) contracting *everything* except the `shared` legs,
      3) splitting that resulting Tensor back into an MPS over `shared`.
    """
    sub = qtn.TensorNetwork()

    # add the single‐tensor block A
    for tid in blocks[A]:
        sub.add_tensor(tn.tensor_map[tid].copy(), tid)

    # add every incoming MPS → A except the one from B → A
    for (X, Y), mps in messages.items():
        if Y == A and X != B:
            for i, Tm in enumerate(mps):
                sub.add_tensor(Tm.copy(), f"msg_{X}_{A}_{i}")

    # contract out all legs except the physical `shared`
    contracted = sub.contract(output_inds=shared)
    print(" AFTER CONTRACT:", contracted.inds)

    # split that back into an MPS over exactly `shared`
    return _split_tensor_to_mps(contracted, shared, chi_msg)

def partition_into_blocks(
    tn: qtn.TensorNetwork
) -> Tuple[Dict[Any, List[Any]], Dict[Any, Set[Any]]]:
    """
    Treat each tensor_id as a singleton block.
    Build adjacency from shared inner indices.
    """
    blocks = {tid: [tid] for tid in tn.tensor_map}
    adj    = {tid: set() for tid in tn.tensor_map}

    for ind in tn.inner_inds():
        tids = tn.ind_map[ind]
        for t1 in tids:
            for t2 in tids:
                if t1 != t2:
                    adj[t1].add(t2)

    return blocks, adj

def block_belief_propagation(
    rho_tn:         qtn.TensorNetwork,
    chi_msg:        int        = 8,
    max_iter:       int        = 50,
    tol:            float      = 1e-6,
) -> Dict[Tuple[Any,Any], MatrixProductState]:
    """
    Run Block‐BP on a doubled-ket+bra TN `rho_tn`, treating each tensor as its own block.
    Returns a dict mapping each directed edge (A→B) to its converged MPS message.

    Convergence is measured by contracting both old and new MPS → Tensors on
    the *same* physical-leg boundary, flattening, and comparing their 2-norm.
    """

    # 1) Partition into singleton blocks + build adjacency
    blocks, adj = partition_into_blocks(rho_tn)

    # 2) Precompute *only* the physical‐leg intersections for every edge
    #    (we filter on names that start with 'phys' so bonds/env legs never slip in)
    boundary_map: Dict[Tuple[Any,Any], Tuple[str,...]] = {}
    for A, neighs in adj.items():
        # gather physical legs in block A
        physA = {
            i
            for tid in blocks[A]
            for i in rho_tn.tensor_map[tid].inds
            if i.startswith("phys_")
        }
        for B in neighs:
            physB = {
                i
                for tid in blocks[B]
                for i in rho_tn.tensor_map[tid].inds
                if i.startswith("phys_")
            }
            shared = tuple(sorted(physA & physB))
            if shared:
                boundary_map[(A, B)] = shared

    # 3) Initialize random messages on exactly those nonempty boundaries
    messages = init_block_messages(boundary_map, rho_tn, chi_msg)

    # 4) Main BP loop
    for it in range(1, max_iter + 1):
        max_diff = 0.0
        new_msgs: Dict[Tuple[Any,Any], MatrixProductState] = {}

        for edge, old_mps in messages.items():
            A, B   = edge
            shared = boundary_map[edge]

            # a) update using the exact same shared tuple
            new_mps = update_block_message(
                A, B,
                blocks, adj, rho_tn,
                messages, chi_msg,
                shared,
            )

            # b) convert both old & new to dense on those *exact* legs
            old_t = _mps_to_tensor(old_mps,  shared)
            new_t = _mps_to_tensor(new_mps, shared)

            old_v = old_t.data.ravel()
            new_v = new_t.data.ravel()

            # c) sanity‐check shapes match 2**len(shared)
            if old_v.shape != new_v.shape:
                raise RuntimeError(
                    f"Edge {edge} shape mismatch: "
                    f"{old_v.shape} vs {new_v.shape} for shared={shared}"
                )

            # d) compute relative Frobenius‐norm diff
            diff = np.linalg.norm(old_v - new_v)
            norm = np.linalg.norm(new_v)
            rel  = diff / (norm + 1e-12)
            max_diff = max(max_diff, rel)

            new_msgs[edge] = new_mps

        messages = new_msgs

        if max_diff < tol:
            print(f"[BlockBP] Converged in {it} iters (Δ={max_diff:.1e})")
            break
    else:
        print(f"[BlockBP] WARNING: no convergence after {max_iter} iters (Δ={max_diff:.1e})")

    return messages


def _calculate_trace_from_messages(
        tn: qtn.TensorNetwork,
        messages: Dict[Tuple[Any, Any], MatrixProductState]
) -> np.ndarray:
    """
    Computes the total network contraction (the trace) from converged messages.
    """
    # Pick any tensor as the center for the belief calculation
    any_tid = next(iter(tn.tensor_map.keys()))
    t_center = tn.tensor_map[any_tid]

    # Build the belief network for this tensor
    belief_tn_tensors = [t_center.copy()]
    for (src, dst), mps in messages.items():
        if dst == any_tid:
            # Add all tensors from the incoming MPS message
            for i, Tm in enumerate(mps):
                belief_tn_tensors.append(Tm.copy())

    # Contract the entire belief network down to a scalar value
    return qtn.TensorNetwork(belief_tn_tensors).contract(all).data

def contract_with_block_bp(
    rho_tn:         qtn.TensorNetwork,
    ket_tags:       Set[str],
    bra_tags:       Set[str],
    cut_inds:       List[str],
    chi_msg:        int,
    max_iterations: int,
) -> qtn.Tensor:
    """
    Approximate partial contraction of `rho_tn` by Block Belief Propagation,
    returning the reduced density matrix on its physical legs.

    Args:
        rho_tn:         Full doubled ket+bra TensorNetwork.
        ket_tags:       Tags identifying the ket tensors.
        bra_tags:       Tags identifying the bra-copy tensors.
        cut_inds:       List of indices to be traced out (the environment).
        chi_msg:        Maximum MPS bond dimension for messages.
        max_iterations: Maximum BlockBP iterations.

    Returns:
        A quimb.Tensor `rho_A` with only the physical subsystem legs open.
    """
    # 1) Run BlockBP on the entire network
    msgs = block_belief_propagation(
        rho_tn,
        chi_msg=chi_msg,
        max_iter=max_iterations,
        tol=1e-6,
    )
    # trace_val = _calculate_trace_from_messages(rho_tn, msgs)

    # 2) Collect all Tensor IDs belonging to the subsystem (ket or bra)
    region_tids = [
        tid for tid, T in rho_tn.tensor_map.items()
        if set(T.tags) & (set(ket_tags) | set(bra_tags))
    ]
    if not region_tids:
        raise RuntimeError("No tensors found with the specified ket/bra tags.")

    # 3) Build a sub‐TensorNetwork of those region tensors + incoming messages
    sub = qtn.TensorNetwork()
    for tid in region_tids:
        sub.add_tensor(rho_tn.tensor_map[tid].copy(), tid)

    for (src, dst), mps in msgs.items():
        if dst in region_tids:
            for i, Tm in enumerate(mps):
                sub.add_tensor(Tm.copy(), f"msg_{src}_{dst}_{i}")

    # 4) Determine which legs are physical (exclude environment cut_inds)
    phys_legs: List[str] = []
    for tid in region_tids:
        for ind in rho_tn.tensor_map[tid].inds:
            if ind not in cut_inds:
                phys_legs.append(ind)

    # 5) Contract the sub‐network, keeping only the physical legs
    #    All other indices—including replicated environment legs—are traced out.
    rho_A = sub.contract(output_inds=tuple(phys_legs))

    return rho_A