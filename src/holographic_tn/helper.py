import re
from typing import List, Set
from collections import Counter
import numpy as np
import quimb.tensor as qtn
import opt_einsum as oe
import torch
from quimb.tensor import tensor_network_1d_compress, MatrixProductOperator, TensorNetwork
from quimb.tensor.tensor_1d_compress import _TN1D_COMPRESS_METHODS
from quimb.tensor.tensor_arbgeom_compress import tensor_network_ag_compress
import quimb.tensor as _qtn
from quimb.tensor import MatrixProductState
from scipy.sparse.linalg import svds, LinearOperator

def tensornetwork_to_mpo(rho_tn: qtn.TensorNetwork, left_inds: List[str]) -> qtn.MatrixProductOperator:
    """
    Given a TN for ρ_A with open ket‐legs `left_inds` and corresponding bra‐legs
    `ix_+` = f"{ix}_", return an MPO of length L = len(left_inds).
    """
    bra_inds = [f"{ix}_" for ix in left_inds]
    remaining = set(rho_tn.copy().nodes)  # work on a fresh copy

    mpo_tensors = []
    for site, (k, b) in enumerate(zip(left_inds, bra_inds)):
        # 1) gather all nodes touching this site’s ket or bra
        site_nodes = [n for n in remaining if k in n.inds or b in n.inds]
        if not site_nodes:
            raise ValueError(f"No nodes for site {site}, inds {k},{b}")

        # remove them so we don’t reuse
        remaining.difference_update(site_nodes)

        # 2) contract them into a single tensor
        sub_tn = qtn.TensorNetwork([n.copy() for n in site_nodes])
        scalar_node = sub_tn.contract(all).node

        # 3) identify physical vs bond legs
        phys_legs = {k, b}
        bonds = [ix for ix in scalar_node.inds if ix not in phys_legs]
        # ensure we have 0 or 2 bond legs
        if len(bonds) not in (0, 2):
            raise RuntimeError(f"Expected 0 or 2 bonds at site {site}, got {bonds}")

        # name them left/right in order
        left_bond, right_bond = (bonds + [None, None])[:2]
        ordered = [left_bond, k, b, right_bond]
        ordered = [ix for ix in ordered if ix is not None]

        # 4) reorder legs into MPO-convention and collect
        qtn.reorder(scalar_node, ordered)
        mpo_tensors.append(scalar_node)

    # 5) build the MPO
    return qtn.MPO(mpo_tensors)

def slice_contract(
    rho_tn: _qtn.TensorNetwork,
    vec: _qtn.Tensor,
    slice_idx: str = None
) -> _qtn.Tensor:
    """
    Contract rho_tn @ vec by slicing over one physical index of `vec`.
    - rho_tn     : quimb.TensorNetwork representing the operator.
    - vec        : quimb.Tensor whose inds are e.g. ['k0','k1',…].
    - slice_idx  : name of the index to slice. If None, picks the largest dim.
    Returns a new quimb.Tensor of same shape as vec.
    """

    # 1) Pick which index to slice
    if slice_idx is None:
        # choose the ind with largest dimension
        slice_idx = max(vec.inds, key=lambda ind: rho_tn.ind_size(ind))

    D = rho_tn.ind_size(slice_idx)

    # 2) Prepare accumulator (as numpy array) and record axis in vec.data
    axis = vec.inds.index(slice_idx)
    out_shape = list(vec.shape)
    out_data = np.zeros(out_shape, dtype=vec.data.dtype)

    # 3) Build remapping for contraction
    right_inds = [f"{ix}_" for ix in vec.inds]
    remap = dict(zip(vec.inds, right_inds))

    # 4) Loop over slices
    for k in range(D):
        # slice off the big index from vec
        # --> data_k has one fewer dimension
        data_k = np.take(vec.data, indices=k, axis=axis)
        inds_k  = [i for i in vec.inds if i != slice_idx]
        v_k     = _qtn.Tensor(data_k, inds=inds_k)

        # reindex to match rho_tn's "bra" legs
        v_k_rhs = v_k.copy().reindex(remap)

        # full small contraction
        res_k = (rho_tn @ v_k_rhs).squeeze()

        # res_k.data has shape = inds_k; re-insert slice axis at position=axis
        out_data[(slice(None),)*axis + (k,)] = res_k.data

    # 5) wrap back into a qtn.Tensor
    return _qtn.Tensor(out_data, inds=vec.inds)

def ag_compress_to_mpo(
    rho_tn: qtn.TensorNetwork,
    left_inds: List[str],
    max_bond: int,
    method: str = 'superorthogonal'
) -> qtn.MatrixProductOperator:
    """
    Compress an arbitrary‐geometry density TN into a boundary MPO that acts
    only on `left_inds` and their bra counterparts (e.g. 'k0', 'k1', … and 'k0_', 'k1_', …).

    Steps:
      1. Fuse the original TN per boundary site: collapse all tensors touching
         each physical index k{i} / k{i}_ into one site‐block.
      2. Tag each blocked tensor uniquely for the AG compressor.
      3. Run the arbitrary‐geometry compressor to truncate every bond ≤ max_bond.
      4. View the compressed TN as a MatrixProductOperator on your boundary indices.

    Args:
      rho_tn:      TensorNetwork for ρₐ, uncontracted.
      left_inds:   List of physical “ket” indices, e.g. ['k0','k1',…,'kL-1'].
      max_bond:    Maximum bond dimension for compression.
      method:      One of the AG‐compress methods ('su'/'superorthogonal',
                   'l2bp', 'local-early', etc.).

    Returns:
      A MatrixProductOperator `rho_mpo` with low bond dimensions,
      ready for `.apply(v, compress=True, max_bond=…)`.
    """

    # Step 1: fuse each boundary site (ket + bra) into a single block
    for i, k in enumerate(left_inds):
        tag = f"site{i}"
        for T in rho_tn.tensors:
            if (k in T.inds) or (f"{k}_" in T.inds):
                T.add_tag(tag)
        # absorb all tensors carrying that tag into one
        rho_tn ^= tag
        rho_tn.fuse_multibonds_()

    # Step 2: tag each fused block uniquely and register site_tags
    site_tags_ag = [f"ag_site{i}" for i in range(len(rho_tn.tensors))]
    for tag, T in zip(site_tags_ag, rho_tn.tensors):
        T.add_tag(tag)
    rho_tn.site_tags = site_tags_ag

    # Step 3: AG‐compress to cap every bond to ≤ max_bond
    tn_comp = tensor_network_ag_compress(
        rho_tn,
        max_bond=max_bond,
        method=method,
        cutoff=0.0,
        inplace=True
    )

    # Step 4: within each site‐block, reindex (fuse) all phys_* → k{i}
    for T in tn_comp.tensors:
        # identify which site this tensor belongs to
        ag_tag = next(tag for tag in T.tags if tag.startswith('ag_site'))
        i = int(ag_tag.split('ag_site', 1)[1])

        # collect lower (ket) phys legs
        lower_phys = [ix for ix in T.inds
                      if ix.startswith('phys_') and not ix.endswith('_')]
        if lower_phys:
            # map every old phys leg onto the same new index name
            mapping = {old: f'k{i}' for old in lower_phys}
            T.reindex(mapping, inplace=True)

        # collect upper (bra) phys legs
        upper_phys = [ix for ix in T.inds
                      if ix.startswith('phys_') and ix.endswith('_')]
        if upper_phys:
            mapping = {old: f'k{i}_' for old in upper_phys}
            T.reindex(mapping, inplace=True)

    # Step 5: reinterpret as a 1D MPO, explicitly giving all naming patterns
    rho_mpo = tn_comp.view_as_(
        MatrixProductOperator,
        site_tag_id='ag_site{}',
        lower_ind_id='k{}',
        upper_ind_id='k{}_',
        cyclic=False,
        L=len(left_inds)
    )

    return rho_mpo

def mps_norm_sq_from_apply(w: TensorNetwork, ket_inds: list[str]) -> float:
    """
    Given w = rho_mpo.apply(v), extract only those tensors carrying
    a ket‐leg in `ket_inds`, sort them by the integer in the physical
    leg name, and do a left→right sweep to compute <w|w>—all without
    ever forming hyper‐edges.
    """
    # 0) collapse *every* multi‐bond in the full network
    w.fuse_multibonds_()

    # 1) pick the tensors that carry ANY of our physical k‐legs
    mps_sites = [T for T in w.tensors if any(ix in ket_inds for ix in T.inds)]
    if not mps_sites:
        raise RuntimeError("No MPS sites found containing any ket_inds")

    # 2) sort by the integer suffix of the physical leg (e.g. 'k10'→10)
    def site_index(T):
        # find the unique phys leg among ket_inds
        phys_list = []
        for ix in T.inds:
            if ix in ket_inds and ix not in phys_list:
                phys_list.append(ix)
        if len(phys_list) != 1:
            raise RuntimeError(f"Expected exactly 1 phys-leg ∈ {ket_inds}, got {phys_list}")
        phys = phys_list[0]
        return int(re.match(r"[^\d]*(\d+)", phys).group(1))

    mps_sites.sort(key=site_index)

    # 3) left→right sweep
    env = None
    for i, T in enumerate(mps_sites):
        # a) extract the one physical leg (deduped)
        phys_list = []
        for ix in T.inds:
            if ix in ket_inds and ix not in phys_list:
                phys_list.append(ix)
        phys = phys_list[0]
        phys_bra = phys + "_"

        # b) build bra tensor
        Tb = T.conj().reindex({phys: phys_bra}, inplace=False)

        # c) dedupe bond legs
        bond_list = []
        for ix in T.inds:
            if ix != phys and ix not in bond_list:
                bond_list.append(ix)

        # d) figure out left/right for this sub‐chain
        if len(bond_list) == 2:
            left_b, right_b = bond_list
        elif len(bond_list) == 1:
            # first site has only right, last has only left
            left_b, right_b = (None, bond_list[0]) if env is None else (bond_list[0], None)
        else:
            raise RuntimeError(f"site with deduped bonds {bond_list!r} is unexpected")

        # e) contract
        if env is None:
            # first site: contract Tb & T, keep right_b
            tn0 = TensorNetwork([Tb, T])
            env = tn0.contract(all, output_inds=[right_b] if right_b else [])
        else:
            # middle/last: env→Tb, then Tb→T
            tn1 = TensorNetwork([env, Tb])
            env1 = tn1.contract(all, output_inds=[phys, right_b] if right_b else [phys])
            tn2 = TensorNetwork([env1, T])
            env = tn2.contract(all, output_inds=[right_b] if right_b else [])

    # env is now a 0‐leg TN = <w|w>
    return float(env)


def apply_operator_outside_in(
        rho_tn: qtn.TensorNetwork,
        v_mps: qtn.MatrixProductState,
        phys_inds: List[str],
        max_bond: int,
) -> qtn.Tensor:
    """
    Approximates `rho @ v` using a memory-bounded "outside-in" sweep.
    This is the definitive, API-corrected version.
    """
    # Start the boundary with a scalar 1.0 tensor
    boundary_tensor = qtn.Tensor(data=1.0 + 0.j, inds=())

    print("  Applying operator with 'outside-in' SVD sweep...")
    for i, p_ind in enumerate(phys_inds):
        print(f"    - Sweeping site {i + 1}/{len(phys_inds)}...", end='\r')

        # 1. Identify the three tensors for this site
        ket_tensor = rho_tn.tensor_map[next(iter(rho_tn.ind_map[p_ind]))]
        bra_tensor = rho_tn.tensor_map[next(iter(rho_tn.ind_map[f"{p_ind}_"]))]
        vec_tensor = v_mps.tensors[i]

        # 2. Contract the current boundary with this site's tensors
        # 2a. First, contract the small, local tensors together.
        local_update = qtn.tensor_contract(ket_tensor, bra_tensor, vec_tensor)

        # 2b. Then, apply this small update to the larger boundary tensor.
        #     This avoids large intermediate tensors.
        new_boundary = qtn.tensor_contract(boundary_tensor, local_update)

        # 3. If not the last site, compress the new boundary
        if i < len(phys_inds) - 1:
            outer_inds = new_boundary.inds

            # Identify indices connecting to the "future" part of the network
            bond_inds = [ix for ix in outer_inds if
                         (ix in rho_tn.ind_map and ix not in (p_ind, f"{p_ind}_"))]

            # Bipartition the tensor into "past" and "future"
            past_inds = list(set(outer_inds) - set(bond_inds))

            # Split the tensor and truncate the bond to max_bond.
            # This returns two tensors, with singular values absorbed.
            past_tensor, future_tensor = new_boundary.split(
                left_inds=past_inds,
                right_inds=bond_inds,
                max_bond=max_bond,
                get='tensors'
            )

            # The 'future_tensor' becomes the boundary for the next step
            boundary_tensor = future_tensor
        else:
            boundary_tensor = new_boundary

    print("\n  Sweep complete.")
    # The final boundary is the resulting vector as a single tensor
    return boundary_tensor


def contract_tn_greedy_manual(
        tn: qtn.TensorNetwork,
        output_inds: List[str],
        mem_limit_gb: int = 16
) -> qtn.Tensor:
    """
    Contracts a TensorNetwork using a manually implemented, memory-aware
    greedy algorithm. This is built to be compatible with older library versions.
    """
    tn_contract = tn.copy()
    mem_limit_elements = int(mem_limit_gb * 1e9 / 16)

    print(f"    - Contracting with manual greedy contractor (limit < {mem_limit_gb} GB)...")

    while tn_contract.num_tensors > 1:
        costs = {}
        for bond in tn_contract.inner_inds():
            tids = tn_contract.ind_map[bond]
            if len(tids) != 2: continue

            t1, t2 = (tn_contract.tensor_map[tid] for tid in tids)
            output_inds_step = set(t1.inds) ^ set(t2.inds)

            cost = 1
            for out_ix in output_inds_step:
                cost *= tn_contract.ind_size(out_ix)
            costs[bond] = cost

        if not costs: break

        cheapest_bond = min(costs, key=costs.get)

        if costs[cheapest_bond] > mem_limit_elements:
            peak_mem_gb = costs[cheapest_bond] * 16 / 1e9
            raise MemoryError(
                f"No further contractions possible within the {mem_limit_gb} GB "
                f"limit. Next cheapest step requires {peak_mem_gb:.2f} GB."
            )

        tid1, tid2 = tn_contract.ind_map[cheapest_bond]
        tag1, tag2 = f"temp_contract_{tid1}", f"temp_contract_{tid2}"
        tn_contract.tensor_map[tid1].add_tag(tag1)
        tn_contract.tensor_map[tid2].add_tag(tag2)
        tn_contract.contract_tags([tag1, tag2], inplace=True)

    return tn_contract.contract(all, output_inds=output_inds)


def apply_operator_manual(
        rho_tn: qtn.TensorNetwork,
        v_mps: qtn.MatrixProductState,
        phys_inds: List[str],
        mem_limit_gb: int = 16
) -> qtn.Tensor:
    """
    Applies the operator `rho` to `v` using the robust, manual
    greedy contractor.
    """
    remap = {ix: f"{ix}_" for ix in phys_inds}
    v_for_contract = v_mps.copy().reindex(remap)
    output_inds = v_mps.outer_inds()
    combined_tn = qtn.TensorNetwork(rho_tn.tensors + v_for_contract.tensors)

    w_tensor = contract_tn_greedy_manual(
        combined_tn,
        output_inds=output_inds,
        mem_limit_gb=mem_limit_gb
    )

    return w_tensor

def rho_apply_with_memcap(rho_tn: qtn.TensorNetwork,
                          vec: qtn.Tensor,
                          remap: dict,
                          memory_limit_bytes: int):
    """
    Contract rho_tn @ vec under a hard memory cap.
    - memory_limit_bytes: the max bytes any intermediate may hold.
    Returns a qtn.Tensor with the same inds & shape as `vec`.
    """
    # 1) prepare the "bra" copy of the vector
    v_rhs = vec.copy().reindex(remap)

    # 2) extract the opt_einsum inputs
    #    this gives us (subscripts, operands, output_subscripts)
    subs, ops, out_subs = rho_tn._opt_einsum_inputs(v_rhs)

    # 3) find a path that respects memory_limit
    #    note: memory_limit is in *elements* not bytes
    bytes_per_elem = vec.data.dtype.itemsize
    elem_limit    = memory_limit_bytes // bytes_per_elem

    path, info = oe.contract_path(
        *subs, *ops,
        output_subscripts=out_subs,
        optimize='auto-hq',
        memory_limit=elem_limit
    )

    # 4) actually do the contraction using that path
    result_data = oe.contract(
        *subs, *ops,
        output_subscripts=out_subs,
        optimize=path
    )

    # 5) wrap back into a quimb.Tensor
    return qtn.Tensor(result_data, inds=vec.inds)

def random_mps(phys_dims,
               bond_dims,
               backend='numpy',
               dtype=None,
               device=None):
    """
    Build a random MPS with shapes
       A[0]: (1,    d0,    χ1)
       A[1]: (χ1,   d1,    χ2)
       …
       A[n-2]: (χ[n-2], d[n-2], χ[n-1])
       A[n-1]: (χ[n-1], d[n-1], 1)
    without ever materializing the full tensor.
    """
    # Helper for random arrays
    if backend == 'torch':
        dtype = dtype or torch.complex64
        rand = lambda *shape: torch.randn(*shape, dtype=dtype, device=device)
    else:
        dtype = dtype or np.complex128
        rand = lambda *shape: np.random.randn(*shape).astype(dtype)

    nsites = len(phys_dims)
    cores = []
    χ_prev = 1

    # 1) Sample raw cores
    for k in range(nsites):
        χ_next = bond_dims[k] if k < len(bond_dims) else 1
        shape = (χ_prev, phys_dims[k], χ_next)
        cores.append(rand(*shape))
        χ_prev = χ_next

    # 2) Left-canonicalize via QR to keep everything orthonormal
    for k in range(nsites - 1):
        A = cores[k]
        χ_prev, d, χ_next = A.shape

        # merge left indices => matrix of shape (χ_prev*d) × χ_next
        M = A.reshape(χ_prev * d, χ_next)

        # QR-decompose
        if backend == 'torch':
            Q, R = torch.linalg.qr(M, mode='reduced')
        else:
            Q, R = np.linalg.qr(M, mode='reduced')

        # reshape Q back into tensor core
        χ_mid = Q.shape[1]
        cores[k] = Q.reshape(χ_prev, d, χ_mid)

        # absorb R into the next core's left leg
        B = cores[k+1]
        # B.shape = (χ_next, d_next, χ_next2)
        B_flat = B.reshape(χ_next, -1)      # (old-χ_next) × (d_next*χ_next2)
        B_new  = R @ B_flat                # (χ_mid) × (d_next*χ_next2)
        cores[k+1] = B_new.reshape(χ_mid, B.shape[1], B.shape[2])

    # Done: cores is a left-canonical MPS
    # Wrap with your favorite MPS class or return as raw list

    # 4) package into an MPS
    #    Quimb will pick up the bond indices ("b1","b2",…) automatically
    quimb_mps = MatrixProductState(cores)

    return quimb_mps


def compress_tensor(
    T: qtn.Tensor,
    left_inds,
    right_inds,
    max_bond=None,
    cutoff=None,
    absorb='right'
):
    """
    Split a qtn.Tensor into two cores along (left_inds | right_inds),
    truncating the new bond to ≤ max_bond and/or dropping singular values
    < cutoff.

    Parameters
    ----------
    T : quimb.tensor.Tensor
        The tensor to compress.
    left_inds : sequence of str
        Names of the indices on the “left” side.
    right_inds : sequence of str
        Names of the indices on the “right” side.
    max_bond : int, optional
        Maximum size of the new bond index.
    cutoff : float, optional
        Threshold below which singular values are discarded.
    absorb : {'left','right'}
        Whether to absorb the R (or L) factor into the right or left core.

    Returns
    -------
    T_left : quimb.tensor.Tensor
        Left core with indices = left_inds + [new_bond].
    T_right : quimb.tensor.Tensor
        Right core with indices = [new_bond] + right_inds.
    """
    # Copy to avoid mutating the original tensor
    T_copy = T.copy()

    # Perform the split with truncation
    T_left, T_right = T_copy.split(
        left_inds  = tuple(left_inds),
        right_inds = tuple(right_inds),
        max_bond   = max_bond,
        cutoff     = cutoff,
        absorb     = absorb
    )

    return T_left, T_right

def apply_split_op_auto(rho_L, rho_R, v):
    """
    Given two operator cores rho_L, rho_R linked by a single bond index
    and a vector-tensor v, figure out which indices to contract and
    which to keep as outputs, then return w = (rho_L ∘ rho_R) · v.
    """

    # 1) figure out which index is the bond
    common = set(rho_L.inds) & set(rho_R.inds)
    assert len(common) == 1, "Expect exactly one linking index"
    bond, = common

    # 2) collect the vector’s indices
    v_inds = set(v.inds)

    # 3) classify legs on each core
    left_in   = tuple(i for i in rho_L.inds if i in v_inds)
    left_out  = tuple(i for i in rho_L.inds
                      if i not in v_inds and i != bond)
    right_in  = tuple(i for i in rho_R.inds if i in v_inds)
    right_out = tuple(i for i in rho_R.inds
                      if i not in v_inds and i != bond)

    # 4) build the tiny TN and contract everything except the outs
    TN = qtn.TensorNetwork([rho_L, rho_R, v])
    keep = left_out + right_out
    w = TN.contract(keep)

    return w

def compress_tn_new(
    tn: TensorNetwork,
    max_bond: int = None,
    cutoff: float = None,
    absorb: str = 'right'
) -> TensorNetwork:
    """
    Return a NEW TensorNetwork whose internal bonds have been
    SVD-truncated to ≤ max_bond and/or with SVs < cutoff dropped.

    Parameters
    ----------
    tn : TensorNetwork
        The original operator network to compress.
    max_bond : int, optional
        Maximum allowed bond dimension on each cut.
    cutoff : float, optional
        Discard singular values below this threshold.
    absorb : {'left','right'}
        Whether to absorb R (or L) into the right (or left) core.

    Returns
    -------
    new_tn : TensorNetwork
        A fresh TN built from the final, truncated cores.
    """
    # 1) Copy every original core
    orig_cores = list(tn.tensors)
    copy_cores = [T.copy() for T in orig_cores]
    orig_to_copy = dict(zip(orig_cores, copy_cores))

    # 2) Sweep over each bond name
    for bond, owners in tn.ind_map.items():
        # only consider true internal bonds
        real_owners = [T for T in orig_cores if bond in T.inds]
        if len(real_owners) != 2:
            continue
        A_orig, B_orig = real_owners

        # get our working copies
        A = orig_to_copy[A_orig]
        B = orig_to_copy[B_orig]

        # contract *only* that one bond, leaving all other legs open
        # (A & B) is a 2-node TN; .contract(bond) fuses them into a single Tensor
        AB = (A & B).contract(bond)

        # now AB is a qtn.Tensor; use its .inds
        left_inds = tuple(i for i in AB.inds if i in A.inds and i != bond)
        right_inds = tuple(i for i in AB.inds if i in B.inds and i != bond)

        # SVD-split + truncate
        A_new, B_new = AB.split(
            left_inds=left_inds,
            right_inds=right_inds,
            max_bond=max_bond,
            cutoff=cutoff,
            absorb=absorb,
        )

        # update for downstream sweeps
        orig_to_copy[A_orig] = A_new
        orig_to_copy[B_orig] = B_new

    # 3) Re-assemble a brand-new TN from all final cores
    final_cores = [orig_to_copy[T] for T in orig_cores]
    return TensorNetwork(*final_cores)

# -----------------------------------------------------------------------------
# 1) A memory-safe SVD split via iterative svds
# -----------------------------------------------------------------------------
def truncated_split_via_svds(
        A: qtn.Tensor,
        B: qtn.Tensor,
        bond: str,
        left_inds: tuple,
        right_inds: tuple,
        max_bond: int = None,
        cutoff: float = None,
        absorb: str = 'right'
):
    # record their dimensions
    left_dims = [A.shape[A.inds.index(i)] for i in left_inds]
    right_dims = [B.shape[B.inds.index(i)] for i in right_inds]
    M_rows = int(np.prod(left_dims))
    M_cols = int(np.prod(right_dims))

    # find the shared axis in the raw data arrays
    axA = A.inds.index(bond)
    axB = B.inds.index(bond)

    # build matvec via two tiny TN contracts
    def matvec(v):
        # 1) wrap v in the right‐legs shape
        v_ten = qtn.Tensor(data=v.reshape(*right_dims),
                       inds=right_inds)

        # 2) contract B + v_ten, keep only the bond
        TN1 = TensorNetwork([B, v_ten])
        Xbond_t = TN1.contract(output_inds=(bond,))

        # 3) contract A + that bond‐tensor, keep left_inds
        TN2 = TensorNetwork([A, Xbond_t])
        X_ten = TN2.contract(output_inds=left_inds)

        return X_ten.data.reshape(M_rows)

    # adjoint matvec
    def rmatvec(u):
        u_ten = qtn.Tensor(data=u.reshape(*left_dims),
                       inds=left_inds)

        TN1 = TensorNetwork([A, u_ten])
        Ubond_t = TN1.contract(output_inds=(bond,))

        TN2 = TensorNetwork([B, Ubond_t])
        Y_ten = TN2.contract(output_inds=right_inds)

        return Y_ten.data.reshape(M_cols).conj()

    linop = LinearOperator((M_rows, M_cols), matvec, rmatvec)
    k = max_bond or min(M_rows, M_cols)
    U, s, Vt = svds(linop, k=k, tol=1e-6 or 0.0, which='LM')

    # absorb singular values
    if absorb == 'right':
        Vt = np.diag(s) @ Vt
    else:
        U = U @ np.diag(s)

    print(f" bond {bond}: kept {len(s)}/{M_rows if M_rows < M_cols else M_cols} singulars, "
          f"v_min={s.min():.2e}, v_max={s.max():.2e}")

    # s_tag = f"s_{bond}"
    s_tag = bond

    # wrap back into two cores, with a single new 's' leg
    A_new = qtn.Tensor(
        data=U.reshape(*left_dims, -1),
        inds=left_inds + (s_tag,),
        tags=A.tags.copy()
    )
    B_new = qtn.Tensor(
        data=Vt.reshape(-1, *right_dims),
        inds=(s_tag,) + right_inds,
        tags=B.tags.copy()
    )

    return A_new, B_new

# -----------------------------------------------------------------------------
# 2) compress_tn_svds: same bond-sweep logic as compress_tn_new,
#    but calling truncated_split_via_svds instead of .contract/.split
# -----------------------------------------------------------------------------
def compress_tn_svds(
    tn: TensorNetwork,
    max_bond: int   = None,
    cutoff: float   = None,
    absorb: str     = 'right'
) -> TensorNetwork:
    orig_cores = list(tn.tensors)
    copy_map   = {T: T.copy() for T in orig_cores}

    i = 0

    for bond, owners in tn.ind_map.items():
        # only bonds shared by exactly two tensors
        real = [T for T in orig_cores if bond in T.inds]
        if len(real) != 2:
            continue
        A_orig, B_orig = real

        # fetch the in-flight copies
        A_copy = copy_map[A_orig]
        B_copy = copy_map[B_orig]

        # skip if that bond was already removed
        #if bond not in A_copy.inds or bond not in B_copy.inds:
        #    continue

        # ←—— HERE’S THE FIX ——→
        # compute left/right legs off the *copies* (not the originals)
        left_inds  = tuple(i for i in A_copy.inds if i != bond)
        right_inds = tuple(i for i in B_copy.inds if i != bond)

        # perform the low-memory SVD split
        A_new, B_new = truncated_split_via_svds(
            A_copy, B_copy, bond=bond,
            left_inds=left_inds,
            right_inds=right_inds,
            max_bond=max_bond,
            cutoff=cutoff,
            absorb=absorb
        )

        # overwrite for downstream bonds
        copy_map[A_orig] = A_new
        copy_map[B_orig] = B_new

        i += 1
        print(f"Iteration {i} / {len(tn.ind_map.items())}")

    # re-assemble a brand new TN from all final cores
    final_cores = [copy_map[T] for T in orig_cores]
    compressed_tn = TensorNetwork(final_cores)

    for bond, owners in compressed_tn.ind_map.items():
        if len(owners) == 2:
            print(f"bond {bond:10s} → dim {compressed_tn.ind_size(bond):4d}")

    for b, owners in compressed_tn.ind_map.items():
        if len(owners) == 2 and compressed_tn.ind_size(b) == 1:
            compressed_tn = compressed_tn.contract(b)

    return compressed_tn

def add_uniform_site_tags(tn: TensorNetwork) -> None:
    """
    Modifies tn in place: Assigns each core a SITE<i> tag based on its position in tn.tensors.
    """

    #    We assume tn.tensors is already in the correct physical order.
    for i, T in enumerate(tn.tensors):
        # remove any existing SITE* tags
        old_site_tags = [tag for tag in T.tags if str(tag).startswith('SITE')]
        for tag in old_site_tags:
            T.tags.remove(tag)

        # add the new uniform site tag
        T.tags.add(f"SITE{i}")

def build_phys_index_map(tn: TensorNetwork):
    index_map = {}
    bra_pat = re.compile(r'^(phys_\d+_\d+)_$')
    ket_pat = re.compile(r'^(phys_\d+_\d+)$')
    for T in tn.tensors:
        for ind in T.inds:
            m_b = bra_pat.match(ind)
            m_k = ket_pat.match(ind)
            if m_b:
                index_map[ind] = f"{m_b.group(1)}_b"
            elif m_k:
                index_map[ind] = f"{m_k.group(1)}_k"
    return index_map

def stabilize_bond_names(mps: MatrixProductState):
    """
    In-place: rename every *physical* bond between SITE i and SITE i+1
    to 'bond0', 'bond1', …, in ascending order.  Leaves other bonds untouched.
    """

    # 1) Build a map: tensor_id → site_number (only for SITE-tagged cores)
    tid2site = {}
    for tid, T in mps.tensor_map.items():
        for tag in T.tags:
            if isinstance(tag, str) and tag.startswith('SITE'):
                tid2site[tid] = int(tag[4:])  # SITE<i> → i
                break

    # 2) Collect only the bonds that connect two SITE cores
    phys_bonds = []
    for bond in mps.inner_inds():
        tids = mps.ind_map[bond]  # e.g. frozenset({3,4})
        # keep only if *both* ends have SITE tags
        if all(tid in tid2site for tid in tids):
            phys_bonds.append(bond)

    # 3) Sort those bonds by (min_site, max_site)
    def bond_sites(bond):
        tids = mps.ind_map[bond]
        sites = sorted(tid2site[tid] for tid in tids)
        return sites  # [i, i+1]

    phys_bonds_sorted = sorted(phys_bonds, key=bond_sites)

    # 4) Build remapping: old_bond → 'bond0', 'bond1', …
    remap = {old: f"bond{new_i}"
             for new_i, old in enumerate(phys_bonds_sorted)}

    # 5) Rename in-place (only the specified bonds change)
    mps.reindex_(remap)