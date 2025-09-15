# src/holographic_tn/physics.py
import warnings

import numpy as np
import quimb.tensor as qtn
import torch
from typing import List, Dict, Set, Tuple
from collections import deque

from .config.entropy_method import ExactConfig, EntropyMethod
from .experimental.bp.block_bp import contract_with_block_bp
from .experimental.bp.config import BBPConfig
from .numerics.kpm.kpm import _compute_chebyshev_moments_ste, _calculate_entropy_kpm
from .geometry import HyperbolicBuilding
from .numerics.kpm.config import KPMConfig
from .tensor import get_perfect_tensor_of_rank


def build_network_from_building(
        building: HyperbolicBuilding,
        bond_dim: int = 2,
        compute_mode: str = 'cpu'  # <-- New parameter
) -> qtn.TensorNetwork:
    """Constructs a holographic tensor network with a selectable compute backend.

    Args:
        building: The fully constructed HyperbolicBuilding object.
        bond_dim: The dimension of each tensor index (leg).
        compute_mode (str): The backend for tensor operations.
            - 'gpu': Uses PyTorch with the best available accelerator (MPS/CUDA)
              and single precision (complex64).
            - 'cpu': Uses NumPy with double precision (complex128) for maximum
              accuracy.

    Returns:
        A quimb.tensor.TensorNetwork object representing the holographic state.
    """
    print(f"▶️ Creating holographic tensor network in '{compute_mode}' mode...")

    # --- Setup Backend Data ---
    tensor_data_np = get_perfect_tensor_of_rank(building.p, bond_dim)
    if compute_mode == 'gpu':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("  Using Apple Metal (MPS) backend for GPU acceleration.")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print("  Using NVIDIA CUDA backend for GPU acceleration.")
        else:
            device = torch.device('cpu')
            print("  GPU not found, falling back to CPU with torch.")

        # Use single precision (complex64) for GPU compatibility
        tensor_backend_data = torch.from_numpy(tensor_data_np).to(torch.complex64).to(device)

    elif compute_mode == 'cpu':
        print("  Using NumPy backend for CPU precision.")
        # Use double precision (complex128) for CPU accuracy
        tensor_backend_data = tensor_data_np.astype(np.complex128)

    tensors = []
    # --- FIX: Create a unique index for each edge in the graph upfront ---
    edge_to_idx_map = {
        tuple(sorted(edge)): f'e_{i}'
        for i, edge in enumerate(building.simplicial_complex.edges())
    }

    faces = [n for n in building.simplicial_complex.nodes()]

    for face_id in faces:
        rank = building.p
        inds = [None] * rank

        # Get neighbors from the graph
        neighbors = list(building.simplicial_complex.neighbors(face_id))

        # Assign contracted indices based on the edge map
        for i, neighbor_id in enumerate(neighbors):
            edge = tuple(sorted((face_id, neighbor_id)))
            # Naively assign the connection to the next available slot.
            # This is sufficient for defining connectivity.
            if i < rank:
                inds[i] = edge_to_idx_map[edge]

        # Assign physical (dangling) indices to the remaining slots
        dangling_counter = 0
        for i in range(rank):
            if inds[i] is None:
                # Use the tuple ID for a unique name, e.g., (2, 5) -> phys_5_0
                face_num = face_id[1]
                inds[i] = f'phys_{face_num}_{dangling_counter}'
                dangling_counter += 1

        # This will now create either a Torch-backed or Numpy-backed tensor
        tensors.append(qtn.Tensor(tensor_backend_data, inds=inds, tags={str(face_id)}))

    tn = qtn.TensorNetwork(tensors)

    print(f"  Placed {len(tensors)} tensors.")
    print("✅ Tensor network and ID map construction complete.")
    return tn

def _get_face_id_from_tag(tag: str) -> tuple:
    """Helper to convert quimb tag string back to a face_id tuple."""
    return tuple(map(int, tag.strip('()').split(',')))


def _find_boundary_endpoints(
        tn: qtn.TensorNetwork,
        building: HyperbolicBuilding,  # Add building as an argument
        boundary_region_inds: List[str]
) -> Tuple[tuple, tuple]:
    """
    Finds the two faces in a boundary region that are maximally separated.

    This function is robust to the ordering of indices in the input list. It
    identifies all unique faces associated with the boundary indices and then
    computes the pairwise distance between them to find the two that are
    furthest apart, which are the true geometric endpoints of the region.
    """
    if not boundary_region_inds:
        raise ValueError("No boundary indices provided")

    # 1. Find all unique faces that are part of the boundary region
    boundary_faces = {
        _get_face_id_from_tag(tag)
        for ind in boundary_region_inds
        for tag in tn.tensor_map[next(iter(tn.ind_map[ind]))].tags
        if tag.startswith('(')
    }

    if len(boundary_faces) <= 1:
        face = boundary_faces.pop()
        return face, face

    # 2. Find the pair of faces in this set with the largest geodesic distance
    max_dist = -1.0
    start_face, end_face = None, None

    face_list = list(boundary_faces)
    # Check all unique pairs of faces
    for i in range(len(face_list)):
        for j in range(i + 1, len(face_list)):
            f1, f2 = face_list[i], face_list[j]
            # Use the building object to compute the distance
            dist = building._get_path_distance(f1, f2)
            if dist > max_dist:
                max_dist = dist
                start_face, end_face = f1, f2

    return start_face, end_face


def _partition_network(
        tn: qtn.TensorNetwork,
        building: HyperbolicBuilding,
        boundary_region_faces: Set[tuple],
        geodesic_path_faces: List[tuple],
) -> Tuple[Set[str], List[str]]:
    """Partitions the tensor network into a 'ket' and an 'environment'.

    This function uses the minimal surface (geodesic) as a wall to divide
    the tensor network. The "ket" is the sub-network corresponding to the
    quantum state of the region bounded by the user-defined boundary `A`
    and the geodesic `γ_A`.

    The algorithm works as follows:
    1.  It performs a Breadth-First Search (BFS) on the building's dual
        graph, starting from the faces of the boundary region `A`.
    2.  The search is constrained and cannot cross the faces that form the
        geodesic path.
    3.  The 'ket' is defined as all faces visited by the BFS, plus the
        faces of the geodesic wall itself.
    4.  A special case is handled: if all boundary faces lie on the
        geodesic, the ket is defined as just the boundary faces.

    Args:
        tn: The full tensor network object.
        building: The HyperbolicBuilding, used for its graph topology.
        boundary_region_faces: A set of face_ids for the boundary region A.
        geodesic_path_faces: A list of face_ids forming the minimal surface.

    Returns:
        A tuple containing:
        - ket_tags (Set[str]): A set of quimb tensor tags identifying all
          tensors in the ket sub-network.
        - cut_inds (List[str]): A list of string indices that are on the
          boundary of the ket (the "cut").
    """
    geodesic_set = set(geodesic_path_faces)

    bfs_start_nodes = list(boundary_region_faces - geodesic_set)
    queue = deque(bfs_start_nodes)
    visited = boundary_region_faces.union(geodesic_set)
    inside_nodes_by_id = set(bfs_start_nodes)

    while queue:
        current_face = queue.popleft()
        for neighbor_face in building._get_face_neighbors(current_face):
            if neighbor_face not in visited:
                visited.add(neighbor_face)
                inside_nodes_by_id.add(neighbor_face)
                queue.append(neighbor_face)

    ket_nodes_by_id = inside_nodes_by_id.union(geodesic_set)

    # --- DEBUGGING PRINT STATEMENT ---
    # Add this block to see exactly what the function is computing
    print("\n--- DEBUG: Inside _partition_network ---")
    print(f"Boundary faces: {boundary_region_faces}")
    print(f"Geodesic path faces: {set(geodesic_path_faces)}")
    print(f"Final ket faces: {ket_nodes_by_id}")
    print("-------------------------------------\n")
    # --- END OF DEBUGGING BLOCK ---

    ket_tags = {str(face_id) for face_id in ket_nodes_by_id}
    ket_tn = tn.select(ket_tags, which='any')
    cut_inds = list(ket_tn.outer_inds())

    return ket_tags, cut_inds

def calculate_rt_entropy(
        building: HyperbolicBuilding,
        tn: qtn.TensorNetwork,
        boundary_region_inds: List[str],
        config: EntropyMethod = ExactConfig(),
) -> Dict:
    """Executes the full Ryu-Takayanagi calculation workflow.

    Args:
        building: The fully constructed HyperbolicBuilding object.
        tn: The quimb TensorNetwork object representing the full quantum state.
        boundary_region_inds: An ordered list of string-based indices.
        config: A configuration object whose type determines the calculation
            method.
            - `isinstance(config, KPMConfig)`: Uses the memory-efficient
              Kernel Polynomial Method. Recommended for large systems. 🧠
            - `isinstance(config, BBPConfig)`: (EXPERIMENTAL) Uses an
              approximate contraction (Block Belief Propagation) to form a
              dense matrix, then diagonalizes. 🧪
            - `isinstance(config, ExactConfig)`: Performs a full, exact
              contraction to a dense matrix before diagonalizing. Only
              feasible for very small systems. 💪

    Returns:
        A dictionary containing the final results.
    """
    # Steps 1 & 2: Find the geodesic and partition the tensor network
    print("▶️ Starting Ryu–Takayanagi calculation...")
    print(f"  Using entropy calculation method: '{config.__class__.__name__}'")
    print("  Step 1/4: Finding minimal surface (geodesic) in the bulk...")
    start, end = _find_boundary_endpoints(tn, building, boundary_region_inds)
    geodesic_path = building.find_geodesic_a_star(start, end)
    if not geodesic_path:
        raise RuntimeError("Failed to find a bulk geodesic for the specified boundary region.")
    discrete_geodesic_length = building._get_path_distance(start, end)
    geometric_geodesic_length = building._get_geometric_path_length(start, end)
    boundary_faces = {_get_face_id_from_tag(tag) for ind in boundary_region_inds for tag in
                      tn.tensor_map[next(iter(tn.ind_map[ind]))].tags if tag.startswith('(')}

    print("  Step 2/4: Partitioning the tensor network...")
    ket_tags, cut_inds = _partition_network(tn, building, boundary_faces, geodesic_path)
    ket_tn = tn.select(ket_tags, which='any')

    # Step 3: Build the network for the density matrix
    print("  Step 3/4: Building the density matrix network ρ_A...")
    all_ket_outer_inds = set(ket_tn.outer_inds())
    phys_inds = set(all_ket_outer_inds).intersection(set(boundary_region_inds))

    # --- CORRECTED: Index Remapping ---
    # Only the physical indices, which will remain open as the legs of the
    # final density matrix, should be re-indexed for the 'bra' network.
    # All internal and cut indices must keep their original names so they are
    # correctly contracted when the ket and bra are combined.
    bra_remap = {ix: f'{ix}_' for ix in phys_inds}

    # --- IMPROVED: Streamlined Normalization (for KPM) ---
    # If using KPM, it's more efficient to normalize the 'ket' state vector
    # *before* building the full density matrix. This avoids an expensive
    # trace calculation on the much larger density matrix network.
    if isinstance(config, KPMConfig):
        print("    - Normalizing quantum state (ket)...")
        # Contract the ket with its conjugate to get the squared norm.
        norm_sq = ket_tn.H @ ket_tn

        backend = ket_tn.backend
        if backend == 'torch':
            if abs(norm_sq.item()) > 1e-12:
                norm_factor = torch.sqrt(norm_sq)
                # --- FIX: Use .modify() to safely update tensor data ---
                ket_tn.tensors[0].modify(data=ket_tn.tensors[0].data / norm_factor)
        else:  # numpy
            if abs(norm_sq) > 1e-12:
                norm_factor = np.sqrt(norm_sq)
                # --- FIX: Use .modify() to safely update tensor data ---
                ket_tn.tensors[0].modify(data=ket_tn.tensors[0].data / norm_factor)

    # Create the 'bra' from the (now possibly normalized) 'ket'.
    bra = ket_tn.copy().conj().reindex(bra_remap)
    bra.add_tag('BRA')

    # Contract the ket and bra along all shared indices (internal and cut).
    # The physical indices are left open, correctly forming ρ_A.
    rho_tn_uncontracted = ket_tn | bra

    # Step 4: Calculate entropy based on the chosen method
    print("  Step 4/4: Computing entanglement entropy...")
    left_inds = sorted(list(phys_inds))

    if isinstance(config, KPMConfig):
        # The state is already normalized, so we can proceed directly.
        moments, scale, shift = _compute_chebyshev_moments_ste(
            rho_tn_uncontracted, left_inds,
            num_moments=config.num_moments,
            num_vectors=config.num_vectors,
            bounds_method=config.bounds_method
        )
        entropy = _calculate_entropy_kpm(moments, scale, shift)
        print(f"✅ Final Calculated Entropy (KPM) = {entropy:.4f}")

    elif isinstance(config, BBPConfig) or isinstance(config, ExactConfig):
        # These methods normalize the final dense matrix, so pre-normalization is not needed.
        method_name = ""
        if isinstance(config, BBPConfig):
            method_name = "BBP"
            print("    - (EXPERIMENTAL) Contracting network with approximate Block BP...")
            rho_A = contract_with_block_bp(
                rho_tn_uncontracted,
                ket_tags=ket_tags,
                bra_tags={'BRA'},
                cut_inds=sorted(list(all_ket_outer_inds - phys_inds)),
                chi_msg=config.bbp_chi_msg, max_iterations=10
            )
        else:  # isinstance(config, ExactConfig)
            method_name = "Exact"
            print("    - Contracting network exactly (can be very slow)...")
            rho_A = rho_tn_uncontracted.contract(all, optimize='auto-hq')

        print("    - Normalizing dense matrix...")
        right_inds = [f"{ix}_" for ix in left_inds]
        trace_val = rho_A.trace(left_inds, right_inds)
        rho_A.modify(data=rho_A.data / trace_val)

        print("    - Performing exact diagonalization...")
        D_A = np.prod([rho_A.ind_size(ix) for ix in left_inds])
        mat = rho_A.to_dense(left_inds, right_inds).reshape(D_A, D_A)
        eigs = np.linalg.eigvalsh(mat)

        probs = eigs[eigs > 1e-12]
        entropy = -np.sum(probs * np.log2(probs))
        print(f"✅ Final Calculated Entropy ({method_name.upper()}) = {entropy:.4f}")

    else:
        raise TypeError(f"Unknown config type: '{type(config)}'.")

    # Return final results
    env_inds = [ind for ind in cut_inds if ind not in phys_inds]
    return {
        "entropy": entropy,
        "discrete_geodesic_length": discrete_geodesic_length,
        "geometric_geodesic_length": geometric_geodesic_length,
        "geodesic_path": geodesic_path,
        "cut_length": len(env_inds)
    }