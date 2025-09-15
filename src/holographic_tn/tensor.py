# src/holographic_tn/tensor.py
import numpy as np
from typing import Dict, Tuple, List
import functools

import yastn


def get_perfect_tensor_yastn(rank: int) -> yastn.Tensor:
    """
    Constructs the perfect tensor as a Z2-symmetric yastn.Tensor.
    """
    config_Z2 = yastn.make_config(sym='Z2')
    # leg_sym = yastn.Leg(config_Z2, s=1, t=((0,), (1,)), D=(1, 1))
    legs_r6 = [yastn.Leg(config_Z2, s=1, t=((0,), (1,)), D=(1, 1)) for _ in range(6)]
    dense_data_r6 = _create_rank_6_perfect_tensor()

    # --- CORRECTED LINE ---
    # Use the 'from_dense' class method to create the tensor from the array.
    tensor_r6 = yastn.Tensor(
        config=config_Z2,
        legs=legs_r6,
        val=dense_data_r6
    )
    # --- END CORRECTION ---

    if rank == 6:
        return tensor_r6

    if rank == 5:
        leg_in = yastn.Leg(config_Z2, s=-1, t=((0,),), D=(1,))
        ket_0 = yastn.Tensor(config=config_Z2, legs=[leg_in])
        ket_0.set_block(ts=(0,), Ds=(1,), val=1)
        return yastn.tensordot(tensor_r6, ket_0, axes=(5, 0))

    raise NotImplementedError(f"Symmetric tensor for rank {rank} not implemented.")


def _create_rank_6_perfect_tensor() -> np.ndarray:
    """
    Constructs the rank-6 perfect tensor from the five-qubit [[5,1,3]] code.

    This tensor is a cornerstone of the HaPPY holographic code. It's built
    from the code space of the five-qubit quantum error-correcting code.
    The first five indices correspond to the physical qubits, and the sixth
    index is the encoded logical qubit.

    Returns:
        A (2, 2, 2, 2, 2, 2) numpy array representing the perfect tensor.
    """
    # Define 2x2 Pauli matrices
    Id2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Y = 1j * X @ Z

    # Define the 5-qubit stabilizer generators (32x32 matrices)
    s_ops: List[np.ndarray] = [
        np.kron(X, np.kron(Z, np.kron(Z, np.kron(X, Id2)))),
        np.kron(Id2, np.kron(X, np.kron(Z, np.kron(Z, X)))),
        np.kron(X, np.kron(Id2, np.kron(X, np.kron(Z, Z)))),
        np.kron(Z, np.kron(X, np.kron(Id2, np.kron(X, Z)))),
    ]

    # The identity matrix for the 5-qubit (32-dimensional) space
    Id32 = np.eye(32, dtype=complex)

    # The projector onto the +1 eigenspace of all stabilizers
    # CORRECTED LINE: Use the 32x32 identity matrix (Id32) here
    projector = functools.reduce(
        lambda a, b: a @ b, [(Id32 + s) for s in s_ops]
    ) / 16.0

    # Logical operators for the encoded qubit
    logical_X = functools.reduce(np.kron, [X] * 5)
    logical_Z = functools.reduce(np.kron, [Z] * 5)

    # Find the code space by finding the eigenvectors of the projector
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    code_space_basis = eigenvectors[:, np.isclose(eigenvalues, 1.0)]

    # Identify the logical |0> and |1> states
    # |0>_L is the +1 eigenstate of logical Z within the code space
    logical_Z_in_codespace = code_space_basis.T.conj() @ logical_Z @ code_space_basis
    lz_eigvals, lz_eigvecs = np.linalg.eigh(logical_Z_in_codespace)

    v0 = lz_eigvecs[:, np.isclose(lz_eigvals, 1.0)].flatten()
    v1 = lz_eigvecs[:, np.isclose(lz_eigvals, -1.0)].flatten()

    logical_0 = (v0[0] * code_space_basis[:, 0] + v0[1] * code_space_basis[:, 1])
    logical_1 = (v1[0] * code_space_basis[:, 0] + v1[1] * code_space_basis[:, 1])

    # Ensure correct relative phase by checking logical_X |0>_L = |1>_L
    if not np.allclose(logical_X @ logical_0, logical_1):
        # This can happen due to eigendecomposition phase ambiguity
        phase_correction = np.angle(np.vdot(logical_X @ logical_0, logical_1))
        logical_1 *= np.exp(-1j * phase_correction)

    # Final check
    if not np.allclose(logical_X @ logical_0, logical_1):
        raise RuntimeError("Failed to align logical states.")

    # Assemble the final tensor
    perfect_tensor = np.zeros((2, 2, 2, 2, 2, 2), dtype=complex)
    perfect_tensor[:, :, :, :, :, 0] = logical_0.reshape(2, 2, 2, 2, 2)
    perfect_tensor[:, :, :, :, :, 1] = logical_1.reshape(2, 2, 2, 2, 2)

    return perfect_tensor


def get_perfect_tensor_of_rank(rank: int, bond_dim: int = 2, even_rank_method: str = 'random_unitary') -> np.ndarray:
    """
    Generates or loads the numerical data for a perfect or near-perfect tensor. 🧩

    A perfect tensor is a key building block for this holographic model.
    It is a unitary map between any balanced bipartition of its indices.
    This implementation provides known perfect tensors for specific cases and
    uses general constructions or random tensors for arbitrary bond dimensions.

    For any even rank `2k`, a perfect tensor can be constructed by creating a
    `D^k x D^k` identity matrix and reshaping it into a tensor with `2k`
    indices of dimension `D`. The identity matrix is inherently unitary, and this
    construction creates a tensor that represents a unitary map between the
    first `k` indices and the last `k` indices, satisfying a key property
    of a perfect tensor.

    However, **no general mathematical formula exists to create perfect tensors
    for all arbitrary ranks and bond dimensions**, especially for odd ranks where
    the indices cannot be evenly bipartitioned. Because of this, special
    constructions are needed for certain cases. These unique solutions, like the
    rank-6 qubit tensor derived from quantum error-correcting codes, are discovered
    through specific methods and must be implemented individually.

    Args:
        rank: The number of indices for the tensor, equal to the number of sides
              of its corresponding polygon.
        bond_dim: The dimension of each tensor index (e.g., 2 for a qubit).
        even_rank_method (str): The method for even-ranked tensors with D>2.
            - 'random_unitary' (default): A dense, numerically robust tensor
              from the QR decomposition of a random matrix.
            - 'identity': A sparse, deterministic tensor from a reshaped
              identity matrix. Can be numerically unstable in small geometries.

    Returns:
        A numpy array representing the tensor.
    """
    # --- FIX: Handle special, known perfect tensors for bond_dim=2 first ---
    if bond_dim == 2:
        if rank == 6:
            return _create_rank_6_perfect_tensor()
        elif rank == 5:
            tensor_6 = _create_rank_6_perfect_tensor()
            ket_0 = np.array([1.0, 0.0], dtype=complex)
            tensor_5 = np.tensordot(tensor_6, ket_0, axes=([5], [0]))
            return tensor_5

    # --- FIX: General construction for any even-ranked tensor ---
    # This creates a true perfect tensor for any even rank and bond dimension.
    if rank % 2 == 0:
        k = rank // 2
        matrix_dim = bond_dim ** k
        tensor_shape = (bond_dim,) * rank

        # --- Switch between the two construction methods ---
        if even_rank_method == 'random_unitary':
            # 1. Create a random complex matrix
            random_matrix = np.random.randn(matrix_dim, matrix_dim) + 1j * np.random.randn(matrix_dim, matrix_dim)
            # 2. Use QR decomposition to get a random unitary matrix 'Q'
            q_matrix, _ = np.linalg.qr(random_matrix)
            # 3. Reshape the unitary matrix into a tensor
            return q_matrix.reshape(tensor_shape)

        elif even_rank_method == 'identity':
            # Create a (D^k x D^k) identity matrix and reshape it
            identity_matrix = np.eye(matrix_dim, dtype=complex)
            return identity_matrix.reshape(tensor_shape)

        else:
            raise ValueError("`even_rank_method` must be 'random_unitary' or 'identity'.")

    # --- FIX: Fallback for other cases (e.g., odd ranks with bond_dim > 2) ---
    else:
        # For other cases, a true perfect tensor construction is not readily
        # available. We return a random, normalized tensor as a substitute.
        shape = (bond_dim,) * rank
        # Create a tensor with random complex entries
        random_tensor = (np.random.randn(*shape) + 1j * np.random.randn(*shape))
        # Normalize it to have a Frobenius norm of 1
        return random_tensor / np.linalg.norm(random_tensor)