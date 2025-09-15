import os
import random
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List, Dict
import numpy as np
import quimb.tensor as qtn
from joblib import Parallel, delayed
from quimb.tensor import MatrixProductOperator
from scipy.integrate import quad
from scipy.sparse.linalg import eigsh, LinearOperator
import torch

from src.holographic_tn.helper import random_mps, compress_tensor, compress_tn_new, truncated_split_via_svds, \
    compress_tn_svds, add_uniform_site_tags, stabilize_bond_names

# --------------------------------------------------------------------------- #
#                      GLOBAL KPM COMPRESSION OPTIONS                         #
# --------------------------------------------------------------------------- #

# Set to True to use the scalable MPS-based KPM. This avoids high-rank tensor
# errors and is necessary for large boundary regions.
USE_KPM_COMPRESSION = True

# The maximum bond dimension (chi) for the compressed MPS probe vectors.
# Smaller values are faster but less accurate.
KPM_COMPRESSION_MAX_BOND = 128

# --------------------------------------------------------------------------- #
#                      CPU PARALLELIZATION WORKER FUNCTION                      #
# --------------------------------------------------------------------------- #

def generate_seeds(num_seeds: int, master_seed: int = None) -> list[int]:
    """
    Generates a list of random seeds.

    Args:
        num_seeds: The number of seeds to generate.
        master_seed: An optional seed to make the sequence of generated
                     seeds reproducible. Defaults to None.

    Returns:
        A list of integers to be used as random seeds.
    """
    if master_seed is not None:
        random.seed(master_seed)

    # Define a large range for the seeds to be chosen from
    max_seed_value = 2 ** 32 - 1

    # Generate the list of seeds
    seed_list = [random.randint(0, max_seed_value) for _ in range(num_seeds)]

    return seed_list

def check_moment_stability(
    mu: complex,
    max_abs: float,
    prev_mu: complex = None,
    prev2_mu: complex = None,
    *,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-10,
    growth_factor: float = 0.05,
    curvature_limit: float = 0.05,
    verbose: bool = False
) -> bool:
    """
    Sign-invariant stability check for Chebyshev moments.

    Args:
        mu: Current moment (complex).
        max_abs: Theoretical absolute bound (D_A).
        prev_mu: Previous moment (optional, for growth check).
        prev2_mu: Moment before previous (optional, for curvature check).
        abs_tol: Absolute tolerance for imaginary part and floor values.
        rel_tol: Small relative slack for the max_abs bound.
        growth_factor: Max allowed ratio |mu| / |prev_mu|.
        curvature_limit: Max allowed normalized second-difference on |mu|.
        verbose: If True, emit warnings with reasons.

    Returns:
        True if stable, False otherwise.
    """
    # 1) Finite
    if not np.isfinite(mu):
        if verbose:
            warnings.warn(f"Moment is not finite: {mu}")
        return False

    # 2) Imaginary part (absolute tolerance, independent of D_A)
    if abs(mu.imag) > abs_tol:
        if verbose:
            warnings.warn(f"Imag part too large: {mu.imag} (abs_tol={abs_tol})")
        return False

    # 3) Theoretical bound on the real part
    if abs(mu.real) > max_abs * (1.0 + rel_tol):
        if verbose:
            warnings.warn(f"Real part {mu.real} exceeds bound {max_abs}")
        return False

    abs_mu = abs(mu)

    # 4) Magnitude growth vs previous (sign-invariant)
    if prev_mu is not None:
        abs_prev = max(abs(prev_mu), abs_tol)
        ratio = abs_mu / abs_prev
        if ratio > growth_factor:
            if verbose:
                warnings.warn(
                    f"Magnitude grew too fast: |μ_k|/|μ_{k-1}| = {ratio:.3e} > {growth_factor}"
                )
            return False

    # 5) Curvature of magnitudes (optional, sign-invariant)
    #    Checks |μ_k| - 2|μ_{k-1}| + |μ_{k-2}| normalized by max(|μ_{k-1}|, abs_tol)
    if prev_mu is not None and prev2_mu is not None:
        abs_prev = abs(prev_mu)
        abs_prev2 = abs(prev2_mu)
        denom = max(abs_prev, abs_tol)
        curvature = abs(abs_mu - 2.0 * abs_prev + abs_prev2) / denom
        if curvature > curvature_limit:
            if verbose:
                warnings.warn(
                    f"Excessive curvature in magnitudes: {curvature:.3e} > {curvature_limit}"
                )
            return False

    return True

def _kpm_cpu_worker(
        worker_id: int,
        rho_tn_uncontracted: qtn.TensorNetwork,
        left_inds: List[str],
        num_moments: int,
        scale: float,
        shift: float
) -> np.ndarray:
    """
    Performs the KPM recurrence for a single random vector.
    This function is designed to be called in a separate process.
    """
    # Ensure each worker process has a unique random seed
    # np.random.seed(worker_id)

    # Re-create objects needed for the calculation within the new process
    with warnings.catch_warnings():
        dims = {ix: rho_tn_uncontracted.ind_size(ix) for ix in left_inds}
        D_A = np.prod(list(dims.values()))
        dtype = np.complex128

        right_inds = [f"{ix}_" for ix in left_inds]
        remap = dict(zip(left_inds, right_inds))

        def apply_rho_rescaled(tensor_v):
            v_for_contract = tensor_v.copy().reindex(remap)
            res = (rho_tn_uncontracted @ v_for_contract).squeeze()
            return (2 * res - shift * tensor_v) / scale

        # --- Main KPM logic for a single vector ---
        moments = np.zeros(num_moments, dtype=dtype)
        # v_data = np.random.randn(D_A).astype(dtype)
        v_data = np.random.choice([-1.0, 1.0], size=D_A).astype(dtype)
        v = qtn.Tensor(v_data.reshape([dims[ix] for ix in left_inds]), inds=left_inds)

        # v.normalize_()

        def get_scalar(tensor_expression):
            return tensor_expression.item() if hasattr(tensor_expression, 'item') else tensor_expression

        REORTHOGONALIZE = True
        ORTH_FREQ = 25

        t_k_minus_1 = v
        moments[0] = get_scalar(v.H @ t_k_minus_1)

        t_k = apply_rho_rescaled(v)
        moments[1] = get_scalar(v.H @ t_k)

        log_factor = np.log10(D_A) if D_A > 1 else 1.0
        amplitude = 10 * max(1.0, log_factor)
        # relative_tolerance = amplitude * epsilon

        # --- "Smart" Tolerance based on machine precision ---
        # Get the machine epsilon for the float type underlying our complex dtype
        float_dtype = np.float64
        epsilon = np.finfo(float_dtype).eps
        # Set a tolerance that is a small multiple of machine epsilon
        relative_tolerance = amplitude * epsilon

        cs = []

        for k in range(1, num_moments - 1):
            t_k_plus_1 = 2 * apply_rho_rescaled(t_k) - t_k_minus_1  #

            mu_k = (v.H @ t_k_plus_1)  #
            #if not np.isfinite(mu_k) or abs(mu_k) > D_A * (1 + relative_tolerance):
            #    warnings.warn(
            #        f"Worker {worker_id} became unstable at moment {k + 2}. Truncating worker's results to {k + 1} moments.")
            #    return moments[:k + 1]

            moments[k + 1] = mu_k  #
            t_k_minus_1, t_k = t_k, t_k_plus_1  #

    return moments

# --------------------------------------------------------------------------- #
#                      SPECTRAL BOUNDS ESTIMATION HELPERS                       #
# --------------------------------------------------------------------------- #

def _estimate_spectral_bounds_fast(
        rho_tn_uncontracted: qtn.TensorNetwork,
        left_inds: List[str],
        num_iter: int = 15,
        compare_methods: bool = False
) -> Tuple[float, float]:
    """
    Estimates spectral bounds using a scalable MPS-based power method.

    If `compare_methods` is set to True, it will also run the original dense
    vector method in the same loop to provide a direct, step-by-step
    performance comparison. The MPS implementation is based on your provided code.
    """
    # --- Universal Setup ---
    bond_dim = 16
    try:
        site_no = lambda ix: int(re.search(r'\d+', ix).group())
        left_inds_sorted = sorted(list(left_inds), key=site_no)
    except (TypeError, AttributeError):
        print("    - ⚠️ Warning: Could not deterministically sort site indices.")
        left_inds_sorted = sorted(list(left_inds))

    L = len(left_inds_sorted)
    phys_dim = rho_tn_uncontracted.ind_size(left_inds_sorted[0])

    # --- MPS Method Initialization (Your Method) ---
    v_mps = qtn.MPS_rand_state(L=L, bond_dim=bond_dim, phys_dim=phys_dim)
    ket_inds_map = {f'k{i}': ind for i, ind in enumerate(left_inds_sorted)}
    v_mps.reindex_(ket_inds_map)
    for i, T in enumerate(v_mps.tensors):
        T.add_tag(f"SITE{i}")
    v_mps.normalize()

    # --- Main Logic: Switch between comparison and MPS-only mode ---
    if not compare_methods:
        # --- Standard MPS-Only Execution (Your Method) ---
        right_inds = [f"{ix}_" for ix in left_inds]
        remap = dict(zip(left_inds, right_inds))

        # rho_tn_uncontracted = rho_tn_uncontracted.compress_all(max_bond=2)

        print("--- Running Scalable MPS Method ---")
        lambda_max_mps = 0.0
        for i in range(num_iter):
            v_mps_for_contract = v_mps.copy().reindex(remap)
            w_tn = rho_tn_uncontracted @ v_mps_for_contract
            lambda_max_mps = w_tn.norm()

            print(f"      MPS Iteration {i + 1}/{num_iter}, λ_max ≈ {lambda_max_mps:.6f}", end='\r')
            v_mps = w_tn / lambda_max_mps
        print("\n--- MPS Method Complete ---")
        return 0.0, float(np.real(lambda_max_mps))
    else:
        # --- In-Loop Comparison Execution ---
        print("--- Running MPS and Dense Methods in Parallel for Comparison ---")

        # --- Dense Method Initialization ---
        try:
            D_A = np.prod([rho_tn_uncontracted.ind_size(ix) for ix in left_inds])
            if D_A == 0: raise ValueError("Dimension D_A overflowed")
            dims = {ix: rho_tn_uncontracted.ind_size(ix) for ix in left_inds}
            v_data = np.random.randn(D_A) + 1j * np.random.randn(D_A)
            v_dense = qtn.Tensor(v_data.reshape([dims[ix] for ix in left_inds]), inds=list(left_inds))
            v_dense.normalize_()
            right_inds = [f"{ix}_" for ix in left_inds]
            remap = dict(zip(left_inds, right_inds))
        except (np.core._exceptions.MemoryError, ValueError) as e:
            print(f"\n    - FATAL ERROR: Cannot run dense method. {e}")
            print("      System is too large. Comparison is not possible.")
            return 0.0, -1.0

        # --- Combined Iteration Loop ---
        start_time = time.time()
        lambda_max_mps, lambda_max_dense = 0.0, 0.0

        for i in range(num_iter):
            # --- MPS Step (Your Method) ---
            v_mps_for_contract = v_mps.copy().reindex(remap)
            w_tn_mps = rho_tn_uncontracted @ v_mps_for_contract
            lambda_max_mps = w_tn_mps.norm()
            v_mps = w_tn_mps / lambda_max_mps

            # --- Dense Step ---
            v_for_contract = v_dense.copy().reindex(remap)
            w_dense = (rho_tn_uncontracted @ v_for_contract).squeeze()
            lambda_max_dense = w_dense.norm()
            v_dense = w_dense / lambda_max_dense

            # --- Print Comparison ---
            print(
                f"  Iter {i + 1: >2}/{num_iter} | MPS λ_max: {lambda_max_mps:.6f} | Dense λ_max: {lambda_max_dense:.6f}")

        end_time = time.time()
        print("\n--- Comparison Complete ---")
        print(f"    Total Time Elapsed: {end_time - start_time:.4f} seconds")

        return 0.0, float(np.real(lambda_max_dense))


def _estimate_spectral_bounds_accurate(
        rho_tn_uncontracted: qtn.TensorNetwork,
        left_inds: List[str]
) -> Tuple[float, float]:
    """Estimates spectral bounds [λ_min, λ_max] accurately using Lanczos (eigsh).

    This method is slower but provides a high-precision estimate of the largest
    eigenvalue by running the Lanczos algorithm until convergence.

    Args:
        rho_tn_uncontracted: The efficient, uncontracted tensor network for ρ_A.
        left_inds: The list of physical 'ket' indices for the subsystem.

    Returns:
        A tuple containing the estimated (λ_min, λ_max).
    """
    print("    - Estimating spectral bounds with 'accurate' method (Lanczos)...")
    first_tensor = next(iter(rho_tn_uncontracted.tensors))
    backend = first_tensor.backend
    device = first_tensor.data.device if backend == 'torch' else None

    D_A = np.prod([rho_tn_uncontracted.ind_size(ix) for ix in left_inds])
    dims = {ix: rho_tn_uncontracted.ind_size(ix) for ix in left_inds}
    right_inds = [f"{ix}_" for ix in left_inds]
    remap = dict(zip(left_inds, right_inds))

    def matvec(v_flat):
        if backend == 'torch':
            v_data = torch.from_numpy(v_flat).to(device, dtype=torch.complex64)
        else:
            v_data = v_flat.astype(np.complex128)

        v_tensor = qtn.Tensor(v_data.reshape([dims[ix] for ix in left_inds]), inds=left_inds)
        v_for_contract = v_tensor.copy().reindex(remap)
        res_tensor = (rho_tn_uncontracted @ v_for_contract).squeeze()

        if backend == 'torch':
            return res_tensor.data.cpu().numpy().flatten()
        else:
            return res_tensor.data.flatten()

    op = LinearOperator(shape=(D_A, D_A), matvec=matvec, dtype=np.complex128)

    try:
        # --- FIX: Increase NCV and provide an explicit start vector (v0) ---
        # Create a random starting vector on the CPU for the solver.
        v0 = np.random.rand(D_A) + 1j * np.random.rand(D_A)

        # Use a larger Krylov subspace (ncv) and more iterations to ensure convergence
        ncv = min(D_A, max(20, 2 * 1 + 1))  # scipy default is min(n, max(2*k+1, 20))
        maxiter = D_A * 10  # Generous number of iterations

        # Set ncv to a larger value (e.g., 50) to aid convergence.
        lambda_max = eigsh(
            op,
            k=1,
            which='LM',
            v0=v0,
            ncv=ncv,
            maxiter=maxiter,
            return_eigenvectors=False
        )[0]

        lambda_min = eigsh(
            op,
            k=1,
            which='SM',
            v0=v0,
            ncv=ncv,
            maxiter=maxiter,
            return_eigenvectors=False
        )[0]
        # --------------------------------------------------------------------

    except Exception as e:
        print(f"Warning: Accurate eigsh failed. Falling back to fixed bounds. Error: {e}")
        return 0.0, 1.0

    return float(abs(np.real(lambda_min))), float(abs(np.real(lambda_max)))
    # return 0.0, float(abs(np.real(lambda_max)))

def _estimate_spectral_bounds(
        rho_tn_uncontracted: qtn.TensorNetwork,
        left_inds: List[str],
        method: str = 'accurate'
) -> Tuple[float, float]:
    """Estimates spectral bounds using a selectable method.

    This function acts as a controller to switch between the fast, approximate
    power iteration method and the slower, more accurate Lanczos method.

    Args:
        rho_tn_uncontracted: The tensor network for the density matrix.
        left_inds: The list of physical 'ket' indices.
        method (str): The estimation method to use.
            - 'accurate': (Default) Uses the Lanczos algorithm for high precision.
            - 'fast': Uses a few power iterations for a quick estimate.

    Returns:
        A tuple containing the estimated (λ_min, λ_max).
    """
    if method == 'accurate':
        return _estimate_spectral_bounds_accurate(rho_tn_uncontracted, left_inds)
    elif method == 'fast':
        return _estimate_spectral_bounds_fast(rho_tn_uncontracted, left_inds)
    else:
        raise ValueError("`method` for spectral bounds must be 'accurate' or 'fast'.")


# --------------------------------------------------------------------------- #
#                       KERNEL POLYNOMIAL METHOD (KPM)                          #
# --------------------------------------------------------------------------- #

def _compute_chebyshev_moments_ste(
        rho_tn_uncontracted: qtn.TensorNetwork,
        left_inds: List[str],
        num_moments: int,
        num_vectors: int = 50,
        sample_batch: int = 10,
        bounds_method: str = 'accurate'
) -> Tuple[np.ndarray, float, float]:
    """Computes Chebyshev moments μ_n = Tr(T_n(ρ_A)) via Stochastic Trace Estimation.

    This function implements the core of the KPM algorithm. For the 'gpu'
    backend, it uses efficient batch processing over all random vectors. For the
    'cpu' backend, it processes vectors sequentially.

    Args:
        rho_tn_uncontracted: The efficient, uncontracted tensor network for ρ_A.
        left_inds: The list of physical 'ket' indices.
        num_moments: The number of Chebyshev moments to compute.
        num_vectors: The number of random probe vectors for the stochastic average.
        bounds_method: Method to use for spectral bounds ('accurate' or 'fast').

    Returns:
        A tuple containing the computed moments, scale factor, and shift factor.
    """
    first_tensor = next(iter(rho_tn_uncontracted.tensors))
    backend = first_tensor.backend
    device = first_tensor.data.device if backend == 'torch' else None

    dtype = torch.complex64 if backend == 'torch' else np.complex128

    print(f"    - Computing {num_moments} moments with {num_vectors} random vectors...")

    # Get spectral bounds using the chosen method
    lambda_min, lambda_max = _estimate_spectral_bounds(
        rho_tn_uncontracted, left_inds, method=bounds_method
    )
    scale = (lambda_max - lambda_min)
    min_scale_threshold = 1e-2  # A more robust threshold

    if scale < min_scale_threshold:
        print(f"    - ⚠️ Warning: Estimated spectral width ({scale:.2e}) is very small.")
        print(f"      Using a minimum width of {min_scale_threshold} to avoid numerical instability.")
        scale = min_scale_threshold

    shift = (lambda_max + lambda_min)
    #shift = 20
    error = shift - 0.0625

    # Define the rescaled operator and index mappings
    right_inds = [f"{ix}_" for ix in left_inds]
    remap = dict(zip(left_inds, right_inds))

    def _apply_rho_rescaled(tensor_v):
        v_for_contract = tensor_v.copy().reindex(remap)
        res = (rho_tn_uncontracted @ v_for_contract).squeeze()
        return (2 * res - shift * tensor_v) / scale

    D_A = np.prod([rho_tn_uncontracted.ind_size(ix) for ix in left_inds])
    dims = {ix: rho_tn_uncontracted.ind_size(ix) for ix in left_inds}

    if backend == 'torch':
        # somewhere once, after you know `scale` and `shift`:
        alpha = 2.0 / scale
        beta = shift / scale

        # promote your scalar factors into GPU tensors of the same dtype
        # (this will live on the GPU and participate in FMA)
        alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
        beta_t = torch.tensor(beta, dtype=dtype, device=device)

        def apply_rho_rescaled(tensor_v: qtn.Tensor) -> qtn.Tensor:
            # 1) build the “right‐side” indices view
            v_rhs = tensor_v.copy().reindex(remap)

            # 2) contract rho onto v_rhs on the GPU
            res_tn = (rho_tn_uncontracted @ v_rhs).squeeze()

            # 3) pull out the raw torch.Tensor for fused arithmetic
            res_data = res_tn.data  # shape = [..., batch], dtype=torch.complex64
            v_data = tensor_v.data  # same shape & dtype

            # 4) do one fused multiply‐add on GPU:
            #      out = (2/scale)*res_data  +  (−shift/scale)*v_data
            #    using .mul/.add so PyTorch emits a single FMA if supported
            out_data = res_data.mul(alpha_t) \
                .add(v_data, alpha=-beta_t)

            # 5) wrap back into a qtn.Tensor with the same indices
            return qtn.Tensor(out_data, inds=tensor_v.inds)

        # --- GPU Batch Processing Path ---
        batch_shape = [dims[ix] for ix in left_inds] + [num_vectors]
        v_data = torch.randn(*batch_shape, dtype=dtype, device=device) / np.sqrt(2.0)
        v_batch = qtn.Tensor(v_data, inds=left_inds + ['batch'])

        def batch_vdot(v_batch_1, v_batch_2, k):
            result_tensor = (v_batch_1.conj() @ v_batch_2)
            #result_tensor -= (D_A * (-1) ^ (k - 1)) * num_vectors
            return (result_tensor.data.sum().item() / num_vectors)

        # --- "Smart" Tolerance based on machine precision ---
        # Get the machine epsilon for the float type underlying our complex dtype
        float_dtype = np.float64
        epsilon = np.finfo(float_dtype).eps
        # Set a tolerance that is a small multiple of machine epsilon
        relative_tolerance = 100 * epsilon

        moments = np.zeros(num_moments, dtype=np.complex128)
        t_k_minus_1 = v_batch
        moments[0] = batch_vdot(v_batch, t_k_minus_1, 0)
        t_k = apply_rho_rescaled(v_batch)
        moments[1] = batch_vdot(v_batch, t_k, 1)
        for k in range(1, num_moments - 1):
            print(f"      Moment {k + 1}/{num_moments}", end='\r')
            t_k_plus_1 = 2 * apply_rho_rescaled(t_k) - t_k_minus_1
            moments[k + 1] = batch_vdot(v_batch, t_k_plus_1, k)
            t_k_minus_1 = t_k
            t_k = t_k_plus_1

            if not np.isfinite(moments[k + 1]) or abs(moments[k + 1]) > D_A * (1 + relative_tolerance):
                print(f"\n    - ⚠️ Warning: Numerical instability detected at moment {k + 2}. "
                      f"Truncating to {k + 1} moments to ensure stability.")

                # Truncate the moments array to the last stable value
                moments = moments[:k + 1]
                break
    else:
        # --- CPU PARALLEL PROCESSING with Truncation ---
        print(f"    - Using all available cores for parallel CPU computation with Joblib.")

        worker_seeds = generate_seeds(num_seeds=num_vectors)  #
        worker_args = [(worker_seeds[i], rho_tn_uncontracted, left_inds, num_moments, scale, shift)
                       for i in range(num_vectors)]

        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(_kpm_cpu_worker)(*args) for args in worker_args
        )

        successful_results = [res for res in results if res is not None and len(res) > 1]
        if not successful_results:
            raise RuntimeError("All KPM workers failed immediately.")

        # 1. Calculate the median length of all successful runs.
        all_lengths = [len(res) for res in successful_results]
        median_len = int(np.median(all_lengths))

        # 2. Set a lower bound as a fraction of the median (e.g., 80%).
        lower_bound_fraction = 0.8
        lower_bound_len = int(median_len * lower_bound_fraction)

        # 3. Discard any "outlier" runs that are shorter than this lower bound.
        main_cluster_results = [res for res in successful_results if len(res) >= lower_bound_len]

        if not main_cluster_results:
            raise RuntimeError(f"No runs reached the lower bound length of {lower_bound_len}.")

        # 4. Find the minimum length among the remaining "main cluster" of runs.
        truncation_len = min(len(res) for res in main_cluster_results)

        print(f"    - Truncating results to {truncation_len} moments (based on a median of {median_len}).")

        # 5. Warn the user if truncation is happening.
        if truncation_len < num_moments:
            warnings.warn(
                f"Numerical instability detected. "
                f"Results truncated to {truncation_len} moments.",
                UserWarning
            )

        # 6. Truncate all runs in the main cluster to this new minimum length.
        truncated_results = [res[:truncation_len] for res in main_cluster_results]

        total_moments = np.sum(truncated_results, axis=0)
        moments = total_moments / len(main_cluster_results)

    return np.real(moments), scale, shift


def _calculate_entropy_kpm(moments, scale, shift) -> float:
    """Reconstructs the spectral density from moments and computes entropy.

    This function runs on the CPU as it operates on the small `moments` array.
    """
    num_moments = len(moments)

    # Jackson kernel for damping Gibbs oscillations
    n = np.arange(num_moments)
    g_k = ((num_moments - n + 1) * np.cos(np.pi * n / (num_moments + 1)) +
           np.sin(np.pi * n / (num_moments + 1)) / np.tan(np.pi / (num_moments + 1))) / (num_moments + 1)
    damped_moments = moments * g_k

    def f_kpm(lam):
        # Rescale lambda back to x in [-1, 1]
        x = (2 * lam - shift) / scale
        if abs(x) > 1:
            return 0.0

        # Sum the Chebyshev series
        k = np.arange(num_moments)
        chebyshev_series = damped_moments * np.cos(k * np.arccos(x))
        # The T_0 term is counted once, others twice
        density = (1 / (np.pi * np.sqrt(1 - x ** 2))) * (chebyshev_series[0] + 2 * np.sum(chebyshev_series[1:]))

        # Return density, rescaled to original lambda domain
        return density * 2 / scale

    def entropy_integrand(lam):
        if lam <= 1e-15:
            return 0.0
        return -lam * np.log2(lam) * f_kpm(lam)

    # Integrate over the original spectral range
    lambda_min_orig = (scale * (-1) + shift) / 2
    lambda_max_orig = (scale * (1) + shift) / 2
    entropy, _ = quad(entropy_integrand, lambda_min_orig, lambda_max_orig, limit=200)

    return entropy