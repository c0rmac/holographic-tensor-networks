from src.holographic_tn.config.entropy_method import EntropyMethod


class KPMConfig(EntropyMethod):
    """Configuration for the Kernel Polynomial Method (KPM) of entropy calculation.

    This method is memory-efficient and well-suited for large tensor networks
    as it avoids constructing the full density matrix. It relies on a
    stochastic trace estimation using a Chebyshev polynomial expansion.
    """

    def __init__(
        self,
        num_moments: int = 250,
        num_vectors: int = 10,
        bounds_method: str = 'fast',
    ):
        """Initializes the KPM configuration with specific parameters.

        Args:
            num_moments (int): The number of Chebyshev moments to compute. A
                higher number generally leads to a more accurate entropy
                estimate at the cost of computation time. Defaults to 250.
                **Warning**: Increasing this value beyond 250 has been found
                to cause numerical instability when computing the Chebyshev
                series integral. This effect is particularly pronounced on
                GPUs due to their limited support for double-precision
                complex numbers (`complex128`). ⚠️
            num_vectors (int): The number of random vectors used for the
                stochastic trace estimation. More vectors improve accuracy
                but increase computational cost. Defaults to 10.
            bounds_method (str): The method for estimating the spectral bounds
                (min/max eigenvalues) of the operator. Can be 'fast' (uses
                a few Lanczos iterations) or 'exact'. Defaults to 'fast'.
        """
        super().__init__()
        self.num_moments = num_moments
        self.num_vectors = num_vectors
        self.bounds_method = bounds_method