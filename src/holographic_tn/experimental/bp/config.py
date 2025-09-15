from src.holographic_tn.config.entropy_method import EntropyMethod


class BBPConfig(EntropyMethod):
    """Configuration for the Block Belief Propagation (BBP) entropy calculation.

    This is an **experimental** method that uses an approximate tensor network
    contraction algorithm to form a dense density matrix. The resulting matrix
    is then diagonalized to compute the von Neumann entropy. It offers a
    balance between the full accuracy of an exact contraction and the
    matrix-free approach of KPM. 🧪
    """

    def __init__(
        self,
        bbp_chi_msg: int = 20,
        max_iterations: int = 10,
    ):
        """Initializes the BBP configuration with specific parameters.

        Args:
            bbp_chi_msg (int): The maximum bond dimension (χ) for the messages
                passed between tensor blocks during the belief propagation
                algorithm. A larger value increases the accuracy of the
                approximate contraction at the cost of higher computational
                and memory requirements. Defaults to 20.
            max_iterations (int): The maximum number of iterations for the
                belief propagation algorithm to run before termination, even
                if convergence has not been reached. Defaults to 10.
        """
        super().__init__()
        self.bbp_chi_msg = bbp_chi_msg
        self.max_iterations = max_iterations