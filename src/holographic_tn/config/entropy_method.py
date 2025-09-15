class EntropyMethod(object):

    def __init__(self):
        pass

class ExactConfig(EntropyMethod):
    """Configuration for the exact entropy calculation method.

    This method performs a full, exact contraction of the tensor network to
    form the dense density matrix. It provides the most accurate result but is
    **extremely demanding** in terms of memory and computation time, making it
    feasible only for very small systems. 💪
    """

    def __init__(
        self,
        optimise: str = 'auto-hq',
    ):
        """Initializes the Exact configuration.

        Args:
            optimise (str): A string specifying the contraction path optimizer
                for `quimb` to use. Finding an optimal path is crucial for
                performance. Defaults to 'auto-hq'.

                ---
                **Commonly Used Optimizers**
                ---

                - **`'greedy'`**: ⚡️ A fast, simple, but "short-sighted"
                  optimizer that often produces suboptimal paths for complex
                  networks. It picks the single best pairwise contraction at
                  each step.

                - **`'random-greedy'`**: ✅ A smarter, randomized version of
                  greedy that finds better paths. It offers a good balance
                  of search time and path quality.

                - **`'auto'` / `'auto-hq'`**: 🧠 The **recommended default**. These
                  are "meta-optimizers" that intelligently analyze the network
                  and select the best strategy. `'auto-hq'` is more exhaustive
                  and finds higher-quality paths, often using advanced methods
                  internally.

                ---
                **Advanced & Specialized Optimizers**
                ---

                - **`'dp'`** (Dynamic Programming): 🧠 The "Optimal but Expensive"
                  option. It finds the **provably optimal** path, but its
                  runtime scales exponentially with network complexity. Only
                  feasible for tree-like or very simple networks.

                - **`'kahypar'`**: 🗺️ The "Divide and Conquer" strategy. Uses a
                  powerful external library to recursively partition large,
                  complex networks. `'auto-hq'` often uses this strategy
                  behind the scenes. Requires `pip install kahypar`.

                - **`'nevergrad'`**: 🧪 An experimental option using advanced
                  search algorithms from the `nevergrad` library. Primarily for
                  researchers tackling exceptionally difficult problems.
        """
        super().__init__()
        self.optimise = optimise