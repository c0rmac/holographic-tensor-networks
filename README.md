# Holographic Tensor Networks (`holographic_tn`)

[![PyPI version](https://badge.fury.io/py/holographic-tn.svg)](https://badge.fury.io/py/holographic-tn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/example/holographic_tn)

`holographic_tn` is a Python library for creating and analyzing holographic tensor networks built on the geometric framework of hyperbolic buildings. This tool serves as a "numerical laboratory for holography", transforming the abstract mathematical theory of "Holographic tensor networks from hyperbolic buildings" into a tangible and interactive computational tool.

The library provides a robust geometric kernel for constructing complex hyperbolic geometries and a physics engine for simulating the quantum states defined upon them. Its primary goal is to enable the numerical investigation of the AdS/CFT correspondence and the Ryu-Takayanagi formula in generalized settings beyond simple, regular lattices.

## 📚 Documentation

- [Hyperbolic Geometry Theory](./docs/theory/THEORY.md)  
  Foundations of hyperbolic geometry, Gromov hyperbolicity, and their role in holographic tensor networks.

- [Quantum Mechanics in Hyperbolic Geometry](./docs/theory_quantum/THEORY_QUANTUM.md)  
  How quantum states, entanglement, and the AdS/CFT correspondence manifest in hyperbolic space.

- [Example Usage](./docs/example_usage/EXAMPLE_USAGE.md)  
  Step-by-step examples demonstrating geometry construction, tensor network building, and entropy calculations.

## ✨ Key Features

* **Geometric Kernel**: A powerful `HyperbolicBuilding` class to programmatically generate, store, and analyze the geometry of hyperbolic buildings.
* **Procedural Construction**: Generate regular hyperbolic tilings (`{p, q}`) or more complex buildings from group-theoretic side-pairing rules.
* **Geodesic Finding**: Implements the A* algorithm to efficiently find discrete minimal surfaces (geodesics) in the bulk geometry.
* **Tensor Network Construction**: Automatically builds a `quimb` tensor network from the geometry, placing perfect tensors on each face and defining connectivity through the building's topology.
* **Flexible Backends**: Supports both CPU (`numpy`) for high-precision and GPU (`torch`) for accelerated computations on large networks.
* **Advanced Entropy Calculation**: Includes multiple methods for calculating entanglement entropy:
    * **Kernel Polynomial Method (KPM)**: A memory-efficient method for large systems that avoids constructing the full density matrix.
    * **Exact Diagonalization**: For small systems requiring high precision.
    * **Block Belief Propagation (BBP)**: An experimental, approximate contraction method tailored to the tree-like nature of hyperbolic space.

## 🧠 Conceptual Background

This library implements a specific model of the AdS/CFT correspondence where a quantum state on a boundary is described by a tensor network in a higher-dimensional bulk space.

### Hyperbolic Geometry

The geometric setting is a **Gromov hyperbolic space**, which exhibits tree-like properties at large scales. The library uses a discrete representation called a **simplicial complex** to model this space. Specifically, it constructs a **hyperbolic building**, which is a complex created by gluing together simpler pieces called "apartments" (regular tessellations of the hyperbolic plane). This framework provides a combinatorial skeleton of the continuous geometry that is perfectly suited for defining a tensor network.

### Tensor Network Holography

The connection between geometry and quantum mechanics is made as follows:

1.  **Tensor Placement**: A **perfect tensor** is associated with each 2-simplex (a polygonal face) in the hyperbolic building. The rank of the tensor matches the number of sides of the polygon.
2.  **Network Connectivity**: The tensor network's structure is identical to the dual graph of the building. If two polygons share an edge, their corresponding tensors are contracted along that index.
3.  **Boundary State**: Uncontracted tensor indices at the edge of the geometry represent the physical degrees of freedom of the boundary quantum state. The entanglement structure of this state is directly dictated by the bulk geometry.
4.  **Entanglement Entropy**: The entanglement entropy of a boundary region `A` is computed using the **Ryu-Takayanagi formula**, which relates the entropy to the area of a minimal surface (a geodesic, `γ_A`) in the bulk that ends on the boundary of `A`.



## 🚀 Installation

You can install `holographic_tn` directly from PyPI:

```bash
pip install holographic_tn
```

To install from source for development, clone the repository and install in editable mode:

```bash
git clone [https://github.com/example/holographic_tn.git](https://github.com/example/holographic_tn.git)
cd holographic_tn
pip install -e .
```

### Dependencies

The library relies on the following major packages:
* `numpy`
* `scipy`
* `networkx`
* `quimb`
* `torch` (optional, for GPU support)
* `yastn`

## 💡 Quick Start

The following example demonstrates the core workflow: constructing a geometry, building a tensor network, defining a boundary region, and calculating its entanglement entropy.

```python
import numpy as np
from holographic_tn.geometry import HyperbolicBuilding
from holographic_tn.physics import build_network_from_building, calculate_rt_entropy
from holographic_tn.numerics.kpm.config import KPMConfig

def main():
    """
    A complete workflow for a holographic entanglement entropy calculation.
    """
    # 1. Initialize the geometric object with {p, v} parameters
    #    Here, we use p=5 (pentagons) and v=4 (4 pentagons meet at a vertex).
    print("Step 1: Constructing the hyperbolic building...")
    building = HyperbolicBuilding(p=5, v=4)
    building.construct_tiling(layers=3) # Build a few layers of the geometry
    print(f"  Constructed a building with {len(building.simplicial_complex.nodes())} faces.")

    # 2. Build the quantum state (a tensor network) on this geometry
    print("\nStep 2: Building the tensor network...")
    # Use 'cpu' for precision or 'gpu' for speed
    tn = build_network_from_building(building, compute_mode='cpu')

    # 3. Define a boundary region 'A'
    #    This corresponds to a set of uncontracted (physical) indices.
    print("\nStep 3: Defining a boundary region...")
    all_boundary_inds = sorted(list(tn.outer_inds()))
    # Let's select a contiguous block of 8 boundary sites for our region A
    region_A_inds = all_boundary_inds[:8]
    print(f"  Selected a region with {len(region_A_inds)} physical indices.")

    # 4. Configure and run the Ryu-Takayanagi calculation
    print("\nStep 4: Calculating entanglement entropy...")
    # Use the Kernel Polynomial Method (KPM) for efficiency
    # Use fewer moments/vectors for a quick example run
    kpm_config = KPMConfig(num_moments=150, num_vectors=20, bounds_method='fast')

    results = calculate_rt_entropy(
        building=building,
        tn=tn,
        boundary_region_inds=region_A_inds,
        config=kpm_config
    )

    # 5. Print the results
    print("\n✅ Calculation Complete!")
    print("---------------------------------")
    print(f"  Entanglement Entropy S(A):   {results['entropy']:.6f}")
    print(f"  Geodesic Length (Area):      {results['discrete_geodesic_length']}")
    print(f"  Number of Cut Bonds:         {results['cut_length']}")
    print("---------------------------------")
    print("\nVerifying the holographic principle:")
    print(f"S(A) ∝ Area  =>  {results['entropy']:.4f} ∝ {results['discrete_geodesic_length']}")


if __name__ == '__main__':
    main()

```

## 🛠️ Core Components

### `HyperbolicBuilding`

This is the central class for all geometric operations.

* `__init__(p, v)`: Initializes with the polygon type (`p`) and vertex configuration (`v`).
* `construct_tiling(layers)`: Generates a regular `{p, q}` tiling, where `q` is derived from `v`.
* `construct_building(side_pairings)`: Constructs more complex geometries from a fundamental domain and group generators.
* `find_geodesic_a_star(start, end)`: Finds the shortest path between two faces (nodes) in the building's graph.
* `identify_gromov_boundary(...)`: An algorithm to identify the "boundary at infinity" by clustering geodesic rays.

### Entropy Calculation Methods

You can control the physics calculation by passing a configuration object to `calculate_rt_entropy`.

* `KPMConfig`: Recommended for large systems. It avoids building the dense density matrix `ρ_A` by using a stochastic trace estimation with a Chebyshev polynomial expansion.
    * `num_moments`: Number of Chebyshev moments. Higher is more accurate but computationally expensive. A warning in the code suggests values beyond 250 can cause instability.
    * `num_vectors`: Number of random probe vectors. More vectors improve accuracy.
* `ExactConfig`: The default method. It performs a full tensor network contraction to get the dense matrix `ρ_A` and then diagonalizes it. Only feasible for very small test systems.
* `BBPConfig`: An experimental method using Block Belief Propagation for approximate contraction. It is designed to leverage the tree-like metric of the hyperbolic graph for a highly accurate approximation.

## 🗺️ Roadmap

The development of this library is planned in three phases:

1.  **Phase 1: The Geometric Kernel**: Implement and validate the `HyperbolicBuilding` class and all core geometric algorithms (geodesic finding, boundary identification).
2.  **Phase 2: Tensor Network Integration**: Integrate the `TensorNetwork` library to translate a `HyperbolicBuilding` object into a contractible quantum state.
3.  **Phase 3: The Physics Engine and Validation**: Implement the full Ryu-Takayanagi workflow and the advanced contraction algorithms. Validate results against known literature (e.g., the HaPPY code).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, feature requests, or suggestions.

## 📜 Citation

If you use `holographic_tn` in your research, please cite the original paper that inspired this work:

> "Holographic tensor networks from hyperbolic buildings" (placeholder link to paper).

And this software implementation:

> [Cormac Kinsella]. (2025). *holographic_tn: A Python Library for Holographic Tensor Networks on Hyperbolic Buildings*. [https://github.com/example/holographic_tn](https://github.com/example/holographic_tn).

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
