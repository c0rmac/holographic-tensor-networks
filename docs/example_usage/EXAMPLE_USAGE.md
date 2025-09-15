# Practical Guide: From Geometric Theory to Holographic Simulation

This document serves as a practical guide to using the holographic tensor network library. It directly connects the abstract geometric concepts outlined in the `THEORY.md` file to the concrete functions and workflows implemented in the source code. Our goal is to demonstrate how to translate the *why* of the theory into the *how* of a computational experiment.

## The Core Investigation: The Ryu-Takayanagi Formula

The central tool for our investigation is the **Ryu-Takayanagi formula**, which provides a quantitative link between geometry and quantum information. As stated in [`THEORY.md#holographic-connection`](https://www.google.com/search?q=THEORY.md%23holographic-connection), the formula relates the entanglement entropy $S_A$ of a boundary region $A$ to the area of a minimal surface $\gamma_A$ in the bulk:

$$ S_A = \frac{\text{Area}(\gamma_A)}{4G_N} $$

Our library is designed to be a "numerical laboratory" where you can construct the geometric bulk, define the quantum state on its boundary, and numerically test this powerful relationship.

---

## Your Guide to a Numerical Experiment

Here, we outline the step-by-step process of running a holographic simulation, linking each stage to the relevant theory and the specific code that implements it.

### Step 1: Building the Geometric Stage

To test our hypothesis, we first need a geometric "stage" for our simulation. As described in [`THEORY.md#poincare-disk`](https://www.google.com/search?q=THEORY.md%23poincare-disk), we use a discrete version of a hyperbolic space by creating a **tessellation** of the Poincaré Disk.

#### Example 1: A Simple, Uniform Universe

The most straightforward method is to generate a perfectly uniform tiling using `construct_tiling`. This is ideal for establishing a baseline or a "control group" for your experiment. The `HyperbolicBuilding` class in `geometry.py` is your primary tool for this:

```python
# Creates the graph representing a {5, 4} tiling
building = HyperbolicBuilding(p=5, v=4)

# Populates the graph with nodes and edges out to a specific number of layers
building.construct_tiling(layers=3)
```
```
# DIAGRAM_PLACEHOLDER: simple_tiling_construction
# Description: A diagram showing a simple {5, 4} pentagonal tiling emerging layer by layer from a central polygon.
```
*(Placeholder for a diagram illustrating the `construct_tiling` process.)*

#### Example 2: A Custom Universe with Non-Trivial Topology

For more advanced research, you can construct a custom manifold by defining a fundamental domain and a set of isometries that "glue" its sides together, as explained in [`THEORY.md#isometries`](https://www.google.com/search?q=THEORY.md%23isometries).

**Hypothesis:** A researcher might hypothesize that a two-sided wormhole geometry (a genus-2 surface) should create a specific, non-trivial entanglement pattern between its two boundaries.

To test this, you would construct the geometry by defining the side-pairings of an octagon:

```python
import numpy as np
from holographic_tn.geometry import HyperbolicBuilding, PoincareIsometry

# 1. Define the 8 vertices of a regular octagon in the Poincaré disk.
#    This requires some trigonometry, placing vertices on a circle of radius r < 1.
num_sides = 8
radius = 0.7
octagon_vertices = [
    (radius * np.cos(2 * np.pi * k / num_sides), radius * np.sin(2 * np.pi * k / num_sides))
    for k in range(num_sides)
]

# 2. Define the PoincareIsometry objects that identify opposite sides.
#    The researcher's intellectual input is designing these gluing rules.
#    These z0 and phi values are illustrative.
isometry_a1 = PoincareIsometry(name='a1', z0=(0.4, 0.1), phi=np.pi/8)
isometry_b1 = PoincareIsometry(name='b1', z0=(-0.1, 0.4), phi=np.pi/8)
isometry_a2 = PoincareIsometry(name='a2', z0=(-0.4, -0.1), phi=np.pi/8)
isometry_b2 = PoincareIsometry(name='b2', z0=(0.1, -0.4), phi=np.pi/8)

# Define the "gluing" instructions for the sides of the octagon.
# Side 0 is glued to side 4, side 1 to side 5, and so on.
side_pairings = {
    0: (4, isometry_a1),
    1: (5, isometry_b1),
    2: (6, isometry_a2),
    3: (7, isometry_b2),
}

# 3. Build the custom universe from this blueprint.
wormhole_geometry = HyperbolicBuilding(p=8, v=8)
wormhole_geometry.construct_building(
    fundamental_domain_vertices=octagon_vertices,
    side_pairings=side_pairings,
    layers=5
)
```
```
# DIAGRAM_PLACEHOLDER: custom_manifold_construction
# Description: A diagram showing an octagon (fundamental domain) with arrows indicating how its sides are identified by isometries to form a two-holed torus.
```
*(Placeholder for a diagram illustrating the `construct_building` process.)*


### Step 2: Creating the Quantum State

With the geometry established, the next step is to create the quantum state. As explained in [`THEORY.md#holographic-connection`](https://www.google.com/search?q=THEORY.md%23holographic-connection), this is done by placing a **perfect tensor** on each face of our geometric tiling and contracting them according to the adjacency of those faces. The function `build_network_from_building` in `physics.py` automates this entire process:

```python
# This function takes your geometric building...
# ...and returns a Quimb TensorNetwork object representing the quantum state.
tn = build_network_from_building(building, compute_mode='gpu')
```

### Step 3: Identifying the Physical Boundary

The tensor network at this stage represents the entire bulk. The uncontracted "dangling" indices on the edge of this network represent the physical degrees of freedom of the boundary state, which lives on the **[Gromov Boundary](https://www.google.com/search?q=THEORY.md%23gromov-boundary)**. It is a critical step to correctly identify these physical indices. The `geometry.py` module provides methods to algorithmically find this boundary:

```python
# Algorithmically find the indices corresponding to the Gromov boundary
boundary_indices = building.get_gromov_boundary()

# Or, for a simple tiling, get the indices from the outermost layer
boundary_indices = building.get_indices_from_layer(layer_index=-1)
```

### Step 4: Probing the State with the "Entanglement Scanner"

To test the Ryu-Takayanagi formula, you must calculate both sides of the equation. The main entry point for any physical calculation is the `calculate_rt_entropy` function in `physics.py`. It orchestrates the entire workflow:

1.  **Find Minimal Surface ($\gamma_A$):** Given a set of `boundary_indices` for your region A, the function uses the **A* search algorithm** to find the shortest path (the geodesic) between these endpoints through the bulk graph. This path is the discrete version of the minimal surface $\gamma_A$.

2.  **Compute Entanglement Entropy ($S_A$):** It then partitions the tensor network along this geodesic and computes the von Neumann entropy using a specified numerical method.

```
# DIAGRAM_PLACEHOLDER: rt_formula_in_action
# Description: A diagram showing the full RT process. A region 'A' on the boundary is selected, a geodesic path is found in the bulk, and this path cuts the tensor network bonds.
```
*(Placeholder for a diagram illustrating the full Ryu-Takayanagi calculation.)*

### Step 5: Choosing Your Computational Method

The final step, computing the entropy, is the most computationally intensive. The library offers several methods, which you can select by passing a configuration object. The recommended method for any reasonably large system is the **Kernel Polynomial Method (KPM)**, configured via the `KPMConfig` object.

```python
from src.holographic_tn.config import KPMConfig
from src.holographic_tn.physics import calculate_rt_entropy

# Select a contiguous region of 12 sites on the boundary to analyze
my_boundary_region = boundary_indices[:12]

# Configure the KPM method for a balance of speed and accuracy
kpm_config = KPMConfig(num_moments=200, num_vectors=20)

# Pass all components to the main function to get the final results
results = calculate_rt_entropy(
    building=building,
    tn=tn,
    boundary_region_inds=my_boundary_region,
    config=kpm_config
)

print(f"Entanglement Entropy S(A): {results['entropy']:.4f}")
print(f"Geodesic Length (Area):   {results['discrete_geodesic_length']}")
```