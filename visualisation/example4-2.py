import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from collections import deque

# --- Import the corrected classes from your local library file ---
from src.holographic_tn.geometry import HyperbolicBuilding
from src.holographic_tn.numerics.kpm.config import KPMConfig
from visualisation.example4 import boundary_A_large

# --- Control Flag ---
# Set to True for a fast geometric estimate of entropy.
# Set to False to run the full, computationally intensive tensor network calculation.
USE_FAST_ESTIMATION = False


def get_rt_data(building, tn, face_id_to_tid_map, boundary_nodes, use_estimation=True, custom_entropy=None, geometric_geodesic_length=None):
    """Gets Ryu-Takayanagi data either by fast estimation or full computation."""
    if not boundary_nodes or len(boundary_nodes) < 2:
        return {'geodesic_path': [], 'length': 0, 'entropy': 0.0, 'label': 'Estimated'}

    start_node, end_node = boundary_nodes[0], boundary_nodes[-1]

    if use_estimation or (custom_entropy is not None and geometric_geodesic_length is not None):
        geodesic_path = building.find_geodesic_a_star(start_node, end_node)
        length = geometric_geodesic_length if geometric_geodesic_length is not None else (len(geodesic_path) - 1 if geodesic_path else 0)
        entropy = custom_entropy if custom_entropy is not None else length * np.log(2)
        return {'geodesic_path': geodesic_path, 'length': length, 'entropy': entropy, 'label': 'Estimated'}
    else:
        # Full calculation requires importing the physics module
        from src.holographic_tn.physics import calculate_rt_entropy

        boundary_tags = {str(face_id) for face_id in boundary_nodes}
        boundary_inds = [ind for ind in tn.outer_inds() if
                         any(tag in tn.tensor_map[next(iter(tn.ind_map[ind]))].tags for tag in boundary_tags)]

        results = calculate_rt_entropy(
            building=building, tn=tn, boundary_region_inds=boundary_inds, config=KPMConfig()
        )
        return {'geodesic_path': results['geodesic_path'], 'length': results['geometric_geodesic_length'],
                'entropy': results['entropy'], 'label': 'Computed'}


def plot_scenario(ax, building, tn, face_id_to_tid_map, boundary_nodes, title, custom_entropy=None, geometric_geodesic_length=None):
    """Plots a single scenario."""
    results = get_rt_data(building, tn, face_id_to_tid_map, boundary_nodes, use_estimation=USE_FAST_ESTIMATION, custom_entropy=custom_entropy, geometric_geodesic_length=geometric_geodesic_length)

    geodesic_path = results['geodesic_path']
    length = results['length']
    entropy = results['entropy']
    label = results['label']

    pos = building.face_centers
    dual_graph = building.simplicial_complex
    geodesic_graph = nx.path_graph(geodesic_path) if geodesic_path else nx.Graph()

    boundary_circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=1)
    ax.add_artist(boundary_circle)
    nx.draw_networkx(dual_graph, pos, ax=ax, with_labels=False, node_color='lightblue', edge_color='gray', alpha=0.7,
                     node_size=50)
    nx.draw_networkx_nodes(dual_graph, pos, nodelist=boundary_nodes, ax=ax, node_color='orange', node_size=150)
    nx.draw_networkx_edges(geodesic_graph, pos, ax=ax, edge_color='red', width=3.0)
    nx.draw_networkx_nodes(dual_graph, pos, nodelist=geodesic_path, ax=ax, node_color='red', node_size=100)

    ax.set_title(title, fontsize=14)
    ax.text(0, -1.25, f"{label}: Geodesic Length = {length:.1f}, Entropy S(A) = {entropy:.3f}",
            ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    ax.set_aspect('equal');
    ax.set_xlim(-1.1, 1.1);
    ax.set_ylim(-1.35, 1.1);
    ax.axis('off')


def main():
    start_time = time.time()

    print(f"Running visualization with USE_FAST_ESTIMATION = {USE_FAST_ESTIMATION}")

    print("Initializing and building geometry...")
    building = HyperbolicBuilding(p=5, v=4)
    layers = 3
    building.construct_tiling(layers=layers)

    tn = None
    face_id_to_tid_map = None
    if not USE_FAST_ESTIMATION:
        from src.holographic_tn.physics import build_network_from_building
        print("Constructing the tensor network...")
        tn = build_network_from_building(building, compute_mode='cpu')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    outer_nodes = [n for n, l in building.layers.items() if l == layers]
    outer_nodes.sort(key=lambda n: np.angle(complex(*building.face_centers[n])))

    print("Processing 'Highly Entangled Region'...")
    # boundary_A_large = random.sample(outer_nodes, 10)
    boundary_A_large = outer_nodes[:2]
    plot_scenario(ax1, building, tn, face_id_to_tid_map, boundary_A_large, "Highly Entangled Region")

    print("Processing 'Moderately Entangled Region'...")
    boundary_A_small = outer_nodes[2: 5]
    plot_scenario(ax2, building, tn, face_id_to_tid_map, boundary_A_small, "Moderately Entangled Region")

    fig.suptitle("HaPPY Code: Holographic Visualization", fontsize=20, y=0.98)
    plt.savefig("happy_code_final_visualization.png", dpi=150)

    end_time = time.time()
    print(f"\nVisualization saved as 'happy_code_final_visualization.png'")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()