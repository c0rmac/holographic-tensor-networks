import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# We assume the user's library is available in the path.
# This imports the necessary class to build the geometry.
from src.holographic_tn.geometry import HyperbolicBuilding


def generate_rt_diagram():
    """
    Generates and saves an improved diagram illustrating the Ryu-Takayanagi
    formula by using the HyperbolicBuilding class to construct the geometry
    and networkx for visualization.
    """
    # --- Setup ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    # --- 1. Construct the Geometry using the Library ---
    # This is the robust approach inspired by your example file.
    # We create an instance of the building and generate a {5, 4} tiling.
    print("Constructing {5, 4} hyperbolic tiling...")
    building = HyperbolicBuilding(p=5, v=4)
    layers = 3
    building.construct_tiling(layers=layers)

    # Extract the graph and node positions directly from the building object.
    dual_graph = building.simplicial_complex
    pos = building.face_centers

    # --- 2. Draw the Full Tiling (the Bulk Geometry) ---
    # Draw the boundary circle of the Poincaré disk.
    boundary_circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=1, zorder=1)
    ax.add_artist(boundary_circle)

    # Use networkx to draw the graph of the tiling.
    nx.draw_networkx(
        dual_graph,
        pos,
        ax=ax,
        with_labels=False,
        node_color='lightblue',
        edge_color='gray',
        alpha=0.7,
        node_size=50
    )

    # --- 3. Define the Boundary Region A ---
    # Find all nodes in the outermost layer.
    outer_nodes = [n for n, l in building.layers.items() if l == layers]
    # Sort them by angle to get a contiguous block.
    outer_nodes.sort(key=lambda n: np.angle(complex(*pos[n])))

    # Select a symmetric slice of nodes around the horizontal axis for a cleaner geodesic.
    num_boundary_nodes = 16
    half_slice = num_boundary_nodes // 2
    boundary_A_nodes = outer_nodes[-half_slice:] + outer_nodes[:half_slice]

    # Highlight the nodes of region A.
    nodes_A_collection = nx.draw_networkx_nodes(
        dual_graph, pos, nodelist=boundary_A_nodes, ax=ax,
        node_color='indianred', node_size=100,
        label='Boundary Region A'
    )
    # Set z-order manually on the returned artist object
    if nodes_A_collection:
        nodes_A_collection.set_zorder(3)

    ax.text(0, 1.05, 'Boundary Region A', ha='center', va='center', fontsize=14, color='indianred')

    # --- 4. Find and Draw the Minimal Surface (Geodesic) ---
    # Identify the geometric endpoints of region A.
    start_node, end_node = boundary_A_nodes[0], boundary_A_nodes[-1]

    # Use the library's A* algorithm to find the shortest path.
    print(f"Finding geodesic between {start_node} and {end_node}...")
    geodesic_path = building.find_geodesic_a_star(start_node, end_node)

    if geodesic_path:
        # Create a graph for just the geodesic path to draw its edges.
        geodesic_graph = nx.path_graph(geodesic_path)

        # Highlight the geodesic path (nodes and edges).
        geodesic_nodes_collection = nx.draw_networkx_nodes(
            dual_graph, pos, nodelist=geodesic_path, ax=ax,
            node_color='cornflowerblue', node_size=70
        )
        if geodesic_nodes_collection:
            geodesic_nodes_collection.set_zorder(4)

        geodesic_edges_collection = nx.draw_networkx_edges(
            geodesic_graph, pos, ax=ax,
            edge_color='cornflowerblue', width=3.0, style='--'
        )
        if geodesic_edges_collection:
            geodesic_edges_collection.set_zorder(4)

        # Dynamically position the annotation at the center of the geodesic path.
        mid_point_index = len(geodesic_path) // 2
        mid_node = geodesic_path[mid_point_index]
        text_pos_x, text_pos_y = pos[mid_node]

        ax.text(text_pos_x, text_pos_y + 0.1, r'Minimal Surface $\gamma_A$',
                ha='center', va='center', fontsize=14, color='cornflowerblue',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # --- 5. Add Final Labels and Title ---
    ax.text(0, -1.1, r'$S(A) = \frac{\mathrm{Area}(\gamma_A)}{4G_N}$', ha='center', va='center', fontsize=20,
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="black", lw=1, alpha=0.9), zorder=5)
    title = 'The Ryu-Takayanagi Formula: Entanglement from Geometry'
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.2, 1.1)

    # --- Save the figure ---
    filename = "ryu_takayanagi_formula.png"
    plt.savefig(filename, dpi=150)
    print(f"Image saved as {filename}")
    plt.show()


if __name__ == '__main__':
    # This assumes `geometry.py` is in the same directory or in the python path.
    generate_rt_diagram()

