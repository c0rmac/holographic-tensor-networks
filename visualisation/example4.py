import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque


def poincare_transform(z, c):
    """Applies a Möbius transformation to place a point in the Poincaré disk."""
    return (z + c) / (1 + np.conj(c) * z)


def generate_happy_code_graph(layers=3):
    """
    Generates the dual graph for a patch of the {5,4} pentagonal tiling (HaPPY code).
    """
    G = nx.Graph()
    pos = {0: np.array([0, 0])}
    layers_dict = {0: 0}

    q = deque([(0, 0 + 0j, 0)])
    visited_coords = {0 + 0j}
    node_counter = 1

    d = np.arccosh(1 / (np.tan(np.pi / 5) * np.tan(np.pi / 4)))
    c_base = np.tanh(d / 2)

    while q:
        parent_id, parent_z, layer = q.popleft()

        if layer >= layers:
            continue

        for i in range(5):
            angle = (2 * np.pi / 5) * i
            if parent_z != 0:
                parent_angle = np.angle(parent_z)
                angle += parent_angle

            c = c_base * np.exp(1j * angle)
            child_z = poincare_transform(parent_z, -c)

            is_new = True
            for vc in visited_coords:
                if np.linalg.norm(child_z - vc) < 1e-4:
                    is_new = False
                    break

            if is_new:
                child_pos = np.array([child_z.real, child_z.imag])
                G.add_node(node_counter)
                pos[node_counter] = child_pos
                layers_dict[node_counter] = layer + 1
                G.add_edge(parent_id, node_counter)

                q.append((node_counter, child_z, layer + 1))
                visited_coords.add(child_z)
                node_counter += 1

    return G, pos, layers_dict


def plot_scenario(ax, G, pos, layers_dict, boundary_region, title):
    """Plots a single scenario (highly or minimally entangled)."""
    # Find and calculate the geodesic
    start_node = boundary_region[0]
    end_node = boundary_region[-1]

    # Check if a path exists
    if nx.has_path(G, source=start_node, target=end_node):
        geodesic_path = nx.shortest_path(G, source=start_node, target=end_node)
        geodesic_length = len(geodesic_path) - 1
        # Entropy is proportional to length, log(bond_dim) factor assumed as log(2)
        entropy = geodesic_length * np.log(2)
    else:
        # Handle cases where boundary is too small for a clear path or graph is disconnected
        geodesic_path = []
        geodesic_length = 0
        entropy = 0.0
        print(f"Warning: No geodesic path found for '{title}' between {start_node} and {end_node}.")

    geodesic_graph = nx.path_graph(geodesic_path)

    # Plotting
    boundary_circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=1)
    ax.add_artist(boundary_circle)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=30)

    # Highlight Boundary Region 'A'
    nx.draw_networkx_nodes(G, pos, nodelist=boundary_region, ax=ax, node_color='orange', node_size=100)

    # Highlight the Geodesic Path
    nx.draw_networkx_edges(geodesic_graph, pos, ax=ax, edge_color='red', width=2.5)
    nx.draw_networkx_nodes(G, pos, nodelist=geodesic_path, ax=ax, node_color='red', node_size=60)

    ax.set_title(title, fontsize=14)
    ax.text(0, -1.25, f"Computed: Geodesic Length = {geodesic_length}, Entropy S(A) = {entropy:.3f}",
            ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.35, 1.1)
    ax.axis('off')


# --- 1. Generate the Geometry ---
layers = 3  # Less dense graph
G, pos, layers_dict = generate_happy_code_graph(layers)

# --- 2. Create Figure for Both Scenarios ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# --- 3. Scenario 1: Highly Entangled Region ---
outer_nodes_1 = [n for n, l in layers_dict.items() if l == layers]
outer_nodes_1.sort(key=lambda n: np.angle(pos[n][0] + 1j * pos[n][1]))
boundary_A_large = outer_nodes_1[:len(outer_nodes_1) // 2]  # A larger segment
plot_scenario(ax1, G, pos, layers_dict, boundary_A_large, "Highly Entangled Region")

# --- 4. Scenario 2: Minimally Entangled Region (Slightly more entangled) ---
outer_nodes_2 = [n for n, l in layers_dict.items() if l == layers]
outer_nodes_2.sort(key=lambda n: np.angle(pos[n][0] + 1j * pos[n][1]))
boundary_A_small = outer_nodes_2[:4]  # 4 nodes for a slightly more visible region
plot_scenario(ax2, G, pos, layers_dict, boundary_A_small, "Moderately Entangled Region")

plt.suptitle("HaPPY Code: Geometrically Faithful Visualization", fontsize=20, y=0.98)
plt.savefig("happy_code_comparison_visualization_less_dense.png")

print("Visualization saved as 'happy_code_comparison_visualization_less_dense.png'")