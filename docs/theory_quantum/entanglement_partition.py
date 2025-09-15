import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def generate_entanglement_partition_diagram():
    """
    Generates and saves an improved diagram illustrating the partition of a
    quantum system, highlighting the entanglement bonds cut by the partition.
    """
    # --- Setup ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')

    # --- Draw a lattice of qubits and their entanglement bonds ---
    n_points = 11
    x, y = np.meshgrid(np.linspace(-1, 1, n_points), np.linspace(-1, 1, n_points))
    points = np.vstack([x.ravel(), y.ravel()]).T
    ax.scatter(points[:, 0], points[:, 1], color='gray', s=30, alpha=0.6, zorder=2)

    # Draw bonds between adjacent qubits
    for i in range(n_points):
        for j in range(n_points):
            p1 = (x[i, j], y[i, j])
            if i + 1 < n_points:
                p2 = (x[i + 1, j], y[i + 1, j])
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', lw=1, zorder=1)
            if j + 1 < n_points:
                p2 = (x[i, j + 1], y[i, j + 1])
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', lw=1, zorder=1)

    # --- Define Subsystem A ---
    region_A_radius = 0.5
    subsystem_A_circle = patches.Circle((0, 0), region_A_radius, facecolor='none', edgecolor='indianred', lw=2.5,
                                        linestyle='--', zorder=4)
    ax.add_patch(subsystem_A_circle)

    # Highlight qubits inside A
    in_A_mask = (points[:, 0] ** 2 + points[:, 1] ** 2) < region_A_radius ** 2
    ax.scatter(points[in_A_mask, 0], points[in_A_mask, 1], color='indianred', s=50, zorder=3,
               label='Qubits in Subsystem A')
    ax.scatter(points[~in_A_mask, 0], points[~in_A_mask, 1], color='darkgray', s=30, zorder=3,
               label='Qubits in Environment B')

    # --- Highlight the "cut" entanglement bonds ---
    cut_bonds = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, p2 = points[i], points[j]
            dist_sq = np.sum((p1 - p2) ** 2)
            # Check if points are adjacent
            if np.isclose(dist_sq, (2.0 / (n_points - 1)) ** 2):
                in1 = (p1[0] ** 2 + p1[1] ** 2) < region_A_radius ** 2
                in2 = (p2[0] ** 2 + p2[1] ** 2) < region_A_radius ** 2
                if in1 != in2:  # If one is in and one is out, the bond is cut
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='mediumseagreen', lw=2.5, zorder=3)
                    cut_bonds += 1

    if cut_bonds > 0:
        ax.plot([], [], color='mediumseagreen', lw=2.5, label='Cut Entanglement Bonds')

    # --- Add Labels ---
    ax.text(0, 0, 'A', ha='center', va='center', fontsize=30, color='indianred', alpha=0.9)
    ax.text(0.8, 0.8, 'B\n(Environment)', ha='center', va='center', fontsize=20, color='darkgray')
    title = 'Partitioning a Quantum System to Visualize Entanglement'
    fig.suptitle(title, fontsize=16)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=12, frameon=False)

    # --- Formula Annotation ---
    ax.text(0, -1.05, r'Entanglement entropy $S(A)$ quantifies the information shared across the cut.',
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="black", lw=1, alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # --- Save the figure ---
    filename = "entanglement_partition.png"
    plt.savefig(filename)
    print(f"Image saved as {filename}")
    plt.show()


if __name__ == '__main__':
    generate_entanglement_partition_diagram()

