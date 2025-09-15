import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import deque


def get_geodesic_arc(z1, z2):
    """
    Calculates the center and radius of a geodesic arc (orthogonal circle)
    passing through two points z1 and z2 in the Poincaré disk.
    This is a robust method based on solving a linear system.
    """
    if np.isclose(z1.real * z2.imag, z1.imag * z2.real):  # Points are collinear with origin
        return None  # Represents a diameter, handled separately

    A = 2 * np.array([[z1.real, z1.imag], [z2.real, z2.imag]])
    b = np.array([abs(z1) ** 2 + 1, abs(z2) ** 2 + 1])

    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None  # Matrix is singular

    cx, cy = center[0], center[1]
    r_sq = cx ** 2 + cy ** 2 - 1
    if r_sq < 0: return None
    r = np.sqrt(r_sq)

    return cx, cy, r


def draw_geodesic(z1, z2, ax, **kwargs):
    """Draws a geodesic line (arc or diameter) between two points."""
    params = get_geodesic_arc(z1, z2)
    if params is None:
        ax.plot([z1.real, z2.real], [z1.imag, z2.imag], **kwargs)
        return
    cx, cy, r = params
    angle1 = np.angle(z1 - (cx + 1j * cy))
    angle2 = np.angle(z2 - (cx + 1j * cy))
    delta = (angle2 - angle1 + np.pi) % (2 * np.pi) - np.pi
    angle2 = angle1 + delta
    arc = patches.Arc((cx, cy), 2 * r, 2 * r, theta1=np.degrees(angle1), theta2=np.degrees(angle2), **kwargs)
    ax.add_patch(arc)


def reflect_point_in_circle(p_z, circle_center_z, circle_radius):
    """Reflects a point p_z across a circle using inversion."""
    p_rel = p_z - circle_center_z
    if np.isclose(np.conj(p_rel), 0): return np.inf
    return circle_radius ** 2 / np.conj(p_rel) + circle_center_z


def generate_simple_pentagon_diagram():
    """
    Generates a simplified, clean diagram of a single hyperbolic pentagon,
    focusing on core, correctly rendered features.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')

    boundary = patches.Circle((0, 0), 1, facecolor='lightcyan', edgecolor='black', lw=2, alpha=0.2)
    ax.add_patch(boundary)

    p, q = 5, 4
    r0 = np.sqrt((np.cos(np.pi / p) - np.sin(np.pi / q)) / (np.cos(np.pi / p) + np.sin(np.pi / q)))
    verts = [r0 * np.exp(1j * (2 * np.pi * k / p + np.pi / p)) for k in range(p)]

    # Draw the central pentagon's geodesic sides
    for i in range(p):
        draw_geodesic(verts[i], verts[(i + 1) % p], ax, color='crimson', lw=3, zorder=10)

    # Label the 90-degree interior angles
    for i in range(p):
        pos = verts[i] * 0.95
        ax.text(pos.real, pos.imag, '90°', ha='center', va='center',
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8), zorder=11)

    # --- Simplified Annotations ---
    ax.annotate("Red Lines: Geodesic sides of the pentagon.\nThese are the 'straight lines' of hyperbolic space.",
                xy=(verts[2].real, verts[2].imag), xytext=(0.8, -0.6),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5, connectionstyle="arc3,rad=0.2"),
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.9))

    ax.annotate(
        "The negative curvature of the space allows\nthe sides to bend inwards, making each\ninterior angle of the pentagon exactly 90°.",
        xy=(verts[4].real, verts[4].imag), xytext=(-0.8, 0.8),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5, connectionstyle="arc3,rad=0.2"),
        ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.9))

    ax.set_title('A Single {5, 4} Pentagon in Hyperbolic Space', fontsize=16, pad=10)
    filename = "poincare_pentagon_detail.png"
    plt.savefig(filename)
    print(f"Image saved as {filename}")
    plt.show()


def generate_full_tessellation_diagram():
    """
    Generates a diagram showing the pentagon and its neighbors.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')

    boundary = patches.Circle((0, 0), 1, facecolor='lightcyan', edgecolor='black', lw=2, alpha=0.2)
    ax.add_patch(boundary)

    p, q = 5, 4
    r0 = np.sqrt((np.cos(np.pi / p) - np.sin(np.pi / q)) / (np.cos(np.pi / p) + np.sin(np.pi / q)))
    central_verts = [r0 * np.exp(1j * (2 * np.pi * k / p + np.pi / p)) for k in range(p)]

    # --- BFS Generation for a robust tiling ---
    queue = deque([(central_verts, 0)])
    visited_keys = {tuple(round(v.real, 5) + 1j * round(v.imag, 5) for v in central_verts)}
    max_layers = 2

    while queue:
        verts, layer = queue.popleft()

        is_central = (layer == 0)
        color = 'crimson' if is_central else 'gray'
        lw = 3 if is_central else 1.5
        for i in range(p):
            draw_geodesic(verts[i], verts[(i + 1) % p], ax, color=color, lw=lw, zorder=10 - layer)

        if layer >= max_layers:
            continue

        for i in range(p):
            z1, z2 = verts[i], verts[(i + 1) % p]
            params = get_geodesic_arc(z1, z2)
            if params:
                cx, cy, r = params
                reflected_verts = [reflect_point_in_circle(v, cx + 1j * cy, r) for v in verts]
                key = tuple(round(v.real, 5) + 1j * round(v.imag, 5) for v in reflected_verts)

                if key not in visited_keys:
                    visited_keys.add(key)
                    queue.append((reflected_verts, layer + 1))

    # --- Annotations ---
    vertex_to_highlight = central_verts[0]
    highlight = patches.Circle((vertex_to_highlight.real, vertex_to_highlight.imag),
                               radius=0.08, facecolor='yellow', edgecolor='black', alpha=0.7, zorder=12)
    ax.add_patch(highlight)

    ax.annotate("Central Pentagon (Red)",
                xy=(0, 0), xytext=(0.5, 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.9))

    ax.annotate("{5, 4} Vertex Rule:\nEach vertex is shared by 4 pentagons.",
                xy=(vertex_to_highlight.real, vertex_to_highlight.imag),
                xytext=(-0.8, 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.9))

    ax.set_title('A {5, 4} Tessellation and its Neighborhood', fontsize=16, pad=10)
    filename = "poincare_tessellation_neighborhood.png"
    plt.savefig(filename)
    print(f"Image saved as {filename}")
    plt.show()


if __name__ == '__main__':
    generate_simple_pentagon_diagram()
    generate_full_tessellation_diagram()

