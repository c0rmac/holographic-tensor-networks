import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# --- Helper functions for geometric constructions ---

def get_geodesic_arc(z1, z2):
    """Calculates center and radius of a geodesic arc passing through z1, z2."""
    if np.isclose(z1.real * z2.imag, z1.imag * z2.real): return None
    A = 2 * np.array([[z1.real, z1.imag], [z2.real, z2.imag]])
    b = np.array([abs(z1) ** 2 + 1, abs(z2) ** 2 + 1])
    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    cx, cy = center[0], center[1]
    r_sq = cx ** 2 + cy ** 2 - 1
    if r_sq < 0: return None
    r = np.sqrt(r_sq)
    return cx, cy, r


def draw_geodesic_arc(z1, z2, ax, **kwargs):
    """Draws a geodesic using the faster but less visually perfect Arc patch."""
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


def draw_geodesic_manual(z1, z2, ax, **kwargs):
    """Draws a geodesic by manually plotting points to avoid rendering artifacts."""
    params = get_geodesic_arc(z1, z2)
    if params is None:
        ax.plot([z1.real, z2.real], [z1.imag, z2.imag], **kwargs)
        return
    cx, cy, r = params
    angle1 = np.angle(z1 - (cx + 1j * cy))
    angle2 = np.angle(z2 - (cx + 1j * cy))
    delta = (angle2 - angle1 + np.pi) % (2 * np.pi) - np.pi

    num_points = 200
    angles = np.linspace(angle1, angle1 + delta, num_points)
    x_points = cx + r * np.cos(angles)
    y_points = cy + r * np.sin(angles)

    valid_mask = (x_points ** 2 + y_points ** 2) < 1.001
    ax.plot(x_points[valid_mask], y_points[valid_mask], **kwargs)


def generate_simple_divergence_diagram():
    """
    Generates a clean, simplified diagram of geodesic rays from a single point
    to illustrate divergence perfectly.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')

    boundary = patches.Circle((0, 0), 1, facecolor='lightcyan', edgecolor='black', lw=2, alpha=0.2, zorder=1)
    ax.add_patch(boundary)
    ax.text(0, 1.05, 'Boundary at Infinity', ha='center', fontsize=14)

    start_point_z = 0.3 - 0.2j
    ax.plot(start_point_z.real, start_point_z.imag, 'ro', markersize=12, zorder=10)

    num_rays = 12
    for i in range(num_rays):
        angle = (i / num_rays) * 2 * np.pi
        end_point_z = 1.0 * np.exp(1j * angle)
        draw_geodesic_manual(start_point_z, end_point_z, ax,
                             color=plt.cm.viridis(i / num_rays), lw=2, zorder=5)

    ax.annotate("Geodesic rays starting from a single point\nalways diverge from each other.",
                xy=(start_point_z.real, start_point_z.imag), xytext=(-0.8, 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.9))

    ax.set_title('Divergence of Geodesic Rays', fontsize=16, pad=10)
    filename = "geodesic_rays_simple_divergence.png"
    plt.savefig(filename)
    print(f"Image saved as {filename}")
    plt.show()


def generate_complex_convergence_diagram():
    """
    Generates the more complex diagram showing how geodesic rays from
    different points converge on the same boundary points.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')

    boundary = patches.Circle((0, 0), 1, facecolor='lightcyan', edgecolor='black', lw=2, alpha=0.2)
    ax.add_patch(boundary)
    ax.text(0, 1.05, 'Boundary at Infinity', ha='center', fontsize=14)

    start_point_1_z = -0.3 + 0.2j
    start_point_2_z = 0.4 - 0.1j

    ax.plot(start_point_1_z.real, start_point_1_z.imag, 'ro', markersize=12)
    ax.plot(start_point_2_z.real, start_point_2_z.imag, 'bo', markersize=12)

    num_rays = 12
    for i in range(num_rays):
        angle = (i / num_rays) * 2 * np.pi
        end_point_z = 1.0 * np.exp(1j * angle)
        draw_geodesic_arc(start_point_1_z, end_point_z, ax, color='crimson', lw=1.5, alpha=0.8)
        draw_geodesic_arc(start_point_2_z, end_point_z, ax, color='darkblue', lw=1.5, alpha=0.8)

    ax.annotate("Rays from a single point always diverge...",
                xy=(start_point_1_z.real, start_point_1_z.imag), xytext=(-0.8, 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.9))

    # --- FIX: Moved the marker to a visually clear convergence point in the bottom-right ---
    convergence_point_angle = 2 * np.pi * (11 / num_rays)  # Corresponds to 330 degrees
    convergence_point_z = 1.0 * np.exp(1j * convergence_point_angle)
    convergence_point = (convergence_point_z.real, convergence_point_z.imag)

    ax.plot(convergence_point[0], convergence_point[1], 'o',
            markersize=14, markerfacecolor='none', markeredgecolor='darkgreen',
            markeredgewidth=3, zorder=15)

    # Reposition the annotation to avoid overlap
    ax.annotate("...but rays from different points can\nconverge to the same 'point at infinity'.",
                xy=convergence_point, xytext=(0.7, -0.7),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.9))

    ax.set_title('Convergence of Geodesic Rays', fontsize=16, pad=10)
    filename = "geodesic_rays_complex_convergence.png"
    plt.savefig(filename)
    print(f"Image saved as {filename}")
    plt.show()


if __name__ == '__main__':
    generate_simple_divergence_diagram()
    generate_complex_convergence_diagram()

