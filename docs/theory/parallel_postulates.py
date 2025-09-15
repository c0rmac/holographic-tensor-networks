import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def get_geodesic_arc_params(p_z, endpoint_z):
    """
    Calculates the center and radius of a geodesic arc passing through a point p_z
    and ending at endpoint_z on the boundary of the Poincare disk.
    This is a robust implementation to avoid numerical instability.
    """
    # Convert points to complex numbers if they aren't already
    p = complex(p_z)
    e = complex(endpoint_z)

    # The inversion of p in the unit circle
    p_star = 1 / p.conjugate()

    # The center of the geodesic circle lies on the intersection of two
    # perpendicular bisectors: one for (p, e) and one for (p, p_star).
    # We solve the system of linear equations for the center (cx, cy).
    A = np.array([
        [2 * (p.real - e.real), 2 * (p.imag - e.imag)],
        [2 * (p.real - p_star.real), 2 * (p.imag - p_star.imag)]
    ])
    B = np.array([
        p.real ** 2 + p.imag ** 2 - (e.real ** 2 + e.imag ** 2),
        p.real ** 2 + p.imag ** 2 - (p_star.real ** 2 + p_star.imag ** 2)
    ])

    try:
        center = np.linalg.solve(A, B)
        cx, cy = center[0], center[1]
        radius = np.sqrt((p.real - cx) ** 2 + (p.imag - cy) ** 2)
        return cx, cy, radius
    except np.linalg.LinAlgError:
        # This occurs if the points are collinear, meaning the geodesic is a diameter.
        return None


def generate_parallel_postulates_diagram():
    """
    Generates and saves a diagram illustrating the parallel postulates,
    including a detailed hyperbolic example.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    ax1, ax2, ax3, ax4 = axes.flatten()
    fig.suptitle('The Parallel Postulate in Different Geometries', fontsize=20, y=0.95)

    # --- Common elements for the first three schematic plots ---
    line_l_schematic = [(-1.5, 0), (1.5, 0)]
    point_p_schematic = (0, 1)

    for ax in [ax1, ax2, ax3]:
        ax.plot([line_l_schematic[0][0], line_l_schematic[1][0]], [line_l_schematic[0][1], line_l_schematic[1][1]],
                'k-', lw=2, label='Line L')
        ax.plot(point_p_schematic[0], point_p_schematic[1], 'ro', markersize=10, label='Point P')
        ax.set_ylim(-0.5, 2)
        ax.set_xlim(-2, 2)
        ax.set_aspect('equal')
        ax.axis('off')

    # --- 1. Euclidean Geometry ---
    ax1.set_title('Euclidean (1 Parallel Line)', fontsize=16)
    ax1.plot([-1.5, 1.5], [1, 1], 'b--', lw=2)

    # --- 2. Spherical (Elliptic) Geometry ---
    ax2.set_title('Spherical (No Parallel Lines)', fontsize=16)
    arc1 = patches.Arc((0, -0.5), 4, 3, theta1=75, theta2=105, color='g', linestyle='--', lw=2)
    arc2 = patches.Arc((0, -0.5), 2, 1.5, theta1=65, theta2=115, color='g', linestyle='--', lw=2)
    ax2.add_patch(arc1)
    ax2.add_patch(arc2)
    ax2.text(0, -0.2, '(All lines intersect)', ha='center')

    # --- 3. Hyperbolic Geometry (Schematic) ---
    ax3.set_title('Hyperbolic (Infinite Parallel Lines)', fontsize=16)
    arc_p1 = patches.Arc((0, 2.5), 3, 3, theta1=240, theta2=300, color='b', linestyle='--', lw=2)
    arc_p2 = patches.Arc((0, 4), 6, 6, theta1=250, theta2=290, color='b', linestyle='--', lw=2)
    ax3.add_patch(arc_p1)
    ax3.add_patch(arc_p2)
    ax3.plot([0, -2], [1, -0.5], 'orange', linestyle=':', lw=2)
    ax3.plot([0, 2], [1, -0.5], 'orange', linestyle=':', lw=2)
    ax3.text(0, -0.4,
             "The orange lines are 'limiting parallels' that meet L at infinity.\nInfinitely many other parallel lines fit between them.",
             ha='center', va='top', fontsize=10, wrap=True)

    # --- 4. Hyperbolic Geometry (Poincaré Disk Detail) ---
    ax4.set_title('Poincaré Disk', fontsize=16)
    ax4.set_aspect('equal')
    ax4.set_xlim(-1.1, 1.1)
    ax4.set_ylim(-1.1, 1.1)
    ax4.axis('off')

    boundary = patches.Circle((0, 0), 1, facecolor='lightcyan', edgecolor='black', lw=2, alpha=0.2)
    ax4.add_patch(boundary)

    ax4.plot([-1, 1], [0, 0], color='k', lw=2)
    ax4.text(0, -0.1, 'Line L', ha='center', color='k')

    p_z = 0 + 0.5j
    ax4.plot(p_z.real, p_z.imag, 'ro', markersize=10)
    ax4.text(p_z.real + 0.08, p_z.imag, 'P', color='r', fontsize=14)

    def draw_hyperbolic_line(p_z, endpoint_z, ax, **kwargs):
        params = get_geodesic_arc_params(p_z, endpoint_z)
        if params is None: return  # Should not happen for off-axis point

        cx, cy, r = params
        center = cx + 1j * cy

        angle1 = np.angle(p_z - center)
        angle2 = np.angle(endpoint_z - center)

        # --- ROBUST FIX for drawing the correct arc segment ---
        delta = (angle2 - angle1 + np.pi) % (2 * np.pi) - np.pi
        angle2_corr = angle1 + delta

        # Swap angles if necessary to ensure drawing direction is correct
        start_angle, end_angle = np.degrees([angle1, angle2_corr])
        if start_angle > end_angle:
            start_angle, end_angle = end_angle, start_angle

        # If the angle span is > 180, we have the wrong arc segment.
        if end_angle - start_angle > 180:
            start_angle, end_angle = end_angle, start_angle + 360

        ax.add_patch(patches.Arc((cx, cy), 2 * r, 2 * r, theta1=start_angle, theta2=end_angle, **kwargs))

    # --- Draw Limiting Parallels (in orange) ---
    for endpoint in [-1 + 0j, 1 + 0j]:
        draw_hyperbolic_line(p_z, endpoint, ax4, edgecolor='orange', lw=2, linestyle=':')

    # --- Draw Ultra-Parallels (in blue) ---
    ultra_parallel_endpoints = [np.exp(1j * theta) for theta in [np.pi / 4, np.pi / 6, 3 * np.pi / 4, 5 * np.pi / 6]]
    for endpoint in ultra_parallel_endpoints:
        draw_hyperbolic_line(p_z, endpoint, ax4, edgecolor='b', lw=2, linestyle='--')

    ax4.text(0, -1.25,
             "This disk is a projection of the infinite hyperbolic plane.\nThe orange lines are 'limiting parallels'; the blue are 'ultra-parallels'.",
             ha='center', va='top', fontsize=10, wrap=True)

    # --- Save the figure ---
    filename = "parallel_postulates_with_detail.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Image saved as {filename}")
    plt.show()


if __name__ == '__main__':
    generate_parallel_postulates_diagram()

