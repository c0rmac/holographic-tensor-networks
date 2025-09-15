import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# --- Utilities ---------------------------------------------------------------

def project_inside_disk(z, margin=1e-9):
    """Project z radially just inside the open unit disk if needed."""
    r = np.abs(z)
    if r >= 1.0 - margin:
        return z / (r + 1e-16) * (1.0 - margin)
    return z


# --- Geodesic construction in the Poincaré disk -----------------------------

def get_geodesic_arc(z1, z2, tol=1e-12):
    """
    Return (cx, cy, r) for the circle orthogonal to the unit circle passing through z1, z2.
    If origin, z1, z2 are collinear (diameter case), return None.
    """
    # Diameter (through origin) test: z1, z2 collinear with 0  <=>  Im(z1 * conj(z2)) = 0
    if np.isclose(z1.real * z2.imag - z1.imag * z2.real, 0.0, atol=tol):
        return None

    # Solve for circle center c = (cx, cy) using orthogonality and passing through z1, z2:
    # 2 Re(z_k * conj(c)) = |z_k|^2 + 1  for k=1,2
    A = 2 * np.array([[z1.real, z1.imag], [z2.real, z2.imag]])
    b = np.array([abs(z1) ** 2 + 1, abs(z2) ** 2 + 1])

    try:
        cx, cy = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    r_sq = cx ** 2 + cy ** 2 - 1.0
    if r_sq <= 0:
        return None

    return cx, cy, np.sqrt(r_sq)


def draw_geodesic(z1, z2, ax, clip_path=None, **kwargs):
    """
    Draw a hyperbolic geodesic (as a circular arc orthogonal to the boundary, or a diameter).
    """
    z1 = complex(project_inside_disk(z1))
    z2 = complex(project_inside_disk(z2))

    params = get_geodesic_arc(z1, z2)
    if params is None:
        # Diameter (Euclidean straight line between z1, z2)
        line, = ax.plot([z1.real, z2.real], [z1.imag, z2.imag], **kwargs)
        if clip_path is not None:
            line.set_clip_path(clip_path)
        return

    cx, cy, r = params

    # Compute arc angles relative to the circle center
    angle1 = np.angle(z1 - (cx + 1j * cy))
    angle2 = np.angle(z2 - (cx + 1j * cy))
    # Choose the shorter oriented arc between angle1 and angle2
    delta = (angle2 - angle1 + np.pi) % (2 * np.pi) - np.pi
    angle2 = angle1 + delta

    arc = patches.Arc(
        (cx, cy), 2 * r, 2 * r,
        theta1=np.degrees(angle1), theta2=np.degrees(angle2), **kwargs
    )
    ax.add_patch(arc)
    if clip_path is not None:
        arc.set_clip_path(clip_path)


# --- Disk isometries: automorphisms of the unit disk ------------------------

class SimplePoincareIsometry:
    """
    Orientation-preserving isometry of the unit disk:
      f(z) = e^{i φ} * (z - z0) / (1 - conj(z0) * z), with |z0| < 1.
    """
    def __init__(self, z0, phi=0.0, eps=1e-9):
        z0 = complex(z0)
        r = np.abs(z0)
        if r >= 1.0:
            # Project into the open unit disk, just inside
            z0 = z0 / (r + eps) * (1.0 - 1e-6)
        self.z0 = z0
        self.phi = float(phi)
        self.exp_iphi = np.exp(1j * self.phi)

    def transform(self, z):
        z = complex(z)
        w = self.exp_iphi * (z - self.z0) / (1.0 - np.conj(self.z0) * z)
        return project_inside_disk(w)

    def inverse(self):
        """
        The exact inverse:
          f^{-1}(w) = e^{-i φ} * (w + z0) / (1 + conj(z0) * w)
        """
        return InversePoincareIsometry(self.z0, self.phi)


class InversePoincareIsometry:
    """
    Inverse of SimplePoincareIsometry:
      f^{-1}(w) = e^{-i φ} * (w + z0) / (1 + conj(z0) * w)
    """
    def __init__(self, z0, phi):
        self.z0 = complex(z0)
        self.phi = float(phi)
        self.exp_minus_iphi = np.exp(-1j * self.phi)

    def transform(self, w):
        w = complex(w)
        z = self.exp_minus_iphi * (w + self.z0) / (1.0 + np.conj(self.z0) * w)
        return project_inside_disk(z)


# --- Diagram generators ------------------------------------------------------

def generate_isometry_labels_diagram():
    """
    Fundamental domain with side-pairing labels (gluing instructions).
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')

    boundary = patches.Circle((0, 0), 1, facecolor='lightcyan', edgecolor='black', lw=2, alpha=0.2, zorder=0)
    ax.add_patch(boundary)

    p = 8
    r0 = 0.7
    verts = [r0 * np.exp(1j * (2 * np.pi * k / p + np.pi / p)) for k in range(p)]
    verts = [project_inside_disk(v) for v in verts]

    polygon_fill = patches.Polygon([(v.real, v.imag) for v in verts], closed=True,
                                   facecolor='crimson', alpha=0.3, zorder=9)
    ax.add_patch(polygon_fill)
    polygon_fill.set_clip_path(boundary)

    for i in range(p):
        draw_geodesic(verts[i], verts[(i + 1) % p], ax, clip_path=boundary, color='crimson', lw=3, zorder=10)

    side_pairs = {0: 4, 1: 5, 2: 6, 3: 7}
    labels = ['a', 'b', 'c', 'd']
    colors = ['darkblue', 'darkgreen', 'purple', 'darkorange']

    for i, label in enumerate(labels):
        p1_idx, p2_idx = i, (i + 1) % p
        mid_point = (verts[p1_idx] + verts[p2_idx]) / 2
        mid_point = project_inside_disk(mid_point, margin=1e-6)

        ax.text(mid_point.real * 1.15, mid_point.imag * 1.15, f'${label}$',
                ha='center', va='center', fontsize=20, color=colors[i])

        ax.arrow(mid_point.real * 0.8, mid_point.imag * 0.8,
                 mid_point.real * 0.15, mid_point.imag * 0.15,
                 head_width=0.04, color=colors[i], length_includes_head=True)

        p1_inv_idx, p2_inv_idx = side_pairs[i], (side_pairs[i] + 1) % p
        mid_point_inv = (verts[p1_inv_idx] + verts[p2_inv_idx]) / 2
        mid_point_inv = project_inside_disk(mid_point_inv, margin=1e-6)

        ax.text(mid_point_inv.real * 1.15, mid_point_inv.imag * 1.15, f'${label}^{{-1}}$',
                ha='center', va='center', fontsize=20, color=colors[i])

        ax.arrow(mid_point_inv.real * 0.95, mid_point_inv.imag * 0.95,
                 -mid_point_inv.real * 0.15, -mid_point_inv.imag * 0.15,
                 head_width=0.04, color=colors[i], length_includes_head=True)

    ax.set_title('Fundamental Domain with Side-Pairing Isometries ("Gluing Rules")', fontsize=16, pad=10)
    filename = "fundamental_domain_labels.png"
    plt.savefig(filename, dpi=220, bbox_inches='tight')
    print(f"Image saved as {filename}")
    plt.show()


def generate_isometry_action_diagram():
    """
    Action of a single isometry: maps the fundamental domain to a neighboring tile.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')

    boundary = patches.Circle((0, 0), 1, facecolor='lightcyan', edgecolor='black', lw=2, alpha=0.2, zorder=0)
    ax.add_patch(boundary)

    # Regular {p, q} polygon centered at origin (embedded radius formula)
    p, q = 8, 8
    r0 = np.sqrt((np.cos(np.pi / p) - np.sin(np.pi / q)) / (np.cos(np.pi / p) + np.sin(np.pi / q)))
    verts = [r0 * np.exp(1j * (2 * np.pi * k / p + np.pi / p)) for k in range(p)]
    verts = [project_inside_disk(v) for v in verts]

    polygon_fill = patches.Polygon([(v.real, v.imag) for v in verts], closed=True,
                                   facecolor='crimson', alpha=0.3, zorder=9)
    ax.add_patch(polygon_fill)
    polygon_fill.set_clip_path(boundary)

    for i in range(p):
        draw_geodesic(verts[i], verts[(i + 1) % p], ax, clip_path=boundary, color='crimson', lw=3, zorder=10)

    # Choose an isometry that "pushes" across side [v0, v1], but remain inside the disk
    side_midpoint = (verts[0] + verts[1]) / 2
    direction = -side_midpoint if np.abs(side_midpoint) > 1e-12 else 1.0 + 0j
    z0 = direction / np.abs(direction) * 0.6  # magnitude strictly < 1
    isometry = SimplePoincareIsometry(z0=z0, phi=0.0)

    transformed_verts = [isometry.transform(v) for v in verts]

    for i in range(p):
        draw_geodesic(transformed_verts[i], transformed_verts[(i + 1) % p], ax,
                      clip_path=boundary, color='gray', lw=2, linestyle='--', zorder=5)

    ax.text(0, 0, "Fundamental Domain", ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.9), zorder=20)

    # Arrow indicating the action of the isometry on a sample point
    z_sample = 0 + 0.2j
    w_sample = isometry.transform(z_sample)
    arrow_start = (z_sample.real, z_sample.imag)
    arrow_end = (w_sample.real, w_sample.imag)
    arrow = patches.FancyArrowPatch(
        arrow_start, arrow_end,
        arrowstyle='simple,head_length=15,head_width=15,tail_width=5',
        connectionstyle="arc3,rad=0.3",
        color="darkblue", lw=2, zorder=15
    )
    ax.add_patch(arrow)
    arrow.set_clip_path(boundary)

    ax.text(0.12, 0.52, "Isometry 'a'", ha='center', fontsize=14, color='darkblue')

    # Label a vertex of the transformed polygon
    idx = 4
    ax.annotate("Transformed Image\n(Neighboring Tile)",
                xy=(transformed_verts[idx].real, transformed_verts[idx].imag),
                xytext=(0.5, -0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.9))

    ax.set_title("Action of Isometry 'a' on a Fundamental Domain", fontsize=16, pad=10)
    filename = "fundamental_domain_isometry_action.png"
    plt.savefig(filename, dpi=220, bbox_inches='tight')
    print(f"Image saved as {filename}")
    plt.show()


def generate_inverse_isometry_action_diagram():
    """
    Inverse isometry a^{-1}: map the neighboring tile back to the fundamental domain.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')

    boundary = patches.Circle((0, 0), 1, facecolor='lightcyan', edgecolor='black', lw=2, alpha=0.2, zorder=0)
    ax.add_patch(boundary)

    p, q = 8, 8
    r0 = np.sqrt((np.cos(np.pi / p) - np.sin(np.pi / q)) / (np.cos(np.pi / p) + np.sin(np.pi / q)))
    verts = [r0 * np.exp(1j * (2 * np.pi * k / p + np.pi / p)) for k in range(p)]
    verts = [project_inside_disk(v) for v in verts]

    # Original fundamental domain (red)
    polygon_fill = patches.Polygon([(v.real, v.imag) for v in verts], closed=True,
                                   facecolor='crimson', alpha=0.3, zorder=9)
    ax.add_patch(polygon_fill)
    polygon_fill.set_clip_path(boundary)
    for i in range(p):
        draw_geodesic(verts[i], verts[(i + 1) % p], ax, clip_path=boundary, color='crimson', lw=3, zorder=10)

    # Forward isometry a
    side_midpoint = (verts[0] + verts[1]) / 2
    direction = -side_midpoint if np.abs(side_midpoint) > 1e-12 else 1.0 + 0j
    z0 = direction / np.abs(direction) * 0.6
    isometry_a = SimplePoincareIsometry(z0=z0, phi=0.0)

    # Neighboring tile (dashed gray)
    transformed_verts = [isometry_a.transform(v) for v in verts]
    for i in range(p):
        draw_geodesic(transformed_verts[i], transformed_verts[(i + 1) % p], ax,
                      clip_path=boundary, color='gray', lw=2, linestyle='--', zorder=5)

    # Inverse isometry a^{-1}
    inv_a = isometry_a.inverse()

    # Arrow showing inverse action from neighbor back to origin
    w_sample = isometry_a.transform(0 + 0.1j)  # a(z)
    z_sample = inv_a.transform(w_sample)       # a^{-1}(a(z)) = z
    arrow_start = (w_sample.real, w_sample.imag)
    arrow_end = (z_sample.real, z_sample.imag)

    arrow = patches.FancyArrowPatch(
        arrow_start, arrow_end,
        arrowstyle='simple,head_length=15,head_width=15,tail_width=5',
        connectionstyle="arc3,rad=0.3",
        color="darkviolet", lw=2, zorder=15
    )
    ax.add_patch(arrow)
    arrow.set_clip_path(boundary)

    ax.text(0, 0, "End State\n(Fundamental Domain)", ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.9), zorder=20)
    ax.text(arrow_start[0] - 0.2, arrow_start[1], "Isometry 'a⁻¹'", ha='center', fontsize=14, color='darkviolet')

    ax.annotate("Start State\n(Neighboring Tile)",
                xy=arrow_start,
                xytext=(0.5, -0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.9))

    ax.set_title("Action of Inverse Isometry 'a⁻¹' (Return to Origin)", fontsize=16, pad=10)
    filename = "fundamental_domain_isometry_action_inverse.png"
    plt.savefig(filename, dpi=220, bbox_inches='tight')
    print(f"Image saved as {filename}")
    plt.show()


# --- Main -------------------------------------------------------------------

if __name__ == '__main__':
    generate_isometry_labels_diagram()
    generate_isometry_action_diagram()
    generate_inverse_isometry_action_diagram()
