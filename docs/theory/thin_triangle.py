"""
thin_triangle_consequence_demo.py

Left: thin triangle in the Poincaré disk (as before).
Right: practical demonstration:
  - Top: Euclidean grid with a large square cycle and a long detour along the cycle.
  - Bottom: Hyperbolic / tree-like expansion where the analogous detour is short,
    illustrating how δ-thinness prevents large grid-like loops.

Saves thin_triangle_consequence_with_demo.png.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------- Poincaré helpers (unchanged) -----------------

def project_inside_disk(z, margin=1e-9):
    z = complex(z)
    r = abs(z)
    if r >= 1.0 - margin:
        return z / (r + 1e-16) * (1.0 - margin)
    return z

def poincare_distance(z, w):
    z, w = complex(z), complex(w)
    num = 2 * abs(z - w) ** 2
    den = (1 - abs(z) ** 2) * (1 - abs(w) ** 2)
    arg = 1 + num / (den + 1e-16)
    return np.arccosh(np.clip(arg, 1.0, None))

def get_geodesic_circle(z1, z2, tol=1e-12):
    z1, z2 = complex(z1), complex(z2)
    if np.isclose(z1.real * z2.imag - z1.imag * z2.real, 0.0, atol=tol):
        return None
    A = 2 * np.array([[z1.real, z1.imag], [z2.real, z2.imag]])
    b = np.array([abs(z1) ** 2 + 1.0, abs(z2) ** 2 + 1.0])
    try:
        cx, cy = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    r_sq = cx*cx + cy*cy - 1.0
    if r_sq <= 0:
        return None
    return cx, cy, np.sqrt(r_sq)

def sample_geodesic(z1, z2, n=240):
    z1, z2 = complex(z1), complex(z2)
    params = get_geodesic_circle(z1, z2)
    if params is None:
        t = np.linspace(0, 1, n)
        pts = z1 * (1 - t) + z2 * t
        return [project_inside_disk(z) for z in pts]
    cx, cy, r = params
    center = cx + 1j * cy
    a1 = np.angle(z1 - center)
    a2 = np.angle(z2 - center)
    delta = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
    thetas = np.linspace(a1, a1 + delta, n)
    pts = center + r * np.exp(1j * thetas)
    return [project_inside_disk(z) for z in pts]


# ----------------- Left thin-triangle plot (unchanged) -----------------

def draw_thin_triangle(ax, verts, delta=0.6, pnts_per_side=400):
    # unit circle
    ax.add_patch(patches.Circle((0,0), 1, edgecolor='black', facecolor='white', lw=1.2, zorder=0))
    # sample sides
    arcs = [sample_geodesic(verts[i], verts[(i+1)%3], n=pnts_per_side) for i in range(3)]
    # compute masked thin sets
    masks = []
    for i in range(3):
        side = arcs[i]
        other = arcs[(i+1)%3] + arcs[(i+2)%3]
        min_d = []
        for z in side:
            dists = [poincare_distance(z, w) for w in other]
            min_d.append(min(dists))
        masks.append(np.array(min_d) <= delta)
    colors = ['crimson','darkorange','darkgreen']
    for arc in arcs:
        ax.plot([z.real for z in arc], [z.imag for z in arc], color='lightgray', lw=1.0, zorder=3)
    for arc, mask, col in zip(arcs, masks, colors):
        pts = np.array([[z.real,z.imag] for z in arc])
        ax.scatter(pts[mask,0], pts[mask,1], s=6, color=col, alpha=0.9, zorder=6)
    for arc, col in zip(arcs, colors):
        pts = np.array([[z.real,z.imag] for z in arc])
        ax.plot(pts[:,0], pts[:,1], color=col, lw=2.0, zorder=8)
    # sample point on side 0 and nearest
    sample_side = arcs[0]
    sample_idx = len(sample_side)//3
    sample_point = sample_side[sample_idx]
    other_all = arcs[1] + arcs[2]
    dists = np.array([poincare_distance(sample_point, w) for w in other_all])
    nearest_point = other_all[int(np.argmin(dists))]
    connector = sample_geodesic(sample_point, nearest_point, n=120)
    ax.plot([z.real for z in connector], [z.imag for z in connector], color='black', lw=1.0, linestyle='--', zorder=11)
    ax.scatter([sample_point.real],[sample_point.imag], color='black', s=40, zorder=12)
    ax.scatter([nearest_point.real],[nearest_point.imag], color='blue', s=40, zorder=12)
    ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("δ‑thin triangle (Poincaré disk)", fontsize=10)
    return sample_point, nearest_point


# ----------------- Right demonstration: grid vs tree -----------------

def draw_grid_with_cycle(ax, grid_size=6, cycle_size=4):
    """
    Draw a square grid and highlight a large cycle (square) of side length cycle_size.
    Draw two points on opposite mid-edges and show the long cycle path between them.
    """
    # build integer grid points
    xs = np.arange(grid_size)
    ys = np.arange(grid_size)
    for i in xs:
        for j in ys:
            ax.plot(i, j, 'ko', markersize=3, zorder=2)
    # grid lines
    for i in xs:
        ax.plot([i]*grid_size, ys, color='lightgray', lw=1, zorder=1)
    for j in ys:
        ax.plot(xs, [j]*grid_size, color='lightgray', lw=1, zorder=1)

    # choose a large square cycle in center
    s = cycle_size
    x0 = (grid_size - s)//2
    y0 = (grid_size - s)//2
    cycle = [(x0+k, y0) for k in range(s)] + [(x0+s-1, y0+k) for k in range(1,s)] + \
            [(x0+s-1-k, y0+s-1) for k in range(1,s)] + [(x0, y0+s-1-k) for k in range(1,s)]
    # lines for cycle
    cx = [p[0] for p in cycle] + [cycle[0][0]]
    cy = [p[1] for p in cycle] + [cycle[0][1]]
    ax.plot(cx, cy, color='firebrick', lw=2, zorder=4)

    # pick two opposite mid-edge points on cycle
    left_mid = (x0, y0 + s//2)
    top_mid = (x0 + s//2, y0 + s - 1)

    # compute long perimeter path between left_mid and top_mid
    def idx_of(pt):
        for idx,p in enumerate(cycle):
            if p == pt: return idx
        return None
    a = left_mid
    b = top_mid
    ia = idx_of(a); ib = idx_of(b)
    if ia is None or ib is None:
        a = cycle[0]; b = cycle[len(cycle)//2]; ia = 0; ib = len(cycle)//2
    if ia <= ib:
        path1 = cycle[ia:ib+1]
        path2 = cycle[ib:] + cycle[:ia+1]
    else:
        path1 = cycle[ia:] + cycle[:ib+1]
        path2 = cycle[ib:ia+1]
    path = path1 if len(path1) >= len(path2) else path2
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, color='navy', lw=2, linestyle='--', zorder=5)
    # mark endpoints
    ax.scatter([a[0]],[a[1]], color='black', s=60, zorder=6)
    ax.scatter([b[0]],[b[1]], color='black', s=60, zorder=6)
    # direct chord inside (Euclidean) for comparison (shorter path across interior)
    ax.plot([a[0], b[0]], [a[1], b[1]], color='gray', lw=2, linestyle=':', zorder=6)
    ax.set_title("Euclidean grid: large cycle yields long detour", fontsize=9)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_hyperbolic_tree_with_thin_triangles(ax,
                                            levels=2,
                                            radial_layers=3,
                                            base_angle=0.0,
                                            angular_span=np.pi*0.9,
                                            triangles_per_gap=3,
                                            delta_visual=0.25,
                                            caption=("Rays and many small δ‑thin triangles near the boundary. "
                                                     "Distinct rays produce separated boundary points.")):
    """
    Draw a tree-like fan of geodesic rays and many small geodesic triangles
    near the boundary to illustrate actual δ-thin triangles and their
    consequences for the boundary. Places a centered caption just below the unit disk.
    - ax: matplotlib Axes where the disk is drawn (must be square with limits covering [-1.05,1.05]).
    - other args: tune density and visual thinness.
    """
    import numpy as np
    import matplotlib.patches as patches

    def get_geodesic_circle(z1, z2, tol=1e-12):
        z1, z2 = complex(z1), complex(z2)
        if np.isclose(z1.real * z2.imag - z1.imag * z2.real, 0.0, atol=tol):
            return None
        A = 2 * np.array([[z1.real, z1.imag], [z2.real, z2.imag]])
        b = np.array([abs(z1) ** 2 + 1.0, abs(z2) ** 2 + 1.0])
        try:
            cx, cy = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        r_sq = cx*cx + cy*cy - 1.0
        if r_sq <= 0:
            return None
        return cx, cy, np.sqrt(r_sq)

    def sample_geodesic(z1, z2, n=180):
        z1, z2 = complex(z1), complex(z2)
        params = get_geodesic_circle(z1, z2)
        if params is None:
            t = np.linspace(0, 1, n)
            pts = z1 * (1 - t) + z2 * t
            return [complex(p) for p in pts]
        cx, cy, r = params
        center = cx + 1j * cy
        a1 = np.angle(z1 - center)
        a2 = np.angle(z2 - center)
        delta_ang = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        thetas = np.linspace(a1, a1 + delta_ang, n)
        pts = center + r * np.exp(1j * thetas)
        return [complex(p) for p in pts]

    def poincare_distance(z, w):
        z, w = complex(z), complex(w)
        num = 2 * abs(z - w) ** 2
        den = (1 - abs(z) ** 2) * (1 - abs(w) ** 2)
        arg = 1 + num / (den + 1e-16)
        return np.arccosh(np.clip(arg, 1.0, None))

    def project_inside(z, margin=1e-9):
        z = complex(z)
        r = abs(z)
        if r >= 1.0 - margin:
            return z / (r + 1e-16) * (1.0 - margin)
        return z

    # ---- prepare rays (dyadic splitting) ----
    rays = []
    for L in range(levels + 1):
        step = angular_span / (2**L)
        start = base_angle - angular_span/2 + step/2
        for k in range(2**L):
            theta = start + k * step
            rays.append((L, theta))
    top_level = sorted([theta for L, theta in rays if L == levels])

    # draw boundary circle (unit disk)
    boundary = patches.Circle((0, 0), 1.0, edgecolor='black', facecolor='none', lw=1.0, zorder=0)
    ax.add_patch(boundary)

    # draw rays as radial geodesics from origin
    max_r = 0.995
    palette = plt.cm.viridis(np.linspace(0.2, 0.9, levels+1))
    for L in range(levels + 1):
        these = [theta for lvl, theta in rays if lvl == L]
        lw = max(0.9, 2.2 - 0.25 * L)
        for theta in these:
            rs = np.linspace(0.0, max_r, 80)
            xs = rs * np.cos(theta)
            ys = rs * np.sin(theta)
            ax.plot(xs, ys, color=palette[L], lw=lw, alpha=0.9, zorder=2)

    # mark endpoints on circle
    endpoint_pts = [np.exp(1j * th) for th in top_level]
    for z in endpoint_pts:
        ax.scatter([z.real], [z.imag], color='maroon', s=18, zorder=5)
        ax.plot([0.98*z.real, 1.02*z.real], [0.98*z.imag, 1.02*z.imag], color='maroon', lw=0.9, zorder=4)

    # ---- place many small triangles near the boundary and show thin neighborhoods ----
    radii = np.linspace(0.6, 0.95, radial_layers + 1)[1:]
    n_top = len(top_level)
    for i in range(n_top - 2):
        th0, th1, th2 = top_level[i], top_level[i+1], top_level[i+2]
        for r in radii:
            for t in range(triangles_per_gap):
                jitter = (t / float(max(1, triangles_per_gap))) * 0.012
                r0 = float(r - jitter*0.5)
                r1 = float(r - jitter*0.3)
                r2 = float(r - jitter*0.0)
                v0 = r0 * np.exp(1j * th0)
                v1 = r1 * np.exp(1j * (th0*0.4 + th1*0.6))
                v2 = r2 * np.exp(1j * th2)
                v0, v1, v2 = project_inside(v0, 1e-6), project_inside(v1, 1e-6), project_inside(v2, 1e-6)

                # faint triangle fill
                tri_xy = [(v.real, v.imag) for v in (v0, v1, v2)]
                poly = patches.Polygon(tri_xy, closed=True, facecolor='salmon', alpha=0.12, edgecolor='none', zorder=4)
                ax.add_patch(poly)

                # draw geodesic edges and sample points along them
                seg01 = sample_geodesic(v0, v1, n=120)
                seg12 = sample_geodesic(v1, v2, n=120)
                seg20 = sample_geodesic(v2, v0, n=120)
                for seg in (seg01, seg12, seg20):
                    ax.plot([p.real for p in seg], [p.imag for p in seg], color='salmon', lw=0.9, zorder=6)

                # flags for δ-neighborhood on each side (cheap sampled check)
                def flags_for_side(side_pts, other_pts_a, other_pts_b, delta):
                    flags = []
                    samp_other = other_pts_a[::6] + other_pts_b[::6]
                    for z in side_pts:
                        dmin = min(poincare_distance(z, w) for w in samp_other)
                        flags.append(dmin <= delta)
                    return np.array(flags, dtype=bool)

                f01 = flags_for_side(seg01, seg12, seg20, delta_visual)
                f12 = flags_for_side(seg12, seg20, seg01, delta_visual)
                f20 = flags_for_side(seg20, seg01, seg12, delta_visual)

                ax.scatter([p.real for p,ok in zip(seg01,f01) if ok],
                           [p.imag for p,ok in zip(seg01,f01) if ok], s=6, color='crimson', alpha=0.95, zorder=8)
                ax.scatter([p.real for p,ok in zip(seg12,f12) if ok],
                           [p.imag for p,ok in zip(seg12,f12) if ok], s=6, color='darkorange', alpha=0.95, zorder=8)
                ax.scatter([p.real for p,ok in zip(seg20,f20) if ok],
                           [p.imag for p,ok in zip(seg20,f20) if ok], s=6, color='darkgreen', alpha=0.95, zorder=8)

    # ---- place caption centered below the disk (outside unit circle) ----
    # Determine axes coordinates for caption placement: use data coords with y just below -1
    caption_y = -1.05
    ax.text(0.0, caption_y, caption, ha='center', va='top', fontsize=9, wrap=True, zorder=12)

    # finalize axes limits and appearance
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.25, 1.05)  # extend bottom to make room for caption
    ax.set_aspect('equal')
    ax.axis('off')


# ----------------- Compose final figure (no bottom caption) -----------------

def thin_triangle_with_demo(savefile="thin_triangle.png", delta=0.6):
    # left triangle vertices
    A = 0.0 + 0.5j
    B = -0.45 + 0.15j
    C = 0.25 - 0.35j
    A, B, C = project_inside_disk(A), project_inside_disk(B), project_inside_disk(C)
    verts = [A,B,C]

    fig = plt.figure(figsize=(14,6))
    # layout: left large panel, right stacked two smaller panels
    axL = plt.subplot2grid((3,4),(0,0), colspan=2, rowspan=3)
    axR_top = plt.subplot2grid((3,4),(0,2), colspan=2, rowspan=1)
    axR_bot = plt.subplot2grid((3,4),(1,2), colspan=2, rowspan=2)

    # Left: thin triangle
    draw_thin_triangle(axL, verts, delta=delta, pnts_per_side=400)

    # Right top: Euclidean grid
    draw_grid_with_cycle(axR_top, grid_size=8, cycle_size=6)

    # Right bottom: tree-like expansion
    draw_hyperbolic_tree_with_thin_triangles(axR_bot)

    plt.tight_layout(rect=[0,0.02,1,0.98])
    plt.savefig(savefile, dpi=220, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    thin_triangle_with_demo(delta=0.6)
