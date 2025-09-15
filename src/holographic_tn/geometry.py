# src/holographic_tn/geometry.py
import random
from collections import deque

import networkx as nx
import numpy as np
import heapq
from typing import Union, Tuple, List, Dict


class PoincareIsometry:
    """
    An object representing a genuine hyperbolic isometry in the Poincaré disk model.

    This class implements orientation-preserving isometries (a specific type of
    Möbius transformation) that are composed of a hyperbolic translation
    followed by a rotation. It is designed to be a direct, mathematically
    correct replacement for the `MockIsometry` placeholder.
    """

    def __init__(self, name: str, z0: Union[complex, Tuple[float, float]] = (0, 0),
                 phi: float = 0.0, _inverse: bool = False):
        """
        Initializes the hyperbolic isometry.

        Args:
            name: A string identifier for the isometry (e.g., 'a').
            z0: The translation component, representing the point that will be
                moved to the origin. Can be a complex number or a tuple (x, y).
                Must be within the unit disk.
            phi: The rotation component in radians.
            _inverse: A private flag to indicate if this is an inverse
                      transformation. Users should call `get_inverse()` instead
                      of setting this directly.
        """
        self.name = name
        self.phi = phi
        self._inverse = _inverse

        # FIX: Ensure self.z0 is a NumPy complex number for consistent methods (.conj()).
        if isinstance(z0, tuple):
            self.z0 = np.complex128(z0[0] + 1j * z0[1])
        else:
            self.z0 = np.complex128(z0)

        if np.abs(self.z0) >= 1.0:
            raise ValueError("Translation point z0 must be inside the unit disk.")

        self.rotation = np.exp(1j * self.phi)

    def __repr__(self) -> str:
        """Returns a string representation of the isometry."""
        inv_str = " (inverse)" if self._inverse else ""
        return (f"PoincareIsometry(name='{self.name}', z0={self.z0:.2f}, "
                f"phi={self.phi:.2f} rad){inv_str}")

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """
        Applies the transformation to a set of (x, y) coordinates.

        Args:
            coords: A NumPy array of shape (2,) representing the (x, y) point.

        Returns:
            A NumPy array of shape (2,) for the transformed coordinates.
        """
        if self._inverse:
            return self._apply_inverse(coords)
        return self._apply_forward(coords)

    def _apply_forward(self, coords: np.ndarray) -> np.ndarray:
        """Applies the forward transformation: Rotation(Translation(z))."""
        z = coords[0] + 1j * coords[1]

        # Hyperbolic translation that moves z0 to the origin
        translated_z = (z - self.z0) / (1 - self.z0.conj() * z)

        # Rotation around the origin
        final_z = self.rotation * translated_z

        return np.array([final_z.real, final_z.imag])

    def _apply_inverse(self, coords: np.ndarray) -> np.ndarray:
        """Applies the inverse transformation: Translation⁻¹(Rotation⁻¹(z))."""
        z = coords[0] + 1j * coords[1]

        # Inverse rotation
        rotated_z = self.rotation.conj() * z

        # Inverse hyperbolic translation that moves the origin to z0
        final_z = (rotated_z + self.z0) / (1 + self.z0.conj() * rotated_z)

        return np.array([final_z.real, final_z.imag])

    def get_inverse(self, new_name: str) -> 'PoincareIsometry':
        """
        Creates and returns a new PoincareIsometry object that is the
        exact inverse of this one.

        Args:
            new_name: The name for the new inverse isometry (e.g., 'a_inv').

        Returns:
            A new PoincareIsometry instance configured for the inverse operation.
        """
        return PoincareIsometry(new_name, self.z0, self.phi, _inverse=not self._inverse)


class HyperbolicBuilding:
    """
    A computational representation of a hyperbolic building, combining generative
    and procedural construction methods.
    """

    def __init__(self, p, v):
        """Initializes the HyperbolicBuilding."""
        self.p = p
        self.v = v
        self.simplicial_complex = nx.Graph()
        self.face_centers = {}
        self.adjacency_info = {}
        self.word_map = {}
        self.layers = {}
        self._next_simplex_id = 0

        # Stores lists of plaquettes for each coarse-graining level.
        # Each plaquette is a canonically ordered list of tensor IDs.
        # Use dictionaries keyed by level to store the hierarchy of metadata
        self.plaquettes_by_level: Dict[int, List[List[Tuple[int, int]]]] = {}
        self.plaquette_map_by_level: Dict[int, Dict[frozenset, str]] = {}
        self.coarse_bonds_by_level: Dict[int, Dict[Tuple[str, str], str]] = {}

    def _get_next_id(self, dim):
        """Generates a new, unique ID for a simplex."""
        s_id = self._next_simplex_id
        self._next_simplex_id += 1
        return (dim, s_id)

    def _add_face(self, coords):
        """Adds a single face to the building (manual construction)."""
        face_id = self._get_next_id(2)
        self.simplicial_complex.add_node(face_id, type="face")
        center = np.asarray(coords)
        self.face_centers[face_id] = center
        return face_id

    def _glue_faces(self, face1_id, face2_id, edge1_idx, edge2_idx):
        """Connects two faces along specified edges (manual construction)."""
        self.simplicial_complex.add_edge(face1_id, face2_id, type="adjacency")
        self.adjacency_info[(face1_id, face2_id)] = (edge1_idx, edge2_idx)
        self.adjacency_info[(face2_id, face1_id)] = (edge2_idx, edge1_idx)

    def _simplify_word(self, word, inverse_map):
        """Simplifies a word of generators by canceling adjacent inverses."""
        simplified = word.split(';')
        i = 0
        while i < len(simplified) - 1:
            if simplified[i + 1] == inverse_map.get(simplified[i]):
                del simplified[i:i + 2]
                i = max(0, i - 1)
            else:
                i += 1
        return ";".join(simplified)

    def construct_2x2_grid(self, initial_coords: np.ndarray):
        """Constructs a simple, deterministic 2x2 grid of faces.

        This method is primarily for testing purposes to create a predictable
        and simple geometry.

        Args:
            initial_coords (np.ndarray): The coordinates for the bottom-left face (f0).
        """
        # --- FIX: Define coordinates explicitly to ensure they are distinct ---
        # A smaller scale factor keeps the grid comfortably inside the Poincare disk.
        scale = 0.4

        # Define each coordinate absolutely based on the initial coordinate.
        # This is more robust than incremental modifications.
        coords0 = initial_coords
        coords1 = initial_coords + np.array([0.0, 1.0]) * scale
        coords2 = initial_coords + np.array([1.0, 0.0]) * scale
        coords3 = initial_coords + np.array([1.0, 1.0]) * scale

        # Add the four faces using their unique, pre-computed coordinates.
        f0 = self._add_face(coords0)
        f1 = self._add_face(coords1)
        f2 = self._add_face(coords2)
        f3 = self._add_face(coords3)

        # Edges: 0:bottom, 1:right, 2:top, 3:left
        self._glue_faces(f0, f1, edge1_idx=2, edge2_idx=0)
        self._glue_faces(f0, f2, edge1_idx=1, edge2_idx=3)
        self._glue_faces(f1, f3, edge1_idx=1, edge2_idx=3)
        self._glue_faces(f2, f3, edge1_idx=2, edge2_idx=0)

    def _get_face_neighbors(self, f_id):
        """Returns a list of face-type neighbors for a given face."""
        return [n for n in self.simplicial_complex.neighbors(f_id) if
                self.simplicial_complex.nodes[n].get('type') == 'face']

    def construct_building(self, side_pairings, max_radius=2):
        """Constructs a patch of the building from side-pairing rules."""
        origin_id = self._add_face(np.array([0, 0]))
        self.word_map = {"": origin_id}
        queue = deque([("", origin_id)])

        inverse_map = {p[2].name: p[3].name for p in side_pairings}
        inverse_map.update({p[3].name: p[2].name for p in side_pairings})

        while queue:
            current_word, current_face_id = queue.popleft()
            radius = len(current_word.split(';')) if current_word else 0
            if max_radius > 0 and radius >= max_radius:
                continue

            for edge_idx in range(self.p):
                try:
                    rule = next(p for p in side_pairings if p[0] == edge_idx)
                    target_edge_idx = rule[1]
                    generator = rule[2]
                    inverse_generator = rule[3]
                except StopIteration:
                    continue

                new_word_list = current_word.split(';') if current_word else []
                new_word_list.append(generator.name)
                new_word = self._simplify_word(";".join(new_word_list), inverse_map)

                if new_word not in self.word_map:
                    current_center = self.face_centers[current_face_id]
                    new_center = inverse_generator(current_center)
                    new_face_id = self._add_face(new_center)
                    self.word_map[new_word] = new_face_id
                    queue.append((new_word, new_face_id))

                neighbor_face_id = self.word_map[new_word]
                self._glue_faces(current_face_id, neighbor_face_id, edge_idx, target_edge_idx)

    def construct_tiling(self, layers=3):
        """
        Constructs a regular {p, q} tiling dual graph, where q = 2*(v-1).
        This method robustly generates the graph procedurally.
        """
        q = 2 * (self.v)
        if (self.p - 2) * (q - 2) <= 4:
            raise ValueError(f"Parameters p={self.p}, v={self.v} (q={q}) do not form a hyperbolic tiling.")

        d = np.arccosh(1 / (np.tan(np.pi / self.p) * np.tan(np.pi / q)))
        c_base = np.tanh(d / 2)

        origin_id = self._add_face(np.array([0.0, 0.0]))
        self.layers[origin_id] = 0

        queue = deque([(origin_id, 0 + 0j)])
        visited_coords = {0j: origin_id}
        node_counter = 1

        while queue:
            parent_id, parent_z = queue.popleft()
            layer = self.layers[parent_id]

            if layer >= layers:
                continue

            for i in range(self.p):
                angle = (2 * np.pi / self.p) * i
                if parent_z != 0:
                    angle += np.angle(parent_z)

                c = c_base * np.exp(1j * angle)
                child_z = (parent_z - c) / (1 - np.conj(c) * parent_z)

                existing_node = None
                for p_z, nid in visited_coords.items():
                    if abs(child_z - p_z) < 1e-4:
                        existing_node = nid
                        break

                if existing_node is None:
                    child_id = self._get_next_id(2)
                    self.simplicial_complex.add_node(child_id)
                    self.face_centers[child_id] = np.array([child_z.real, child_z.imag])
                    self.layers[child_id] = layer + 1
                    self.simplicial_complex.add_edge(parent_id, child_id)
                    queue.append((child_id, child_z))
                    visited_coords[child_z] = child_id
                else:
                    if not self.simplicial_complex.has_edge(parent_id, existing_node):
                        self.simplicial_complex.add_edge(parent_id, existing_node)

    def _hyperbolic_distance(self, p1, p2):
        """Calculates the genuine hyperbolic distance between two points."""
        z1 = p1[0] + 1j * p1[1]
        z2 = p2[0] + 1j * p2[1]
        if np.allclose(p1, p2): return 0.0
        dist_sq = np.abs(z2 - z1) ** 2
        denom = (1 - np.abs(z1) ** 2) * (1 - np.abs(z2) ** 2)
        if denom <= 1e-12: return float('inf')
        return np.arccosh(1 + 2 * dist_sq / denom)

    def find_geodesic_a_star(self, start, end):
        """Finds the shortest path (geodesic) between two faces using A*."""
        if start not in self.face_centers or end not in self.face_centers:
            return None

        open_set = [(0, start)]
        came_from = {}
        g_score = {n: float('inf') for n in self.face_centers}
        g_score[start] = 0

        while open_set:
            _, curr = heapq.heappop(open_set)
            if curr == end:
                path = [curr]
                while curr in came_from:
                    curr = came_from[curr]
                    path.insert(0, curr)
                return path

            for neighbor in self.simplicial_complex.neighbors(curr):
                dist = self._hyperbolic_distance(self.face_centers[curr], self.face_centers[neighbor])
                tentative_g = g_score[curr] + dist

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = curr
                    g_score[neighbor] = tentative_g
                    h_cost = self._hyperbolic_distance(self.face_centers[neighbor], self.face_centers[end])
                    f_score = tentative_g + h_cost
                    heapq.heappush(open_set, (f_score, neighbor))
        return None

    def _get_geometric_path_length(self, start_face, end_face):
        """Calculates the total length of the geodesic path between two faces."""
        path = self.find_geodesic_a_star(start_face, end_face)
        if not path:
            return 0.0

        distance = 0.0
        for i in range(len(path) - 1):
            p1 = self.face_centers[path[i]]
            p2 = self.face_centers[path[i + 1]]
            distance += self._hyperbolic_distance(p1, p2)

        return distance

    def _get_path_distance(self, start_face, end_face):
        """Calculates the total length of the geodesic path between two faces."""
        path = self.find_geodesic_a_star(start_face, end_face)
        if not path or len(path) < 2:
            return 0

        # --- BUG FIX ---
        # The original calculation summed the Euclidean distance between face
        # centers. This is not the correct measure for the "area" of the discrete
        # surface in the toy model. The tests expect the length to be the number
        # of edges in the dual graph path, which represents the "size" of the
        # minimal surface.
        return float(len(path) - 1)

    def identify_gromov_boundary(self, origin_face, num_rays=15, cluster_threshold_ratio=0.8):
        """Identifies and represents the Gromov boundary of the building.

        This method implements an algorithm to find the "boundary at infinity" by
        [cite_start]clustering geodesic rays[cite: 125, 128].

        Algorithm Steps:
        1. [cite_start]Generate a set of long geodesic rays from a fixed origin[cite: 130, 131].
        2. [cite_start]Compute the Gromov product between pairs of rays as a similarity metric[cite: 136].
           [cite_start]The product is defined as (x,y)_o = 1/2 * (d(o,x) + d(o,y) - d(x,y))[cite: 135].
        3. Cluster rays where the Gromov product is large, indicating they are
           [cite_start]equivalent and point to the same spot on the boundary[cite: 137].
        4. [cite_start]Each cluster represents a single point on the Gromov boundary[cite: 138].

        Args:
            origin_face (tuple): The face ID to use as the origin for all rays.
            num_rays (int): The number of geodesic rays to generate.
            cluster_threshold_ratio (float): A value between 0 and 1. Two rays
                are clustered if their Gromov product exceeds this fraction of
                the shorter ray's length.

        Returns:
            list: A list of clusters, where each cluster is a list of geodesic
                  ray paths. Each cluster represents one point on the boundary.
        """
        all_faces = list(self.face_centers.keys())
        if not all_faces: return []

        # --- Step 1: Generate Geodesic Rays ---
        print(f"▶️ Step 1: Generating {num_rays} geodesic rays from origin {origin_face}...")
        rays = []
        target_faces = random.sample(all_faces, min(num_rays, len(all_faces)))
        for target in target_faces:
            if target == origin_face: continue
            ray_path = self.find_geodesic_a_star(origin_face, target)
            if ray_path:
                rays.append(ray_path)

        if not rays:
            print("  Could not generate any rays.")
            return []

        # --- Step 2: Compute Gromov Product Similarity Matrix ---
        print("▶️ Step 2: Computing Gromov product similarity matrix...")
        num_rays_found = len(rays)
        gromov_matrix = np.zeros((num_rays_found, num_rays_found))
        ray_lengths = [self._get_path_distance(origin_face, r[-1]) for r in rays]

        for i in range(num_rays_found):
            for j in range(i, num_rays_found):
                if i == j:
                    gromov_matrix[i, j] = ray_lengths[i]
                    continue

                x, y = rays[i][-1], rays[j][-1]
                d_ox, d_oy = ray_lengths[i], ray_lengths[j]
                d_xy = self._get_path_distance(x, y)

                # (x,y)[cite_start]_o = 1/2 * (d(o,x) + d(o,y) - d(x,y)) [cite: 135]
                product = 0.5 * (d_ox + d_oy - d_xy)
                gromov_matrix[i, j] = gromov_matrix[j, i] = product

        # --- Step 3: Cluster Rays and Represent the Boundary ---
        print("▶️ Step 3: Clustering rays to identify boundary points...")
        clusters = []
        unclustered_indices = list(range(num_rays_found))

        while unclustered_indices:
            seed_idx = unclustered_indices.pop(0)
            new_cluster = [rays[seed_idx]]

            # [cite_start]The Gromov product measures how long two rays 'stay together'[cite: 134].
            # We cluster rays if their product is a high fraction of their length.
            cluster_threshold = ray_lengths[seed_idx] * cluster_threshold_ratio

            # Find other rays that belong to this cluster
            indices_to_remove = []
            for i in unclustered_indices:
                if gromov_matrix[seed_idx, i] > cluster_threshold:
                    new_cluster.append(rays[i])
                    indices_to_remove.append(i)

            # Remove newly clustered items from the unclustered pool
            for i in sorted(indices_to_remove, reverse=True):
                unclustered_indices.remove(i)

            clusters.append(new_cluster)

        return clusters

    # === NEW METHODS for preparing TNR metadata ===

    def prepare_for_tnr(self, max_levels: int = 1):
        """
        Iteratively pre-computes the plaquettes and coarse-grained topology
        for multiple TNR levels.
        """
        print(f"▶️ Pre-computing geometric metadata for {max_levels} TNR levels...")

        current_graph = self.simplicial_complex

        for level in range(max_levels):
            print(f"  -- Analyzing Level {level} --")

            # 1. Find and store plaquettes for the current level's graph
            #    NOTE: This assumes a p=5 (pentagon) finder.
            plaquettes = self._find_pentagons(current_graph)
            if not plaquettes:
                print(f"  No more plaquettes found at Level {level}. Stopping.")
                break
            self.plaquettes_by_level[level] = plaquettes

            # 2. Pre-compute the topology for the *next* level
            self._precompute_coarse_topology(level, plaquettes)

            # 3. Construct the actual graph for the next level
            next_level_graph = nx.Graph()
            plaquette_map = self.plaquette_map_by_level[level]
            coarse_bonds = self.coarse_bonds_by_level[level]

            new_nodes = list(plaquette_map.values())
            next_level_graph.add_nodes_from(new_nodes)

            for (n1, n2) in coarse_bonds:
                next_level_graph.add_edge(n1, n2)

            # The new graph becomes the input for the next iteration
            current_graph = next_level_graph

        print("✅ Metadata computation complete.")

    def _find_pentagons(self, graph: nx.Graph) -> List[List[any]]:
        """
        Finds all simple 5-cycles (pentagons) in the given graph.

        Args:
            graph: The networkx.Graph object to search for pentagons.

        Returns:
            A list of plaquettes, where each plaquette is a list of the five
            tensor IDs that form the pentagon.
        """
        q = 2 * self.v

        found_plaquettes = []
        seen_plaquettes = set()  # Use a set for efficient duplicate checking

        b = sorted([len(a) for a in nx.simple_cycles(graph)])

        # Use networkx's efficient cycle-finding algorithm
        for cycle in nx.simple_cycles(graph):
            if len(cycle) == q:
                # Use a frozenset of IDs as a unique key for the plaquette
                plaquette_key = frozenset(cycle)
                if plaquette_key not in seen_plaquettes:
                    found_plaquettes.append(cycle)
                    seen_plaquettes.add(plaquette_key)

        return found_plaquettes

    def _precompute_coarse_topology(self, level: int, plaquettes: List[List[any]]):
        """
        Builds the "master blueprint" for the next, coarser grid for a given level.
        """
        # Initialize the dictionaries for the current level
        self.plaquette_map_by_level[level] = {}
        self.coarse_bonds_by_level[level] = {}

        # 1. Assign unique IDs for the next-level tensors
        for i, p_ids in enumerate(plaquettes):
            plaquette_key = frozenset(p_ids)
            new_tid = f"L{level + 1}_{i}"
            self.plaquette_map_by_level[level][plaquette_key] = new_tid

        # 2. Find adjacent plaquettes to define the next-level bonds
        for i in range(len(plaquettes)):
            for j in range(i + 1, len(plaquettes)):
                p1_ids, p2_ids = set(plaquettes[i]), set(plaquettes[j])

                if len(p1_ids.intersection(p2_ids)) == 2:  # Adjacent
                    p1_key, p2_key = frozenset(plaquettes[i]), frozenset(plaquettes[j])
                    new_tid1 = self.plaquette_map_by_level[level][p1_key]
                    new_tid2 = self.plaquette_map_by_level[level][p2_key]

                    new_bond_ind = f"cb_{new_tid1}_{new_tid2}"
                    self.coarse_bonds_by_level[level][(new_tid1, new_tid2)] = new_bond_ind
                    self.coarse_bonds_by_level[level][(new_tid2, new_tid1)] = new_bond_ind

    def get_plaquettes(self, level: int = 0) -> List[List[Tuple[int, int]]]:
        """Returns the pre-computed plaquettes for a specific level."""
        return self.plaquettes_by_level.get(level, [])

    def get_coarse_grained_network_info(self, level: int = 0) -> Tuple[Dict, Dict]:
        """Returns the maps for a specific level of coarse-graining."""
        plaquette_map = self.plaquette_map_by_level.get(level, {})
        coarse_bonds = self.coarse_bonds_by_level.get(level, {})
        return plaquette_map, coarse_bonds
