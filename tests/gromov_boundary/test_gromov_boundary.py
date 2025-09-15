# tests/test_gromov_boundary_custom.py

import pytest

import holographic_tn.geometry as geom
from holographic_tn.geometry import HyperbolicBuilding, MockIsometry

@pytest.fixture
def small_building():
    # same radius-2 patch
    b = HyperbolicBuilding(p=4, v=4)
    gx, gx_inv = MockIsometry("X+"), MockIsometry("X-")
    gy, gy_inv = MockIsometry("Y+"), MockIsometry("Y-")
    side_pairings = [
        (0, 2, gy_inv, gy),
        (1, 3, gx,    gx_inv),
        (2, 0, gy,    gy_inv),
        (3, 1, gx_inv, gx),
    ]
    b.construct_building(side_pairings, max_radius=2)
    return b

def test_all_rays_cluster_together(monkeypatch, small_building):
    """
    By monkey-patching:
      - random.sample → fixed targets
      - find_geodesic_a_star → [origin, target]
      - _get_path_distance so that:
          d(origin, target) = 1,
          d(target_i, target_j) = 0  (i≠j)

    we guarantee Gromov_product(i,j) = (1 + 1 − 0)/2 = 1 > 0,
    and with cluster_threshold_ratio=0 → one big cluster.
    """
    b = small_building
    origin = b.word_map[""]

    # pick three targets
    all_faces = [f for f in b.face_centers.keys() if f != origin]
    targets   = all_faces[:3]

    # 1) fix random.sample
    monkeypatch.setattr(geom.random, "sample", lambda pop, k: targets)

    # 2) force each ray = [origin, target]
    monkeypatch.setattr(
        HyperbolicBuilding, "find_geodesic_a_star",
        lambda self, o, t: [o, t]
    )

    # 3) define a fake path‐distance:
    #    origin→any_target = 1.0; target_i→target_j = 0.0
    def fake_path_distance(self, a, b_):
        if a == origin and b_ in targets:
            return 1.0
        if b_ == origin and a in targets:
            return 1.0
        if a in targets and b_ in targets:
            return 0.0
        # fallback
        return 0.0

    monkeypatch.setattr(
        HyperbolicBuilding, "_get_path_distance", fake_path_distance
    )

    # now run with threshold_ratio=0 → threshold=0
    clusters = b.identify_gromov_boundary(
        origin_face=origin,
        num_rays=5,
        cluster_threshold_ratio=0.0
    )

    # expect exactly one cluster containing all 3 rays
    assert isinstance(clusters, list)
    assert len(clusters) == 1, f"got {len(clusters)} clusters, expected 1"
    assert len(clusters[0]) == len(targets), (
        f"got cluster size {len(clusters[0])}, expected {len(targets)}"
    )


def test_each_ray_separate_cluster(monkeypatch, small_building):
    """
    With the same monkey‐patching as above, but threshold_ratio=1.0:
      cluster_threshold = ray_length * 1 = 1.0,
      Gromov_product = 1.0 → not > threshold,
    so every ray stands alone.
    """
    b = small_building
    origin = b.word_map[""]

    all_faces = [f for f in b.face_centers.keys() if f != origin]
    targets   = all_faces[:4]

    monkeypatch.setattr(geom.random, "sample", lambda pop, k: targets)
    monkeypatch.setattr(
        HyperbolicBuilding, "find_geodesic_a_star",
        lambda self, o, t: [o, t]
    )

    def fake_path_distance(self, a, b_):
        if a == origin and b_ in targets:
            return 1.0
        if b_ == origin and a in targets:
            return 1.0
        if a in targets and b_ in targets:
            return 0.0
        return 0.0

    monkeypatch.setattr(
        HyperbolicBuilding, "_get_path_distance", fake_path_distance
    )

    clusters = b.identify_gromov_boundary(
        origin_face=origin,
        num_rays=10,
        cluster_threshold_ratio=1.0
    )

    # expect one‐per‐ray
    assert len(clusters) == len(targets), (
        f"got {len(clusters)} clusters, expected {len(targets)}"
    )
    for cluster in clusters:
        assert isinstance(cluster, list)
        assert len(cluster) == 1

    # ensure each target appears exactly once
    ends = {cluster[0][-1] for cluster in clusters}
    assert ends == set(targets)
