"""
Tests for the ALS emulation (utils/als_simulation.py) on small synthetic scenes.
"""
import numpy as np
import pytest

from utils.als_simulation import ALSConfig, FlightLine, emulate_als, flight_line_returns, voxel_subsample


# ----------------------------------------------------------------------------------------------------------------------
# Synthetic scenes
# ----------------------------------------------------------------------------------------------------------------------

def grid(x_range, y_range, step, z=0.0):
    x, y = np.meshgrid(np.arange(*x_range, step), np.arange(*y_range, step), indexing='ij')
    return np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], axis=1)


def ground(half=20.0, step=0.2):
    return grid((-half, half + 1e-9), (-half, half + 1e-9), step)


def box(half=5.0, height=6.0, step=0.1):
    """Roof and four facades of a closed building, centred at the origin."""
    roof = grid((-half, half + 1e-9), (-half, half + 1e-9), step, z=height)
    s = np.arange(-half, half + 1e-9, step)
    z = np.arange(0.0, height, step)
    S, Zs = np.meshgrid(s, z, indexing='ij')
    S, Zs = S.ravel(), Zs.ravel()
    facades = {
        '+y': np.stack([S, np.full(S.size, half), Zs], axis=1),
        '-y': np.stack([S, np.full(S.size, -half), Zs], axis=1),
        '+x': np.stack([np.full(S.size, half), S, Zs], axis=1),
        '-x': np.stack([np.full(S.size, -half), S, Zs], axis=1),
    }
    return roof, facades


@pytest.fixture(scope='module')
def building_scene():
    """Ground (z=0, outside the building) + building; returns points and a dict of part masks."""
    g = ground()
    roof, facades = box()
    parts = [('ground', g), ('roof', roof)] + [(f'facade{k}', v) for k, v in facades.items()]
    points = np.concatenate([p for _, p in parts])
    labels = np.concatenate([[name] * len(p) for name, p in parts])
    return points, labels


def frac(mask, sel):
    return mask[sel].mean()


# ----------------------------------------------------------------------------------------------------------------------
# Basics
# ----------------------------------------------------------------------------------------------------------------------

def test_flat_ground_is_fully_visible_from_nadir():
    mask = emulate_als(ground())
    assert mask.mean() > 0.99


def test_roof_hides_the_ground_below(building_scene):
    points, labels = building_scene
    # Ground below the roof (inside the footprint of the building)
    below_roof = (labels == 'ground') & (np.abs(points[:, 0]) < 4.5) & (np.abs(points[:, 1]) < 4.5)
    mask = emulate_als(points)
    assert frac(mask, labels == 'roof') > 0.99
    assert frac(mask, below_roof) == 0.0


def test_swath_is_limited_by_max_scan_angle():
    line = grid((-0.05, 0.05), (-100, 100), 0.1)
    config = ALSConfig(flight_lines=[FlightLine(height=100.0, max_scan_angle=10.0)])
    mask = emulate_als(line, config)
    half_swath = 100.0 * np.tan(np.deg2rad(10.0))
    assert np.all(mask == (np.abs(line[:, 1]) <= half_swath - 0.05)) or \
        np.abs(np.abs(line[mask, 1]).max() - half_swath) < 0.2
    assert not mask[np.abs(line[:, 1]) > half_swath + 0.1].any()


def test_points_above_the_sensor_are_not_seen():
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 150.0]])
    returns = flight_line_returns(pts, FlightLine(height=100.0), reference_height=0.0)
    assert returns[1] == 0


# ----------------------------------------------------------------------------------------------------------------------
# Viewing angles
# ----------------------------------------------------------------------------------------------------------------------

def test_nadir_sees_only_the_top_of_facades(building_scene):
    points, labels = building_scene
    mask = emulate_als(points)
    for side in ['+y', '-y', '+x', '-x']:
        assert frac(mask, labels == f'facade{side}') < 0.2


def test_oblique_view_sees_the_facing_facade_only(building_scene):
    points, labels = building_scene
    # Track 40 m to the +y side (left of heading 0) -> looks at the +y facade under ~19 deg
    config = ALSConfig(flight_lines=[FlightLine(heading=0.0, offset=40.0)])
    mask = emulate_als(points, config)
    assert frac(mask, labels == 'facade+y') > 0.8
    # The back facade is hidden, apart from a thin band below the roof edge (slope tolerance, see docstring)
    assert frac(mask, (labels == 'facade-y') & (points[:, 2] < 5.0)) == 0.0
    assert frac(mask, labels == 'facade-y') < 0.15
    # Shadow of the building on the ground behind it (6 m high, ~19 deg -> ~2 m long)
    shadow = (labels == 'ground') & (np.abs(points[:, 0]) < 4) & (points[:, 1] < -5.3) & (points[:, 1] > -6.8)
    assert frac(mask, shadow) < 0.05


def test_two_flight_lines_see_both_facades(building_scene):
    points, labels = building_scene
    config = ALSConfig(flight_lines=[FlightLine(offset=40.0), FlightLine(offset=-40.0)])
    mask = emulate_als(points, config)
    assert frac(mask, labels == 'facade+y') > 0.8
    assert frac(mask, labels == 'facade-y') > 0.8


def test_result_rotates_with_heading(building_scene):
    points, labels = building_scene
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg around z
    mask = emulate_als(points, ALSConfig(flight_lines=[FlightLine(heading=0.0, offset=40.0)]))
    mask_rot = emulate_als(points @ rot.T, ALSConfig(flight_lines=[FlightLine(heading=90.0, offset=40.0)]))
    assert np.mean(mask == mask_rot) > 0.99


def test_georeferenced_coordinates(building_scene):
    points, _ = building_scene
    shifted = points + np.array([4.5e6, 5.3e6, 520.0])
    config = ALSConfig(flight_lines=[FlightLine(offset=40.0)])
    assert np.mean(emulate_als(points, config) == emulate_als(shifted, config)) > 0.999


# ----------------------------------------------------------------------------------------------------------------------
# Multiple returns
# ----------------------------------------------------------------------------------------------------------------------

def test_second_return_sees_through_sparse_canopy():
    rng = np.random.default_rng(0)
    g = ground(half=10.0, step=0.1)
    # Sparse canopy at 10 m (30 % of a 0.1 m grid, like leaves with gaps) over x < 0
    c = grid((-10.0, 0.0), (-10.0, 10.0), 0.1, z=10.0)
    c = c[rng.random(len(c)) < 0.3]
    points = np.concatenate([g, c])
    is_ground = np.arange(len(points)) < len(g)
    # Ground clearly below the canopy (away from its edge, also in the perspective of a nadir line)
    under = is_ground & (points[:, 0] < -2.0) & (points[:, 0] > -8.0)

    mask1 = emulate_als(points, ALSConfig(max_returns=1))
    mask2, returns = emulate_als(points, ALSConfig(max_returns=2), return_numbers=True)

    # First return: only the few beams that pass through gaps of the whole footprint reach the ground
    assert frac(mask1, under) < 0.2
    # Second return: most of the ground below the canopy is seen
    assert frac(mask2, under) > 0.7
    assert np.mean(returns[under & mask2] == 2) > 0.8
    assert frac(mask2, is_ground & (points[:, 0] > 2.0)) > 0.99


def test_no_second_return_through_a_closed_roof(building_scene):
    points, labels = building_scene
    below_roof = (labels == 'ground') & (np.abs(points[:, 0]) < 4.5) & (np.abs(points[:, 1]) < 4.5)
    mask = emulate_als(points, ALSConfig(max_returns=2))
    assert frac(mask, below_roof) < 0.01


def test_return_separation_must_exceed_tolerance():
    with pytest.raises(ValueError):
        flight_line_returns(np.zeros((1, 3)), FlightLine(), depth_tolerance=1.0, return_separation=0.5)


# ----------------------------------------------------------------------------------------------------------------------
# Voxel subsampling (ALS point spacing)
# ----------------------------------------------------------------------------------------------------------------------

def test_voxel_subsample_one_point_per_voxel():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 10, size=(20000, 3))
    idx = voxel_subsample(pts, 1.0)
    keys = np.floor((pts - pts.min(axis=0)) / 1.0).astype(np.int64)
    assert len(idx) == len(np.unique(keys, axis=0))
    assert len(np.unique(keys[idx], axis=0)) == len(idx)


def test_voxel_subsample_respects_mask():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 10, size=(5000, 3))
    mask = pts[:, 0] < 5
    idx = voxel_subsample(pts, 1.0, mask=mask)
    assert np.all(mask[idx])


def test_empty_input():
    assert emulate_als(np.zeros((0, 3))).shape == (0,)
    assert voxel_subsample(np.zeros((0, 3)), 1.0).shape == (0,)


# ----------------------------------------------------------------------------------------------------------------------
# Whole clouds: chunking and flight strips
# ----------------------------------------------------------------------------------------------------------------------

def test_chunked_processing_gives_the_same_result(building_scene):
    points, _ = building_scene
    for line in [FlightLine(), FlightLine(heading=30.0, offset=40.0), FlightLine(heading=200.0, offset=-25.0)]:
        full = flight_line_returns(points, line, reference_height=0.0, max_returns=2)
        chunked = flight_line_returns(points, line, reference_height=0.0, max_returns=2, chunk_points=5000)
        np.testing.assert_array_equal(full, chunked)


def test_strips_cover_the_whole_cloud():
    from utils.als_simulation import strip_flight_lines, cloud_center
    rng = np.random.default_rng(0)
    # Long, curved corridor (2 km, 400 m bend)
    t = rng.uniform(0, 2000, 50000)
    pts = np.stack([t, 400 * np.sin(t / 2000 * np.pi), np.zeros_like(t)], axis=1)
    for heading in [0.0, 37.0, 90.0]:
        lines = strip_flight_lines(pts, heading=heading, height=100.0, max_scan_angle=30.0, side_overlap=0.3,
                                   phase=rng.uniform())
        mask = emulate_als(pts, ALSConfig(flight_lines=lines))
        assert mask.all()
        offsets = sorted(line.offset for line in lines)
        swath = 2 * 100.0 * np.tan(np.deg2rad(30.0))
        np.testing.assert_allclose(np.diff(offsets), 0.7 * swath)


def test_object_outside_a_sphere_casts_a_shadow_into_it():
    """Why whole clouds are converted: a tree 7 m outside the input sphere still shadows ground inside it."""
    from utils.als_simulation import strip_flight_lines
    g = ground(half=30.0, step=0.1)
    rng = np.random.default_rng(1)
    crown = rng.normal(size=(20000, 3)) * np.array([1.5, 1.5, 1.5]) + np.array([0.0, 16.0, 12.0])
    points = np.concatenate([g, crown])
    is_ground = np.arange(len(points)) < len(g)
    # Strips seen from the +y side: the shadow of the crown falls towards -y, into a sphere around the origin
    lines = [FlightLine(heading=0.0, offset=60.0)]
    whole = emulate_als(points, ALSConfig(flight_lines=lines, center=(0.0, 0.0)))
    in_sphere = np.linalg.norm(points, axis=1) < 9.0
    sphere_only = emulate_als(points[in_sphere], ALSConfig(flight_lines=lines, center=(0.0, 0.0)))
    shadow = is_ground & in_sphere & (np.abs(points[:, 0]) < 1.0) & (points[:, 1] > 4.0) & (points[:, 1] < 8.0)
    assert whole[shadow].mean() < 0.2
    assert sphere_only[shadow[in_sphere]].mean() > 0.99
