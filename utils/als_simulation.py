#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Emulation of airborne laser scanning (ALS) on dense MLS / TLS point clouds
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      RailTwin addition
#
#      The emulation answers: "which of these points would an airborne scanner see?"
#
#      The sensor flies along straight lines at a given height and scans across the track (line scanner). For every
#      flight line the points are projected into a depth image in sensor coordinates (position along the track x scan
#      angle). A point is hidden if another point within the beam footprint around its ray is clearly closer to the
#      sensor (slope-tolerant z-buffer, see flight_line_returns). Optionally the beam continues through gaps (e.g.
#      vegetation) to produce later returns. A point is visible if any flight line sees it.
#
#      Training (datasets/common.py): prepare_als_versions flies parallel strips over every whole training cloud
#      (so objects outside an input sphere still cast shadows into it), prepared_als_mask mixes the versions in.
#      voxel_subsample() reduces a result to an ALS-like point spacing (emulate_als.py --spacing).
#
#      Limitation: MLS / TLS rarely capture roofs or canopy tops. Ground below a missing roof is not occluded.
#

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FlightLine:
    """Straight flight line of a line-scanning ALS sensor."""
    heading: float = 0.0            # [deg] flight direction in the xy-plane, 0 = +x, 90 = +y
    offset: float = 0.0             # [m] lateral offset of the track from the cloud centre (positive = left of heading)
    height: float = 100.0           # [m] flying height above the reference height
    max_scan_angle: float = 30.0    # [deg] half swath angle across the track


@dataclass
class ALSConfig:
    flight_lines: list = field(default_factory=lambda: [FlightLine()])
    footprint: float = 0.3          # [m] beam footprint at nadir = depth map cell size
    depth_tolerance: float = 0.2    # [m] points up to this distance behind the closest point are still seen
    max_incidence: float = 80.0     # [deg] surfaces seen up to this incidence angle do not hide themselves
    max_returns: int = 1            # returns per beam (> 1 lets the beam continue behind a hit)
    return_separation: float = 1.5  # [m] minimal range between two returns of the same beam
    solid_coverage: float = 0.6     # footprint fraction covered by a return above which the beam does not pass
                                    # (measured on pixels of footprint / 3: closed surfaces sampled at about that
                                    # spacing reach ~0.6-0.7 because of aliasing)
    center: tuple = None            # [m] xy origin for headings / offsets (None = centre of the xy bounding box)


def _window_offsets(radius, pixel):
    """Pixel offsets (du, dv) within a disc of the given radius and their distances in metres."""
    r = int(np.ceil(radius / pixel))
    du, dv = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing='ij')
    dist = pixel * np.sqrt(du ** 2 + dv ** 2)
    keep = dist <= radius + 1e-9
    return du[keep], dv[keep], dist[keep], r


def cloud_center(points):
    """xy centre of the bounding box, the origin of flight line headings / offsets in emulate_als."""
    points = np.asarray(points)
    return (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2


def strip_flight_lines(points, heading=0.0, height=100.0, max_scan_angle=30.0, side_overlap=0.3, phase=0.0,
                       center=None):
    """
    Parallel flight strips that cover the whole cloud, like a real ALS block: every point lies within the swath
    of at least one strip, neighbouring strips overlap by side_overlap of the swath width.

    :param points: float[N, 3]
    :param heading: [deg] flight direction of all strips
    :param height: [m] flying height (above the reference height of emulate_als)
    :param side_overlap: overlap of neighbouring swaths, 0 .. <1
    :param phase: shift of the strip pattern in units of the strip spacing, 0 .. 1 (varies the viewing angles)
    :param center: xy origin of the offsets (None = cloud_center, as in emulate_als)
    :return: list of FlightLine
    """
    if not 0.0 <= side_overlap < 1.0:
        raise ValueError('side_overlap must be in [0, 1)')
    center = cloud_center(points) if center is None else np.asarray(center, dtype=np.float64)
    h = np.deg2rad(heading)
    xy = np.asarray(points, dtype=np.float64)[:, :2] - center
    across = -xy[:, 0] * np.sin(h) + xy[:, 1] * np.cos(h)
    swath = 2 * height * np.tan(np.deg2rad(max_scan_angle))
    spacing = swath * (1.0 - side_overlap)
    starts = np.arange(across.min() - phase * spacing, across.max() + spacing, spacing)
    return [FlightLine(heading=heading, offset=float(o + spacing / 2), height=height, max_scan_angle=max_scan_angle)
            for o in starts]


def flight_line_returns(points, line, footprint=0.3, depth_tolerance=0.2, max_incidence=80.0, max_returns=1,
                        return_separation=1.5, solid_coverage=0.6, reference_height=0.0, chunk_points=2_000_000):
    """
    Return number of every point for one flight line (0 = not seen, 1 = first return, 2 = second return, ...).

    Depth image in sensor coordinates (along track x scan angle, in metres at nadir) with pixels of footprint / 3.
    A point q hides a point p if q is within the beam footprint around p's ray and
        range_q < range_p - depth_tolerance - tan(max_incidence) * (lateral distance + pixel size).
    The slope term keeps surfaces seen under an incidence angle up to max_incidence (e.g. facades in an oblique
    view) from hiding themselves, while objects stacked along the ray (roof over ground) still occlude.
    Later returns only pass where the previous return covers less than solid_coverage of the footprint (gaps in
    vegetation, wires), not through closed surfaces, and only behind the return separation.
    Known artefact: up to about tan(max_incidence) * footprint / 3 below an occluding edge stays visible.

    :param points: float[N, 3], already centred in xy (see emulate_als)
    :param line: FlightLine
    :return: int8[N]
    """
    if return_separation <= depth_tolerance:
        raise ValueError('return_separation must be larger than depth_tolerance')

    points = np.asarray(points, dtype=np.float64)
    returns = np.zeros(len(points), dtype=np.int8)
    if len(points) == 0:
        return returns

    # Sensor frame: x along the track, y across the track (relative to the track), z down from the sensor
    h = np.deg2rad(line.heading)
    c, s = np.cos(h), np.sin(h)
    along = points[:, 0] * c + points[:, 1] * s
    across = -points[:, 0] * s + points[:, 1] * c - line.offset
    below = reference_height + line.height - points[:, 2]

    scan = np.arctan2(across, below)
    valid = (below > 0) & (np.abs(scan) <= np.deg2rad(line.max_scan_angle))
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return returns

    rng = np.sqrt(across[idx] ** 2 + below[idx] ** 2)

    # Depth image coordinates in metres at nadir. Occlusion only acts within the window around a pixel, so long
    # lines are processed in chunks along the track (overlapping by the window) with the exact same result.
    pixel = footprint / 3
    window = _window_offsets(footprint / 2, pixel)
    r = window[3]
    iu = np.floor(along[idx] / pixel).astype(np.int64)
    iv = np.floor(scan[idx] * line.height / pixel).astype(np.int64)
    params = dict(window=window, pixel=pixel, depth_tolerance=depth_tolerance, max_incidence=max_incidence,
                  max_returns=max_returns, return_separation=return_separation, solid_coverage=solid_coverage)

    if len(idx) <= chunk_points:
        returns[idx] = _image_returns(iu, iv, rng, **params)
        return returns

    order = np.argsort(iu, kind='stable')
    iu_sorted = iu[order]
    n_chunks = int(np.ceil(len(idx) / chunk_points))
    bounds = iu_sorted[np.linspace(0, len(idx) - 1, n_chunks + 1).astype(np.int64)]
    bounds[-1] += 1
    for u0, u1 in zip(bounds[:-1], bounds[1:]):
        lo, hi = np.searchsorted(iu_sorted, [u0 - r, u1 + r])
        sub = order[lo:hi]
        core = (iu[sub] >= u0) & (iu[sub] < u1)
        sub_returns = _image_returns(iu[sub], iv[sub], rng[sub], **params)
        returns[idx[sub[core]]] = sub_returns[core]

    return returns


def _image_returns(iu, iv, rng, window, pixel, depth_tolerance, max_incidence, max_returns, return_separation,
                   solid_coverage):
    """Return numbers of points given by depth image pixel (iu, iv) and range (see flight_line_returns)."""
    du, dv, dist, r = window
    width = iv.max() - iv.min() + 2 * r + 1
    keys = (iu - iu.min() + r) * width + (iv - iv.min() + r)
    pix_keys, pix = np.unique(keys, return_inverse=True)
    pix = pix.ravel()
    n_pix = len(pix_keys)

    # Position of every window neighbour of every occupied pixel (-1 = not occupied)
    neighbours = np.full((len(du), n_pix), -1, dtype=np.int64)
    for o in range(len(du)):
        nkeys = pix_keys + du[o] * width + dv[o]
        pos = np.minimum(np.searchsorted(pix_keys, nkeys), n_pix - 1)
        found = pix_keys[pos] == nkeys
        neighbours[o, found] = pos[found]

    def window_min(sel, slope):
        """Per pixel: min over the window of (closest selected range + slope * (distance + pixel))."""
        D = np.full(n_pix + 1, np.inf)  # last entry = empty neighbour
        np.minimum.at(D, pix[sel], rng[sel])
        E = np.full(n_pix, np.inf)
        for o in range(len(du)):
            E = np.minimum(E, D[neighbours[o]] + slope * (dist[o] + pixel))
        coverage = np.mean(np.isfinite(D[neighbours]), axis=0)
        return E, coverage

    returns = np.zeros(len(rng), dtype=np.int8)
    slope = np.tan(np.deg2rad(max_incidence))
    remaining = np.ones(len(rng), dtype=bool)
    for k in range(1, max_returns + 1):
        if not remaining.any():
            break
        E, _ = window_min(remaining, slope)
        hit = remaining & (rng <= E[pix] + depth_tolerance)
        returns[hit] = k

        # The beam continues through gaps of this return, behind the return separation
        front, coverage = window_min(hit, 0.0)
        remaining &= ~hit & (coverage[pix] < solid_coverage) & (rng >= front[pix] + return_separation)

    return returns


def _voxelize(points, voxel_size):
    """Voxel index of every point and the centroid of every voxel."""
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse)
    centroids = np.stack([np.bincount(inverse, weights=points[:, d]) / counts for d in range(3)], axis=1)
    return inverse, centroids


def emulate_als(points, config=None, return_numbers=False):
    """
    Emulate an ALS recording of a dense point cloud.

    :param points: float[N, 3]
    :param config: ALSConfig (None = defaults: one nadir flight line at 100 m)
    :param return_numbers: also return the return number of every point (smallest over all flight lines)
    :return: bool[N] visibility mask [, int8[N] return numbers (0 = not seen)]
    """
    config = ALSConfig() if config is None else config
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        empty = np.zeros(0, dtype=bool)
        return (empty, np.zeros(0, dtype=np.int8)) if return_numbers else empty

    # Centre in xy (also avoids precision problems with georeferenced coordinates)
    if config.center is None:
        center = cloud_center(points)
    else:
        center = np.asarray(config.center, dtype=np.float64)
    local = points.copy()
    local[:, :2] -= center
    reference_height = points[:, 2].min()

    best = np.zeros(len(local), dtype=np.int8)
    for line in config.flight_lines:
        returns = flight_line_returns(local, line,
                                      footprint=config.footprint,
                                      max_incidence=config.max_incidence,
                                      depth_tolerance=config.depth_tolerance,
                                      max_returns=config.max_returns,
                                      return_separation=config.return_separation,
                                      solid_coverage=config.solid_coverage,
                                      reference_height=reference_height)
        seen = returns > 0
        best[seen] = np.where(best[seen] > 0, np.minimum(best[seen], returns[seen]), returns[seen])

    mask = best > 0
    return (mask, best) if return_numbers else mask


def voxel_subsample(points, voxel_size, mask=None):
    """
    Indices of one point per occupied voxel (the point closest to the voxel centroid), e.g. to reduce the
    visible points to an ALS-like point spacing.

    :param points: float[N, 3]
    :param voxel_size: [m]
    :param mask: bool[N], only consider these points
    :return: int[M] sorted indices into points
    """
    points = np.asarray(points, dtype=np.float64)
    candidates = np.arange(len(points)) if mask is None else np.flatnonzero(mask)
    if len(candidates) == 0:
        return candidates
    sub = points[candidates] - points[candidates].min(axis=0)
    inverse, centroids = _voxelize(sub, voxel_size)
    dist = np.sum((sub - centroids[inverse]) ** 2, axis=1)
    order = np.lexsort((dist, inverse))
    first = np.ones(len(order), dtype=bool)
    first[1:] = inverse[order][1:] != inverse[order][:-1]
    return np.sort(candidates[order[first]])


# ----------------------------------------------------------------------------------------------------------------------
#
#           Prepared ALS versions (dataset preparation step + mixing)
#       \***********************************************************/
#

def prepared_als_settings(config):
    """ALS settings of a config that define the prepared versions (also the cache key)."""
    return {'versions': int(config.augment_als_versions),
            'height': float(config.augment_als_height),
            'max_scan_angle': float(config.augment_als_max_scan_angle),
            'side_overlap': float(config.augment_als_side_overlap),
            'footprint': float(config.augment_als_footprint),
            'returns': int(config.augment_als_returns),
            'algorithm': 1}  # increase when the emulation changes, invalidates old caches


def prepare_als_versions(points, settings, seed=0):
    """
    ALS versions of a whole cloud: version v is flown in parallel strips with heading v * 180 / versions (plus a
    random rotation of the pattern per cloud), so the versions see the cloud from different directions.

    :param points: float[N, 3] whole cloud
    :param settings: dict from prepared_als_settings
    :param seed: makes the random heading offset and strip phases reproducible
    :return: uint8[versions, ceil(N / 8)] visibility masks packed with np.packbits
    """
    rng = np.random.default_rng(seed)
    base_heading = rng.uniform(0.0, 180.0)
    masks = []
    for v in range(settings['versions']):
        lines = strip_flight_lines(points, heading=base_heading + v * 180.0 / settings['versions'],
                                   height=settings['height'], max_scan_angle=settings['max_scan_angle'],
                                   side_overlap=settings['side_overlap'], phase=rng.uniform(0.0, 1.0))
        als_config = ALSConfig(flight_lines=lines, footprint=settings['footprint'], max_returns=settings['returns'])
        masks.append(emulate_als(points, als_config))
    return np.packbits(np.stack(masks), axis=1)


def prepared_als_mask(packed, input_inds, config, rng=None):
    """
    Mixing step: with probability config.augment_als, the sample uses one random prepared ALS version.

    :param packed: uint8[versions, ceil(N / 8)] from prepare_als_versions
    :param input_inds: int[M] indices of the sample in the whole cloud
    :param rng: np.random (default, follows np.random.seed of the data loader workers) or a np.random.Generator
    :return: bool[M] mask of the points to keep, or None to keep the original sample
    """
    # No random draw when disabled, so trainings without ALS augmentation are unchanged
    if not config.augment_als or packed is None or len(packed) == 0:
        return None
    rng = np.random if rng is None else rng
    if rng.random() >= config.augment_als:
        return None
    v = rng.choice(len(packed))
    input_inds = np.asarray(input_inds)
    mask = ((packed[v, input_inds >> 3] >> (7 - (input_inds & 7))) & 1).astype(bool)
    if mask.sum() < config.augment_als_min_points:
        return None
    return mask
