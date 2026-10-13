#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Convert MLS / TLS point clouds (.ply) into ALS-like point clouds, see utils/als_simulation.py
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      RailTwin addition
#
#      Prepare an ALS dataset (recommended): parallel flight strips over the whole cloud, so every object occludes.
#      Writing the versions next to the originals (same folder, --suffix) mixes MLS and ALS samples in training
#      without any code change, because the datasets load every .ply of their folder.
#
#          # two ALS versions per cloud (headings 0 and 90 deg), next to the originals as *_als0.ply / *_als1.ply
#          python emulate_als.py -i dataset/synth/rtib3+_16classes -o dataset/synth/rtib3+_16classes \
#                                --strips --versions 2 --suffix _als
#
#          # one version into a separate folder, random heading, voxel mode, ALS point spacing of 0.25 m
#          python emulate_als.py -i in/ -o out/ --strips --random-heading --voxel 0.1 --spacing 0.25 --seed 1
#
#      Single flight lines relative to the cloud centre (small clouds / tests):
#          python emulate_als.py -i in.ply -o out/ --offsets -40 40
#          python emulate_als.py -i in/ -o out/ --random-lines 3 --returns 2 --seed 1
#

import argparse
from pathlib import Path

import numpy as np

from utils.als_simulation import (ALSConfig, FlightLine, emulate_als, random_flight_lines, strip_flight_lines,
                                  voxel_subsample)
from utils.ply import read_ply, write_ply


def flight_lines(points, args, version, rng):
    if args.random_heading:
        heading = rng.uniform(0.0, 360.0)
    else:
        # Versions differ in heading, spread over half a circle
        heading = args.heading + version * 180.0 / args.versions

    if args.strips:
        return strip_flight_lines(points, heading=heading, height=args.height, max_scan_angle=args.max_scan_angle,
                                  side_overlap=args.side_overlap, phase=rng.uniform(0.0, 1.0))
    if args.random_lines:
        return random_flight_lines(args.random_lines, height=(args.height * 0.8, args.height * 1.2),
                                   max_off_nadir=args.max_scan_angle * 0.8, max_scan_angle=args.max_scan_angle,
                                   rng=rng)
    return [FlightLine(heading=heading, offset=offset, height=args.height, max_scan_angle=args.max_scan_angle,
                       pitch=args.pitch)
            for offset in args.offsets]


def convert(in_path, out_paths, args, rng):
    data = read_ply(str(in_path))
    points = np.vstack((data['x'], data['y'], data['z'])).T
    names = list(data.dtype.names)

    for version, out_path in enumerate(out_paths):
        lines = flight_lines(points, args, version, rng)
        config = ALSConfig(flight_lines=lines,
                           footprint=args.footprint,
                           depth_tolerance=args.depth_tolerance,
                           max_returns=args.returns,
                           voxel_size=args.voxel)
        mask, returns = emulate_als(points, config, return_numbers=True)
        keep = voxel_subsample(points, args.spacing, mask=mask) if args.spacing else np.flatnonzero(mask)

        fields = [data[name][keep] for name in names] + [returns[keep].astype(np.uint8)]
        write_ply(str(out_path), fields, names + ['als_return'])
        print(f'{in_path.name} -> {out_path.name}: {len(points)} -> {len(keep)} points '
              f'({len(lines)} flight line(s), heading {lines[0].heading:.0f} deg)')


def output_paths(path, args):
    relative = Path(path.name) if args.input.is_file() else path.relative_to(args.input)
    stem = relative.stem
    if args.versions == 1:
        names = [f'{stem}{args.suffix}.ply']
    else:
        names = [f'{stem}{args.suffix}{v}.ply' for v in range(args.versions)]
    return [args.output / relative.parent / name for name in names]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Emulate ALS recordings of MLS / TLS point clouds')
    parser.add_argument('-i', '--input', type=Path, required=True, help='.ply file or folder (searched recursively)')
    parser.add_argument('-o', '--output', type=Path, required=True, help='output folder (may be the input folder)')
    parser.add_argument('--suffix', type=str, default='',
                        help='appended to the file names, e.g. _als (required when writing into the input folder)')
    parser.add_argument('--versions', type=int, default=1, help='number of ALS versions per cloud')
    parser.add_argument('--strips', action='store_true', help='parallel flight strips covering the whole cloud')
    parser.add_argument('--side-overlap', type=float, default=0.3, help='--strips: overlap of neighbouring swaths')
    parser.add_argument('--height', type=float, default=100.0, help='flying height above the lowest point [m]')
    parser.add_argument('--heading', type=float, default=0.0, help='flight direction [deg], 0 = +x')
    parser.add_argument('--random-heading', action='store_true', help='random heading per version')
    parser.add_argument('--offsets', type=float, nargs='+', default=[0.0],
                        help='single lines: lateral offsets from the cloud centre [m], one line per offset')
    parser.add_argument('--pitch', type=float, default=0.0, help='single lines: forward / backward look [deg]')
    parser.add_argument('--random-lines', type=int, default=0, help='N random lines through the cloud centre')
    parser.add_argument('--max-scan-angle', type=float, default=30.0, help='half swath angle [deg]')
    parser.add_argument('--footprint', type=float, default=0.3, help='beam footprint at nadir [m]')
    parser.add_argument('--depth-tolerance', type=float, default=0.2, help='[m]')
    parser.add_argument('--returns', type=int, default=1, help='maximum number of returns per beam')
    parser.add_argument('--voxel', type=float, default=None, help='voxel mode: visibility per voxel of this size [m]')
    parser.add_argument('--spacing', type=float, default=None,
                        help='reduce the visible points to one per voxel of this size [m] (ALS point spacing)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    in_place = args.input.resolve() == args.output.resolve() or \
        (args.input.is_file() and args.input.parent.resolve() == args.output.resolve())
    if in_place and not args.suffix:
        parser.error('writing into the input folder needs a --suffix, otherwise the originals are overwritten')

    rng = np.random.default_rng(args.seed)
    if args.input.is_file():
        inputs = [args.input]
    else:
        # Skip outputs of earlier runs when writing next to the originals
        inputs = [p for p in sorted(args.input.glob('**/*.ply')) if not (args.suffix and args.suffix in p.stem)]
    for path in inputs:
        targets = output_paths(path, args)
        targets[0].parent.mkdir(parents=True, exist_ok=True)
        convert(path, targets, args, rng)
