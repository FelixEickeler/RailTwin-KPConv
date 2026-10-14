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
#      For inspecting the emulation and exporting ALS-like clouds. Training does not need this script: with
#      config.augment_als > 0 the datasets prepare the ALS versions themselves (see utils/als_simulation.py).
#
#      The cloud is flown in parallel strips that cover it completely, so every object occludes.
#
#          # two ALS versions per cloud (headings 0 and 90 deg) as *_als0.ply / *_als1.ply
#          python emulate_als.py -i dataset/synth/rtib3+_16classes -o out/ --versions 2 --suffix _als
#
#          # ALS point spacing of 0.25 m, two returns per beam
#          python emulate_als.py -i in.ply -o out/ --spacing 0.25 --returns 2
#

import argparse
from pathlib import Path

import numpy as np

from utils.als_simulation import ALSConfig, emulate_als, strip_flight_lines, voxel_subsample
from utils.ply import read_ply, write_ply


def convert(in_path, out_paths, args, rng):
    data = read_ply(str(in_path))
    points = np.vstack((data['x'], data['y'], data['z'])).T
    names = list(data.dtype.names)

    for version, out_path in enumerate(out_paths):
        # Versions differ in heading, spread over half a circle
        heading = args.heading + version * 180.0 / args.versions
        lines = strip_flight_lines(points, heading=heading, height=args.height, max_scan_angle=args.max_scan_angle,
                                   side_overlap=args.side_overlap, phase=rng.uniform(0.0, 1.0))
        config = ALSConfig(flight_lines=lines,
                           footprint=args.footprint,
                           depth_tolerance=args.depth_tolerance,
                           max_returns=args.returns)
        mask, returns = emulate_als(points, config, return_numbers=True)
        keep = voxel_subsample(points, args.spacing, mask=mask) if args.spacing else np.flatnonzero(mask)

        fields = [data[name][keep] for name in names] + [returns[keep].astype(np.uint8)]
        write_ply(str(out_path), fields, names + ['als_return'])
        print(f'{in_path.name} -> {out_path.name}: {len(points)} -> {len(keep)} points '
              f'({len(lines)} flight strip(s), heading {heading:.0f} deg)')


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
    parser.add_argument('--side-overlap', type=float, default=0.3, help='overlap of neighbouring swaths')
    parser.add_argument('--height', type=float, default=100.0, help='flying height above the lowest point [m]')
    parser.add_argument('--heading', type=float, default=0.0, help='flight direction [deg], 0 = +x')
    parser.add_argument('--max-scan-angle', type=float, default=30.0, help='half swath angle [deg]')
    parser.add_argument('--footprint', type=float, default=0.3, help='beam footprint at nadir [m]')
    parser.add_argument('--depth-tolerance', type=float, default=0.2, help='[m]')
    parser.add_argument('--returns', type=int, default=1, help='maximum number of returns per beam')
    parser.add_argument('--spacing', type=float, default=None,
                        help='reduce the visible points to one per voxel of this size [m] (ALS point spacing)')
    parser.add_argument('--seed', type=int, default=None, help='random seed (phase of the strip pattern)')
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
