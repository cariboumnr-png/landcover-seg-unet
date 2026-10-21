# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''
Assemble multiple raster tiles into a unified GDAL Virtual Raster (VRT).
'''

# standard imports
import argparse
import pathlib
import sys
# local imports
import landseg.geopipe.harmonize as harmonize


# ----- public functions
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    '''Parse command-line arguments for building a raster mosaic VRT.'''
    parser = argparse.ArgumentParser(
        description=(
            'Harmonize and assemble multiple raster tiles into a single '
            'reprojected GDAL Virtual Raster (.vrt) mosaic.'
        )
    )
    parser.add_argument(
        'inputs',
        nargs='+',
        type=str,
        help='Input raster file paths or glob patterns to mosaic.',
    )
    parser.add_argument(
        '-o',
        '--output',
        required=True,
        type=str,
        help='Destination file path for the output Virtual Raster (.vrt).',
    )
    parser.add_argument(
        '--target-crs',
        type=str,
        default='EPSG:3161',
        help='Target coordinate reference system (default: EPSG:3161).',
    )
    parser.add_argument(
        '--target-resolution',
        nargs='+',
        type=float,
        default=None,
        help=(
            'Target pixel resolution in target CRS units (1 or 2 values: '
            'res or res_x res_y).'
        ),
    )
    parser.add_argument(
        '--resampling',
        type=str,
        default='bilinear',
        choices=[
            'nearest',
            'bilinear',
            'cubic',
            'cubic_spline',
            'lanczos',
            'average',
            'mode',
        ],
        help='Resampling method for warped tiles (default: bilinear).',
    )
    parser.add_argument(
        '--categorical',
        action='store_true',
        help='Flag indicating categorical raster (forces nearest resampling).',
    )
    parser.add_argument(
        '--nodata',
        type=float,
        default=None,
        help='Explicit nodata value override for the mosaic.',
    )
    parser.add_argument(
        '--relative-paths',
        action='store_true',
        help='Write relative source paths inside the generated VRT.',
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Directory to store intermediate warped VRT tile cache.',
    )

    return parser.parse_args(argv)


def resolve_inputs(patterns: list[str]) -> list[pathlib.Path]:
    '''Resolve file paths and wildcard glob patterns into sorted paths.'''
    resolved: list[pathlib.Path] = []
    for pattern in patterns:
        path = pathlib.Path(pattern)
        if any(char in pattern for char in ('*', '?', '[')):
            parent = path.parent if str(path.parent) != '' else pathlib.Path('.')
            matches = sorted(parent.glob(path.name))
            if not matches:
                print(f'Warning: Pattern {pattern!r} matched no files.')
            resolved.extend(matches)
        else:
            resolved.append(path)

    # remove duplicates while preserving order
    seen: set[pathlib.Path] = set()
    unique: list[pathlib.Path] = []
    for p in resolved:
        abs_p = p.resolve()
        if abs_p not in seen:
            seen.add(abs_p)
            unique.append(p)
    return unique


def main(argv: list[str] | None = None) -> None:
    '''CLI entrypoint for building a raster mosaic VRT.'''
    args = parse_args(argv)
    input_files = resolve_inputs(args.inputs)
    if not input_files:
        print('Error: No valid input files specified.', file=sys.stderr)
        sys.exit(1)

    target_res: float | tuple[float, float] | None = None
    if args.target_resolution is not None:
        if len(args.target_resolution) == 1:
            target_res = args.target_resolution[0]
        elif len(args.target_resolution) == 2:
            target_res = (args.target_resolution[0], args.target_resolution[1])
        else:
            print(
                'Error: --target-resolution requires 1 or 2 values.',
                file=sys.stderr,
            )
            sys.exit(1)

    print(f'Building VRT mosaic from {len(input_files)} raster sources...')
    out_vrt = harmonize.build_vrt_mosaic(
        input_paths=input_files,
        output_path=args.output,
        target_crs=args.target_crs,
        target_resolution=target_res,
        resampling_method=args.resampling,
        is_categorical=args.categorical,
        nodata=args.nodata,
        relative_paths=args.relative_paths,
        warped_cache_dir=args.cache_dir,
    )
    print(f'Successfully built VRT mosaic: {out_vrt}')


if __name__ == '__main__':
    main()
