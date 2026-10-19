# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Multi-unit raster mosaicking and virtual coordinate reprojection.

This module provides utilities to assemble disparate raster tiles (such
as Ontario Crown Forest Management Units managed in separate UTM zones)
into a unified, reprojected GDAL Virtual Raster (VRT) mosaic without
materializing intermediate imagery arrays to disk.

Public APIs:
    - `build_vrt_mosaic`: Assemble rasters into a warped VRT mosaic.
'''

# standard imports
import dataclasses
import math
import pathlib
import typing
# third-party imports
import rasterio
import rasterio.crs
import rasterio.enums
import rasterio.transform
import rasterio.warp
# local imports
import landseg.geopipe.harmonize.rasters.utils as raster_utils


# ----- private dataclasses
@dataclasses.dataclass(frozen=True)
class _TileGeometry:
    '''Calculated spatial geometry and grid bounds for a raster tile.'''
    vrt_path: pathlib.Path
    min_x: float
    max_y: float
    width: int
    height: int


@dataclasses.dataclass(frozen=True)
class _WarpContext:
    '''Context parameters for virtual tile warping and snapping.'''
    dst_crs: rasterio.crs.CRS
    res_x: float
    res_y: float
    resample_alg: rasterio.enums.Resampling
    nodata_val: float | int | None
    cache_dir: pathlib.Path


@dataclasses.dataclass(frozen=True)
class _MosaicMetadata:
    '''Structural raster metadata across bands for the mosaic.'''
    band_count: int
    dtypes: tuple[str, ...]
    descriptions: tuple[str | None, ...]
    nodata_val: float | int | None


@dataclasses.dataclass(frozen=True)
class _MosaicLayout:
    '''Calculated spatial geometry and grid layout for the mosaic.'''
    min_x: float
    max_y: float
    width: int
    height: int
    res_x: float
    res_y: float
    transform: rasterio.transform.Affine


# ----- public functions
def build_vrt_mosaic( # pylint: disable=too-many-arguments,too-many-locals
    *,
    input_paths: typing.Sequence[pathlib.Path | str],
    output_path: pathlib.Path | str,
    target_crs: str | rasterio.crs.CRS = 'EPSG:3161',
    target_resolution: float | tuple[float, float] | None = None,
    resampling_method: str = 'bilinear',
    is_categorical: bool = False,
    nodata: float | int | None = None,
    relative_paths: bool = False,
    warped_cache_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    '''
    Build a unified GDAL Virtual Raster (.vrt) mosaic from input rasters.

    This function harmonizes multiple raster sources (such as management
    units in differing UTM projections) into a single virtual mosaic
    aligned to a canonical provincial grid without duplicating imagery.

    Args:
        input_paths:
            Sequence of paths to input raster files to mosaic.
        output_path:
            Destination path for the composite Virtual Raster (.vrt).
        target_crs:
            Coordinate reference system for the mosaic. Defaults to
            'EPSG:3161'.
        target_resolution:
            Optional pixel resolution (res_x, res_y) in target CRS
            units. If None, uses the minimum pixel resolution across
            all inputs projected to target CRS.
        resampling_method:
            Resampling method for warping ('bilinear', 'nearest',
            'cubic'). Defaults to 'bilinear'.
        is_categorical:
            If True, strictly forces nearest-neighbor resampling.
        nodata:
            Explicit nodata value override. If None, inherits source
            nodata value or falls back to 255 for categorical rasters.
        relative_paths:
            If True, writes source paths relative to the output VRT.
        warped_cache_dir:
            Directory to store intermediate warped VRT files if
            reprojection is required. Defaults to a hidden folder next
            to the output VRT.

    Returns:
        pathlib.Path:
            Absolute path to the generated master mosaic VRT file.
    '''
    if not input_paths:
        raise ValueError('input_paths cannot be empty.')

    resolved_inputs = [pathlib.Path(p).resolve() for p in input_paths]
    for p in resolved_inputs:
        if not p.exists():
            raise FileNotFoundError(f'Source raster not found: {p}')

    out_p = pathlib.Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    dst_crs = (
        rasterio.crs.CRS.from_user_input(target_crs)
        if isinstance(target_crs, str)
        else target_crs
    )

    ref_info = _read_reference_info(resolved_inputs[0])
    for p in resolved_inputs[1:]:
        _validate_source_compatibility(ref_info, p)

    res_x, res_y = _determine_target_resolution(
        resolved_inputs,
        dst_crs,
        target_resolution,
    )

    final_nodata = _resolve_nodata_value(
        nodata,
        ref_info['nodata'],
        is_categorical,
    )

    cache_dir = (
        pathlib.Path(warped_cache_dir).resolve()
        if warped_cache_dir is not None
        else out_p.parent / f'.{out_p.stem}_warped'
    )

    warp_ctx = _WarpContext(
        dst_crs=dst_crs,
        res_x=res_x,
        res_y=res_y,
        resample_alg=raster_utils.resolve_resampling_algorithm(
            resampling_method,
            is_categorical,
        ),
        nodata_val=final_nodata,
        cache_dir=cache_dir,
    )

    tiles = _prepare_source_tiles(resolved_inputs, warp_ctx)

    meta = _MosaicMetadata(
        band_count=ref_info['count'],
        dtypes=ref_info['dtypes'],
        descriptions=ref_info['descriptions'],
        nodata_val=final_nodata,
    )

    _build_mosaic_vrt_xml(
        tiles=tiles,
        output_path=out_p,
        dst_crs=dst_crs,
        res_x=res_x,
        res_y=res_y,
        meta=meta,
        relative_paths=relative_paths,
    )

    return out_p


# ----- private helpers
def _read_reference_info(path: pathlib.Path) -> dict[str, typing.Any]:
    '''Read structural reference metadata from the first raster.'''
    with rasterio.open(path) as src:
        return {
            'count': src.count,
            'dtypes': src.dtypes,
            'descriptions': src.descriptions,
            'nodata': src.nodata,
            'crs': src.crs,
            'res': src.res,
        }


def _validate_source_compatibility(
    ref_info: dict[str, typing.Any],
    path: pathlib.Path,
) -> None:
    '''Validate that secondary source matches band count and types.'''
    with rasterio.open(path) as src:
        if src.count != ref_info['count']:
            raise ValueError(
                f'Band count mismatch for {path}: expected '
                f"{ref_info['count']}, found {src.count}"
            )
        if src.dtypes != ref_info['dtypes']:
            raise ValueError(
                f'Data types mismatch for {path}: expected '
                f"{ref_info['dtypes']}, found {src.dtypes}"
            )


def _determine_target_resolution(
    inputs: list[pathlib.Path],
    dst_crs: rasterio.crs.CRS,
    target_res: float | tuple[float, float] | None,
) -> tuple[float, float]:
    '''Resolve pixel resolution in target CRS coordinate units.'''
    if target_res is not None:
        if isinstance(target_res, (int, float)):
            return (float(target_res), float(target_res))
        return (float(target_res[0]), float(target_res[1]))

    res_xs: list[float] = []
    res_ys: list[float] = []

    for p in inputs:
        with rasterio.open(p) as src:
            if src.crs == dst_crs:
                res_xs.append(abs(src.res[0]))
                res_ys.append(abs(src.res[1]))
            else:
                _, w, h = rasterio.warp.calculate_default_transform(
                    src.crs,
                    dst_crs,
                    src.width,
                    src.height,
                    *src.bounds,
                )
                left, bottom, right, top = (
                    rasterio.warp.transform_bounds(
                        src.crs,
                        dst_crs,
                        *src.bounds,
                    )
                )
                res_xs.append(abs((right - left) / w))
                res_ys.append(abs((top - bottom) / h))

    return (min(res_xs), min(res_ys))


def _resolve_nodata_value(
    nodata_arg: float | int | None,
    source_nodata: float | int | None,
    is_categorical: bool,
) -> float | int | None:
    '''Resolve effective nodata value across all tiles.'''
    if nodata_arg is not None:
        return nodata_arg
    if source_nodata is not None:
        return source_nodata
    if is_categorical:
        return 255
    return None


def _prepare_source_tiles(
    inputs: list[pathlib.Path],
    ctx: _WarpContext,
) -> list[_TileGeometry]:
    '''Reproject and snap all source tiles to a uniform coordinate grid.'''
    return [
        _prepare_single_tile(idx, p, ctx)
        for idx, p in enumerate(inputs)
    ]


def _calculate_tile_bounds(
    src: rasterio.DatasetReader,
    ctx: _WarpContext,
) -> tuple[float, float, int, int, rasterio.transform.Affine]:
    '''Calculate snapped tile bounds and transform in target CRS.'''
    left, bottom, right, top = (
        src.bounds
        if src.crs == ctx.dst_crs
        else rasterio.warp.transform_bounds(
            src.crs,
            ctx.dst_crs,
            *src.bounds,
        )
    )

    min_x = math.floor(left / ctx.res_x) * ctx.res_x
    max_y = math.ceil(top / ctx.res_y) * ctx.res_y
    max_x = math.ceil(right / ctx.res_x) * ctx.res_x
    min_y = math.floor(bottom / ctx.res_y) * ctx.res_y

    width = int(round((max_x - min_x) / ctx.res_x))
    height = int(round((max_y - min_y) / ctx.res_y))

    tile_transform = (
        rasterio.transform.Affine.translation(min_x, max_y)
        * rasterio.transform.Affine.scale(ctx.res_x, -ctx.res_y)
    )

    return (min_x, max_y, width, height, tile_transform)


def _prepare_single_tile(
    idx: int,
    path: pathlib.Path,
    ctx: _WarpContext,
) -> _TileGeometry:
    '''Reproject and snap a single raster tile to target grid.'''
    with rasterio.open(path) as src:
        min_x, max_y, width, height, tile_transform = (
            _calculate_tile_bounds(src, ctx)
        )

        needs_warp = (
            src.crs != ctx.dst_crs
            or abs(src.res[0] - ctx.res_x) > 1e-5
            or abs(src.res[1] - ctx.res_y) > 1e-5
            or abs(src.transform.c - min_x) > 1e-5
            or abs(src.transform.f - max_y) > 1e-5
        )

        if needs_warp:
            vrt_file = ctx.cache_dir / f'warped_{idx:04d}_{path.stem}.vrt'
            raster_utils.write_warped_vrt(
                src=src,
                output_path=vrt_file,
                crs=ctx.dst_crs,
                transform=tile_transform,
                width=width,
                height=height,
                resampling=ctx.resample_alg,
                nodata=ctx.nodata_val,
            )
            ref_path = vrt_file
        else:
            ref_path = path

        return _TileGeometry(
            vrt_path=ref_path,
            min_x=min_x,
            max_y=max_y,
            width=width,
            height=height,
        )


def _calculate_mosaic_layout(
    tiles: list[_TileGeometry],
    res_x: float,
    res_y: float,
) -> _MosaicLayout:
    '''Calculate total bounds and transform for the mosaic dataset.'''
    min_x = min(t.min_x for t in tiles)
    max_y = max(t.max_y for t in tiles)
    max_x = max(t.min_x + t.width * res_x for t in tiles)
    min_y = min(t.max_y - t.height * res_y for t in tiles)

    width = int(round((max_x - min_x) / res_x))
    height = int(round((max_y - min_y) / res_y))

    transform = (
        rasterio.transform.Affine.translation(min_x, max_y)
        * rasterio.transform.Affine.scale(res_x, -res_y)
    )

    return _MosaicLayout(
        min_x=min_x,
        max_y=max_y,
        width=width,
        height=height,
        res_x=res_x,
        res_y=res_y,
        transform=transform,
    )


def _build_mosaic_vrt_xml( # pylint: disable=too-many-arguments
    *,
    tiles: list[_TileGeometry],
    output_path: pathlib.Path,
    dst_crs: rasterio.crs.CRS,
    res_x: float,
    res_y: float,
    meta: _MosaicMetadata,
    relative_paths: bool,
) -> None:
    '''Construct master GDAL VRT XML connecting all mosaic tiles.'''
    layout = _calculate_mosaic_layout(tiles, res_x, res_y)
    root = raster_utils.create_vrt_root(
        width=layout.width,
        height=layout.height,
        crs=dst_crs,
        transform=layout.transform,
    )

    for band_idx in range(1, meta.band_count + 1):
        desc = (
            meta.descriptions[band_idx - 1]
            if meta.descriptions and meta.descriptions[band_idx - 1]
            else f'band_{band_idx}'
        )
        band_node = raster_utils.add_vrt_band(
            root=root,
            band_idx=band_idx,
            dtype=meta.dtypes[band_idx - 1],
            nodata=meta.nodata_val,
            description=desc,
        )

        for t in tiles:
            dst_x = int(round((t.min_x - layout.min_x) / layout.res_x))
            dst_y = int(round((layout.max_y - t.max_y) / layout.res_y))

            raster_utils.add_vrt_source(
                band_node=band_node,
                source_path=t.vrt_path,
                source_band=band_idx,
                width=t.width,
                height=t.height,
                dtype=meta.dtypes[band_idx - 1],
                dst_rect=(dst_x, dst_y, t.width, t.height),
                src_rect=(0, 0, t.width, t.height),
                nodata=meta.nodata_val,
                relative_paths=relative_paths,
                output_dir=output_path.parent,
            )

    raster_utils.write_vrt(root, output_path)
