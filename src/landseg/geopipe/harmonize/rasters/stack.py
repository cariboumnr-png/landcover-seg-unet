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
Multi-raster channel composition and stacking operations.

This module provides functionality to combine multiple feature and label
rasters into unified multi-band GDAL Virtual Raster (VRT) composites.

Public APIs:
    - `stack_rasters`: Stack feature and label rasters into composite VRTs.
'''

# standard imports
from __future__ import annotations
import ast
import pathlib
import typing
import xml.etree.ElementTree
# third-party imports
import rasterio
import rasterio.crs
# local imports
import landseg.geopipe.harmonize.rasters.utils as raster_utils


# ----- public functions
def stack_rasters(
    features_fpaths: list[pathlib.Path],
    labels_fpaths: list[pathlib.Path],
    output_dir: pathlib.Path,
) -> typing.Generator[str, None, dict[str, pathlib.Path]]:
    '''
    Stack feature and label rasters into composite VRT files.

    Args:
        features_fpaths:
            List of file paths for feature rasters.
        labels_fpaths:
            List of file paths for label rasters.
        output_dir:
            Directory path where stacked VRT files will be saved.

    Yields:
        str:
            Status messages during stacking process.

    Returns:
        dict[str, pathlib.Path]:
            Mapping of raster role ('features', 'labels') to stacked
            VRT file path.
    '''
    output_dir = pathlib.Path(output_dir)

    def _out_path(tag: str) -> pathlib.Path:
        return output_dir / f'harmonized_{tag}_STACKED.vrt'

    stacked: dict[str, pathlib.Path] = {}
    yield 'Stacking rasters if applicable'

    # features
    n = len(features_fpaths)
    if n == 0:
        pass
    elif n == 1:
        stacked.update({'features': features_fpaths[0]})
    else:
        out_path = _out_path('features')
        _composite_vrt(features_fpaths, out_path, 'feature')
        stacked.update({'features': out_path})
        yield f'Feature rasters stacked to {out_path} (n={n})'

    # labels
    n = len(labels_fpaths)
    if n == 0:
        pass
    elif n == 1:
        stacked.update({'labels': labels_fpaths[0]})
    else:
        out_path = _out_path('labels')
        _composite_vrt(labels_fpaths, out_path, 'label')
        stacked.update({'labels': out_path})
        yield f'Label rasters stacked to {out_path} (n={n})'

    return stacked


# ----- private helpers
def _composite_vrt(
    source_paths: list[pathlib.Path],
    output_path: pathlib.Path,
    raster_type: typing.Literal['feature', 'label']
) -> pathlib.Path:
    '''Stack multiple rasters into one composite VRT.'''
    if not source_paths:
        raise ValueError('source_paths list cannot be empty.')

    out_p = output_path.resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    resolved_sources = [str(pathlib.Path(p).resolve()) for p in source_paths]
    _build_stacked_vrt_xml(resolved_sources, output_path, raster_type)

    return out_p


def _build_stacked_vrt_xml(
    source_paths: list[str],
    output_path: pathlib.Path,
    raster_type: typing.Literal['feature', 'label']
) -> None:
    '''Build a VRT by stacking bands from multiple source rasters.'''
    if not source_paths:
        raise ValueError('source_paths must not be empty')

    width, height, crs, transform = _get_reference_grid(source_paths[0])
    root = raster_utils.create_vrt_root(width, height, crs, transform)

    merged_schemes: dict[str, typing.Any] = {}
    band_idx = 1

    for path in source_paths:
        with rasterio.open(path) as src:
            _validate_source(src, path, width, height, crs)

            if raster_type == 'feature':
                _merge_feature_schemes(
                    merged_schemes,
                    src.tags().get('schemes'),
                )

            for src_band in range(1, src.count + 1):
                nodata_val = (
                    src.nodatavals[src_band - 1]
                    if src.nodatavals
                    and src.nodatavals[src_band - 1] is not None
                    else src.nodata
                )

                band_node = raster_utils.add_vrt_band(
                    root=root,
                    band_idx=band_idx,
                    dtype=src.dtypes[src_band - 1],
                    nodata=nodata_val,
                    description=src.descriptions[src_band - 1],
                )

                _add_band_metadata(
                    band_node=band_node,
                    src=src,
                    raster_type=raster_type,
                )

                raster_utils.add_vrt_source(
                    band_node=band_node,
                    source_path=path,
                    source_band=src_band,
                    width=src.width,
                    height=src.height,
                    dtype=src.dtypes[src_band - 1],
                )

                band_idx += 1

    if raster_type == 'feature':
        _add_dataset_schemes(root, merged_schemes)

    raster_utils.write_vrt(root, output_path)


def _get_reference_grid(
    source_path: str,
) -> tuple[int, int, rasterio.crs.CRS, rasterio.Affine]:
    '''Extract dimensions, CRS, and transform from reference raster.'''
    with rasterio.open(source_path) as src:
        return (src.width, src.height, src.crs, src.transform)


def _validate_source(
    src: rasterio.DatasetReader,
    path: str,
    width: int,
    height: int,
    crs: rasterio.crs.CRS,
) -> None:
    '''Validate that source raster matches reference dimensions and CRS.'''
    if src.width != width or src.height != height:
        raise ValueError(
            f'Source raster has different dimensions: {path} '
            f'({src.width}x{src.height} != {width}x{height})'
        )

    if src.crs != crs:
        raise ValueError(f'Source raster has a different CRS: {path}')


def _add_band_metadata(
    *,
    band_node: xml.etree.ElementTree.Element,
    src: rasterio.DatasetReader,
    raster_type: typing.Literal['feature', 'label'],
) -> None:
    '''Attach band tags XML metadata node to VRT band node.'''
    if raster_type == 'feature':
        return

    # labels are expected to be single-band,
    # so dataset-level metadata describes the output label band.
    tags = src.tags()
    if not tags:
        return

    metadata_node = xml.etree.ElementTree.SubElement(band_node, 'Metadata')
    for key, value in tags.items():
        xml.etree.ElementTree.SubElement(
            metadata_node,
            'MDI',
            key=str(key),
        ).text = str(value)


def _merge_feature_schemes(
    merged: dict[str, typing.Any],
    schemes_as_str: str | None,
) -> None:
    '''Merge feature schemes dictionary into the target schemes map.'''
    if schemes_as_str is None:
        return

    try:
        schemes_dict = ast.literal_eval(schemes_as_str)
    except (ValueError, SyntaxError):
        return

    if not isinstance(schemes_dict, dict):
        return

    for key, val in schemes_dict.items():
        if isinstance(val, dict):
            target = merged.setdefault(key, {})
            for scheme_name, bands in val.items():
                existing = target.get(scheme_name)
                if existing is not None and existing != bands:
                    raise ValueError(
                        f'Conflicting definitions for feature scheme '
                        f'"{scheme_name}" under raster "{key}": '
                        f'{existing!r} != {bands!r}'
                    )
                target[scheme_name] = bands
        elif isinstance(val, list):
            existing = merged.get(key)
            if existing is not None and existing != val:
                raise ValueError(
                    f'Conflicting definitions for feature scheme "{key}": '
                    f'{existing!r} != {val!r}'
                )
            merged[key] = val


def _add_dataset_schemes(
    root: xml.etree.ElementTree.Element,
    schemes: dict[str, typing.Any],
) -> None:
    '''Add merged feature schemes as VRT dataset metadata.'''
    if not schemes:
        return

    metadata_node = xml.etree.ElementTree.SubElement(
        root,
        'Metadata',
    )

    xml.etree.ElementTree.SubElement(
        metadata_node,
        'MDI',
        key='schemes',
    ).text = str(schemes)
