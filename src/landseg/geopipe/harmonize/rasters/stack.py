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
import os
import typing
import xml.etree.ElementTree
# third-party imports
import rasterio
import rasterio.crs


# ----- public functions
def stack_rasters(
    features_fapths: list[str],
    labels_fapths: list[str],
    output_dir: str,
) -> typing.Generator[str, None, dict[str, str]]:
    '''
    Stack feature and label rasters into composite VRT files.

    Args:
        features_fapths:
            List of file paths for feature rasters.
        labels_fapths:
            List of file paths for label rasters.
        output_dir:
            Directory path where stacked VRT files will be saved.

    Yields:
        str:
            Status messages during stacking process.

    Returns:
        dict[str, str]:
            Mapping of raster role ('features', 'labels') to stacked
            VRT file path.
    '''
    def _out_path(tag: str) -> str:
        return os.path.join(output_dir, f'harmonized_{tag}_STACKED.vrt')

    stacked: dict[str, str] = {}
    yield 'Stacking rasters if applicable'

    # features
    n = len(features_fapths)
    if n == 0:
        pass
    elif n == 1:
        stacked.update({'features': features_fapths[0]})
    else:
        out_path = _out_path('features')
        _composite_vrt(features_fapths, out_path, 'feature')
        stacked.update({'features': out_path})
        yield f'Feature rasters stacked to {out_path} (n={n})'

    # labels
    n = len(labels_fapths)
    if n == 0:
        pass
    elif n == 1:
        stacked.update({'labels': labels_fapths[0]})
    else:
        out_path = _out_path('labels')
        _composite_vrt(labels_fapths, out_path, 'label')
        stacked.update({'labels': out_path})
        yield f'Label rasters stacked to {out_path} (n={n})'

    return stacked


# ----- private helpers
def _composite_vrt(
    source_paths: list[str],
    output_path: str,
    raster_type: typing.Literal['feature', 'label']
) -> str:
    '''Stack multiple rasters into one composite VRT.'''
    if not source_paths:
        raise ValueError('source_paths list cannot be empty.')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    source_paths = [os.path.abspath(p) for p in source_paths]
    _build_stacked_vrt_xml(source_paths, output_path, raster_type)

    return os.path.abspath(output_path)


def _build_stacked_vrt_xml(
    source_paths: list[str],
    output_path: str,
    raster_type: typing.Literal['feature', 'label']
) -> None:
    '''Build a VRT by stacking bands from multiple source rasters.'''
    if not source_paths:
        raise ValueError('source_paths must not be empty')

    width, height, crs, transform = _get_reference_grid(source_paths[0])

    root = _create_vrt_root(width, height, crs, transform)

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

                dtype = _gdal_dtype_name(src.dtypes[src_band - 1])

                band_node = xml.etree.ElementTree.SubElement(
                    root,
                    'VRTRasterBand',
                    dataType=dtype,
                    band=str(band_idx),
                )

                _add_band_description(
                    band_node=band_node,
                    description=src.descriptions[src_band - 1],
                    output_band=band_idx
                )

                _add_band_metadata(
                    band_node=band_node,
                    src=src,
                    raster_type=raster_type
                )

                _add_nodata(
                    band_node=band_node,
                    src=src,
                    src_band=src_band
                )

                _add_source(
                    band_node=band_node,
                    src=src,
                    source_path=path,
                    source_band=src_band,
                    dtype=dtype
                )

                band_idx += 1

    if raster_type == 'feature':
        _add_dataset_schemes(root, merged_schemes)

    # write
    xml.etree.ElementTree.indent(root, space='  ', level=0)
    xml.etree.ElementTree.ElementTree(root).write(
        output_path,
        encoding='utf-8',
        xml_declaration=True,
    )


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


def _create_vrt_root(
    width: int,
    height: int,
    crs: rasterio.crs.CRS,
    transform: rasterio.Affine,
) -> xml.etree.ElementTree.Element:
    '''Create root XML element for VRT dataset with CRS and GeoTransform.'''
    transform_txt = (
        f'{transform.c}, {transform.a}, {transform.b}, '
        f'{transform.f}, {transform.d}, {transform.e}'
    )

    root = xml.etree.ElementTree.Element(
        'VRTDataset',
        rasterXSize=str(width),
        rasterYSize=str(height),
    )
    xml.etree.ElementTree.SubElement(root, 'SRS').text = crs.to_wkt()
    xml.etree.ElementTree.SubElement(root, 'GeoTransform').text = transform_txt

    return root


def _gdal_dtype_name(dtype) -> str:
    '''Map rasterio/numpy data type name to canonical GDAL data type name.'''
    gdal_dtype_map = {
        'uint8': 'Byte',
        'int8': 'Int8',
        'uint16': 'UInt16',
        'int16': 'Int16',
        'uint32': 'UInt32',
        'int32': 'Int32',
        'float32': 'Float32',
        'float64': 'Float64',
    }
    s = str(dtype).lower()
    return gdal_dtype_map.get(s, s.capitalize())


def _add_band_description(
    *,
    band_node: xml.etree.ElementTree.Element,
    description: str | None,
    output_band: int,
) -> None:
    '''Attach band description XML node to VRT band node.'''
    if description and description.strip():
        name = description.strip()
    else:
        name = f'band_{output_band}'
    xml.etree.ElementTree.SubElement(band_node, 'Description').text = name


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


def _add_nodata(
    *,
    band_node: xml.etree.ElementTree.Element,
    src: rasterio.DatasetReader,
    src_band: int,
) -> None:
    '''Attach NoDataValue XML element to VRT band node if defined.'''
    nodata_val = (
        src.nodatavals[src_band - 1]
        if src.nodatavals
        and src.nodatavals[src_band - 1] is not None
        else src.nodata
    )

    if nodata_val is not None:
        xml.etree.ElementTree.SubElement(
            band_node,
            'NoDataValue',
        ).text = str(nodata_val)


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


def _add_source(
    *,
    band_node: xml.etree.ElementTree.Element,
    src: rasterio.DatasetReader,
    source_path: str,
    source_band: int,
    dtype: str,
) -> None:
    '''Attach SimpleSource XML node for source band to VRT band node.'''
    source_node = xml.etree.ElementTree.SubElement(
        band_node,
        'SimpleSource',
    )

    xml.etree.ElementTree.SubElement(
        source_node,
        'SourceFilename',
        relativeToVRT='0',
    ).text = source_path

    xml.etree.ElementTree.SubElement(
        source_node,
        'SourceBand',
    ).text = str(source_band)

    xml.etree.ElementTree.SubElement(
        source_node,
        'SourceProperties',
        RasterXSize=str(src.width),
        RasterYSize=str(src.height),
        DataType=dtype,
    )
