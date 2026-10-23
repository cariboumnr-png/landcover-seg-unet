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
import dataclasses
import pathlib
import typing
import xml.etree.ElementTree
# third-party imports
import rasterio
import rasterio.crs
import rasterio.transform
# local imports
import landseg.geopipe.rasters.vrt as vrt


# ----- private dataclasses
@dataclasses.dataclass
class _SpatialReference:
    '''Spatial grid against which stack sources are validated.'''
    width: int
    height: int
    crs: rasterio.crs.CRS
    transform: rasterio.transform.Affine


# ----- public functions
def stack_rasters(
    source_paths: list[pathlib.Path],
    output_path: pathlib.Path,
    *,
    band_tags: list[dict[str, str]] | None = None,
    dataset_tags: dict[str, str] | None = None,
) -> pathlib.Path:
    '''
    Stack bands from spatially aligned rasters into a VRT.

    Source order determines output band order. For each source, its bands
    are appended in their native order.

    Args:
        source_paths:
            Ordered source raster paths.
        output_path:
            Destination VRT path.
        dataset_tags:
            Optional metadata attached to the output dataset.
        band_tags:
            Optional metadata for each output band. Its length must match
            the total number of source bands. A `None` entry leaves the
            corresponding output band untagged.

    Returns:
        Resolved path to the created VRT.

    Raises:
        ValueError:
            If no sources are provided, sources are not spatially aligned,
            or the number of band metadata entries is incorrect.
        FileNotFoundError:
            If a source raster does not exist.
    '''
    if not source_paths:
        raise ValueError('source_paths list cannot be empty.')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ref = _get_spatial_reference(source_paths[0]) # from the first raster

    root = vrt.create_vrt_root(ref.width, ref.height, ref.crs, ref.transform)

    val_band_tags = _validate_band_tags(source_paths, band_tags)

    band_idx = 1
    for path in source_paths:
        with rasterio.open(path) as src:

            _validate_source(src, path, ref)

            for src_band in range(1, src.count + 1):
                nodata_val = (
                    src.nodatavals[src_band - 1]
                    if src.nodatavals
                    and src.nodatavals[src_band - 1] is not None
                    else src.nodata
                )

                band_node = vrt.add_vrt_band(
                    root=root,
                    band_idx=band_idx,
                    dtype=src.dtypes[src_band - 1],
                    nodata=nodata_val,
                    description=src.descriptions[src_band - 1],
                )

                band_meta = val_band_tags[band_idx - 1]
                if band_meta:
                    _add_band_metadata(band_node, band_meta)

                vrt.add_vrt_source(
                    band_node=band_node,
                    source_path=path,
                    source_band=src_band,
                    width=src.width,
                    height=src.height,
                    dtype=src.dtypes[src_band - 1],
                )

                band_idx += 1

    if dataset_tags:
        _add_band_metadata(root, dataset_tags)

    return vrt.write_vrt(root, output_path)


# ----- private helpers
def _get_spatial_reference(source_path: pathlib.Path) -> _SpatialReference:
    '''Read raster and retrieve spatial reference.'''
    with rasterio.open(source_path) as source:
        if source.crs is None:
            raise ValueError(f'Source raster has no CRS: {source_path}')
        return _SpatialReference(
            width=source.width,
            height=source.height,
            crs=source.crs,
            transform=source.transform,
        )


def _validate_band_tags(
    source_paths: list[pathlib.Path],
    band_tags: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    '''Return the total number of bands across all sources.'''
    total = 0
    for source_path in source_paths:
        with rasterio.open(source_path) as source:
            total += source.count

    if band_tags is None:
        return [{}] * total

    if len(band_tags) != total:
        raise ValueError(
            'band_tags length must equal the total source band count: '
            f'{len(band_tags)} != {total}'
        )

    return list(band_tags)


def _validate_source(
    src: rasterio.DatasetReader,
    path: pathlib.Path,
    ref: _SpatialReference,
) -> None:
    '''Validate that source raster matches reference dimensions and CRS.'''
    if src.width != ref.width or src.height != ref.height:
        raise ValueError(
            f'Source raster has different dimensions: {path} '
            f'({src.width}x{src.height} != {ref.width}x{ref.height})'
        )

    if src.crs != ref.crs:
        raise ValueError(f'Source raster has a different CRS: {path}')

    if not src.transform.almost_equals(ref.transform):
        raise ValueError(
            f'Source raster transform differs from the reference grid: {path}'
        )


def _add_band_metadata(
    band_node: xml.etree.ElementTree.Element,
    metadata: dict[str, typing.Any]
) -> None:
    '''Attach band tags XML metadata node to VRT band node.'''
    metadata_node = xml.etree.ElementTree.SubElement(band_node, 'Metadata')
    for key, value in metadata.items():
        xml.etree.ElementTree.SubElement(
            metadata_node,
            'MDI',
            key=str(key),
        ).text = str(value)
