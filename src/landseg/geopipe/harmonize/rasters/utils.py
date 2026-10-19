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
Shared GDAL Virtual Raster (VRT) utilities and XML builders.

This module provides reusable primitives for constructing GDAL VRT
XML definitions, attaching bands and sources, mapping data types, and
materializing virtual warped raster definitions.

Public APIs:
    - `add_vrt_band`: Append a VRTRasterBand element to VRT dataset.
    - `add_vrt_source`: Attach a SimpleSource or ComplexSource element.
    - `create_vrt_root`: Create root XML element for a VRT dataset.
    - `gdal_dtype_name`: Map Python/rasterio dtype to canonical GDAL name.
    - `write_vrt`: Serialize and write indented VRT XML to disk.
    - `write_warped_vrt`: Write a virtual warped raster (.vrt) to disk.
'''

# standard imports
import os
import pathlib
import typing
import xml.etree.ElementTree
# third-party imports
import numpy
import rasterio
import rasterio.crs
import rasterio.enums
import rasterio.shutil
import rasterio.transform
import rasterio.vrt


# ----- public functions
def gdal_dtype_name(dtype: str | typing.Any) -> str:
    '''
    Map rasterio/numpy data type name to canonical GDAL data type name.

    Args:
        dtype:
            Data type string or object (e.g. 'uint8', 'float32').

    Returns:
        str:
            Canonical GDAL data type name (e.g. 'Byte', 'Float32').
    '''
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
    try:
        s = numpy.dtype(dtype).name.lower()
    except (TypeError, ValueError):
        s = str(dtype).lower()
    return gdal_dtype_map.get(s, s.capitalize())


def create_vrt_root(
    width: int,
    height: int,
    crs: rasterio.crs.CRS,
    transform: rasterio.transform.Affine,
) -> xml.etree.ElementTree.Element:
    '''
    Create root XML element for VRT dataset with CRS and GeoTransform.

    Args:
        width:
            Raster width in pixels.
        height:
            Raster height in pixels.
        crs:
            Coordinate reference system.
        transform:
            Affine geotransform.

    Returns:
        xml.etree.ElementTree.Element:
            The root VRTDataset XML element.
    '''
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


def add_vrt_band(
    root: xml.etree.ElementTree.Element,
    band_idx: int,
    dtype: str,
    nodata: float | int | None = None,
    description: str | None = None,
) -> xml.etree.ElementTree.Element:
    '''
    Append a VRTRasterBand node to the VRT dataset root element.

    Args:
        root:
            The root VRTDataset XML element.
        band_idx:
            1-based band index.
        dtype:
            Data type of the band.
        nodata:
            Optional nodata value to attach.
        description:
            Optional description label for the band.

    Returns:
        xml.etree.ElementTree.Element:
            The created VRTRasterBand XML element.
    '''
    band_node = xml.etree.ElementTree.SubElement(
        root,
        'VRTRasterBand',
        dataType=gdal_dtype_name(dtype),
        band=str(band_idx),
    )
    if nodata is not None:
        xml.etree.ElementTree.SubElement(
            band_node,
            'NoDataValue',
        ).text = str(nodata)

    desc = (
        description
        if description and description.strip()
        else f'band_{band_idx}'
    )
    xml.etree.ElementTree.SubElement(band_node, 'Description').text = desc
    return band_node


def add_vrt_source( # pylint: disable=too-many-arguments,too-many-locals
    *,
    band_node: xml.etree.ElementTree.Element,
    source_path: pathlib.Path | str,
    source_band: int,
    width: int,
    height: int,
    dtype: str,
    dst_rect: tuple[int, int, int, int] | None = None,
    src_rect: tuple[int, int, int, int] | None = None,
    nodata: float | int | None = None,
    relative_paths: bool = False,
    output_dir: pathlib.Path | None = None,
) -> xml.etree.ElementTree.Element:
    '''
    Attach a SimpleSource or ComplexSource XML node to a band node.

    Args:
        band_node:
            The VRTRasterBand XML element.
        source_path:
            Path to the underlying raster file.
        source_band:
            1-based band index within the source raster.
        width:
            Raster width in pixels.
        height:
            Raster height in pixels.
        dtype:
            Data type of the band.
        dst_rect:
            Optional (xOff, yOff, xSize, ySize) tuple in destination pixels.
        src_rect:
            Optional (xOff, yOff, xSize, ySize) tuple in source pixels.
        nodata:
            Optional nodata value. If provided, uses ComplexSource.
        relative_paths:
            If True, writes relative path relative to output_dir.
        output_dir:
            Parent directory of destination VRT, required if relative_paths.

    Returns:
        xml.etree.ElementTree.Element:
            The created source XML element.
    '''
    src_tag = 'ComplexSource' if nodata is not None else 'SimpleSource'
    src_node = xml.etree.ElementTree.SubElement(band_node, src_tag)

    if relative_paths and output_dir is not None:
        file_ref = os.path.relpath(source_path, output_dir)
        rel_val = '1'
    else:
        file_ref = str(pathlib.Path(source_path).resolve())
        rel_val = '0'

    xml.etree.ElementTree.SubElement(
        src_node,
        'SourceFilename',
        relativeToVRT=rel_val,
    ).text = file_ref

    xml.etree.ElementTree.SubElement(
        src_node,
        'SourceBand',
    ).text = str(source_band)

    xml.etree.ElementTree.SubElement(
        src_node,
        'SourceProperties',
        RasterXSize=str(width),
        RasterYSize=str(height),
        DataType=gdal_dtype_name(dtype),
    )

    src_x, src_y, src_w, src_h = src_rect or (0, 0, width, height)
    xml.etree.ElementTree.SubElement(
        src_node,
        'SrcRect',
        xOff=str(src_x),
        yOff=str(src_y),
        xSize=str(src_w),
        ySize=str(src_h),
    )

    dst_x, dst_y, dst_w, dst_h = dst_rect or (0, 0, width, height)
    xml.etree.ElementTree.SubElement(
        src_node,
        'DstRect',
        xOff=str(dst_x),
        yOff=str(dst_y),
        xSize=str(dst_w),
        ySize=str(dst_h),
    )

    if nodata is not None:
        xml.etree.ElementTree.SubElement(
            src_node,
            'NODATA',
        ).text = str(nodata)

    return src_node


def write_vrt(
    root: xml.etree.ElementTree.Element,
    output_path: pathlib.Path | str,
) -> pathlib.Path:
    '''
    Indent and serialize VRT XML dataset element tree to disk.

    Args:
        root:
            The root VRTDataset XML element.
        output_path:
            Target destination path for the .vrt file.

    Returns:
        pathlib.Path:
            Resolved absolute path to the written VRT file.
    '''
    out_p = pathlib.Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    xml.etree.ElementTree.indent(root, space='  ', level=0)
    xml.etree.ElementTree.ElementTree(root).write(
        out_p,
        encoding='utf-8',
        xml_declaration=True,
    )
    return out_p


def write_warped_vrt( # pylint: disable=too-many-arguments
    *,
    src: rasterio.DatasetReader,
    output_path: pathlib.Path | str,
    crs: rasterio.crs.CRS,
    transform: rasterio.transform.Affine,
    width: int,
    height: int,
    resampling: rasterio.enums.Resampling = rasterio.enums.Resampling.bilinear,
    nodata: float | int | None = None,
) -> pathlib.Path:
    '''
    Write a virtual warped raster (VRT) to disk.

    Args:
        src:
            Open source dataset reader.
        output_path:
            Target destination path for the warped .vrt file.
        crs:
            Target coordinate reference system.
        transform:
            Target affine transform.
        width:
            Target width in pixels.
        height:
            Target height in pixels.
        resampling:
            Resampling algorithm enum.
        nodata:
            Optional nodata value.

    Returns:
        pathlib.Path:
            Resolved absolute path to the created warped VRT file.
    '''
    out_p = pathlib.Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.vrt.WarpedVRT(
        src,
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        resampling=resampling,
        nodata=nodata,
    ) as warped:
        rasterio.shutil.copy(warped, out_p, driver='VRT')
    return out_p
