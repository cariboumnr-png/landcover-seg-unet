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
Validate and summarize raster geometry for image/label datasets.

This module ingests one or two rasters and validates core geometric
properties required for co-registration workflows. It checks that the
coordinate reference systems (CRS) match, verifies pixel sizes, and
computes the overlapping bounding box (intersection).

Public APIs:
    - GeometrySummary: TypedDict summarizing raster geometry metadata.
    - validate_geometry: Ingests rasters and validates alignment.
'''

# standard imports
import dataclasses
import math
import pathlib
import typing
# local imports
import landseg.geopipe.rasters as rasters


# ----- public types
class GeometrySummary(typing.TypedDict):
    '''Typed dictionary to summarize the validated raster geometry.'''
    crs: str
    pixel_size: tuple[float, float]
    image_bbox: tuple[float, ...]
    label_bbox: tuple[float, ...] | None
    same_bbox: bool | None
    inter_bbox: tuple[float, ...]
    image_transform: rasters.Affine
    label_transform: rasters.Affine | None


# ----- private dataclasses
@dataclasses.dataclass
class _OverlapBoundingBox:
    '''Container for overlapping bounding boxes.'''
    image_bbox: rasters.BoundingBox
    label_bbox: rasters.BoundingBox | None
    same_bbox: bool | None
    inter_bbox: rasters.BoundingBox


# ----- public functions
def validate_geometry(
    image_fpath: pathlib.Path,
    label_fpath: pathlib.Path | None,
) -> GeometrySummary:
    '''
    Ingest raster inputs and validate their alignment.

    When both image and label rasters are provided, alignment between
    CRS and pixel size is validated before intersected extent
    computation. When only an image is provided, the summary is
    composed from the image transform and bounds.

    Args:
        image_fpath:
            File path to source image raster.
        label_fpath:
            Optional file path to source label raster. If provided,
            must be co-registered with the image raster.

    Returns:
        GeometrySummary:
            Typed dictionary containing validated geometry metadata.

    Raises:
        ValueError:
            If image is missing, CRSs disagree, pixel sizes differ,
            or rasters do not share overlapping spatial extent.
    '''
    with rasters.read_rasters(image_fpath, label_fpath) as (img, lbl):
        # make sure at least image is present
        if img is None:
            raise ValueError('A valid image raster is required')

        # get the overlapping extent from the input rasters
        bbox = _compute_overlap_extent(img, lbl)

        summary: GeometrySummary = {
            'crs': _check_raster_proj(img, lbl),
            'pixel_size': _check_raster_pixels(img, lbl),
            'image_bbox': bbox.image_bbox,
            'label_bbox': bbox.label_bbox,
            'same_bbox': bbox.same_bbox,
            'inter_bbox': bbox.inter_bbox,
            'image_transform': img.transform,
            'label_transform': lbl.transform if lbl else None,
        }
    return summary


# ----- private helpers
def _check_raster_proj(
    img: rasters.DatasetReader,
    lbl: rasters.DatasetReader | None,
) -> str:
    '''Check if the input rasters have matching coordinate systems.'''
    if img.crs is None:
        raise ValueError('A valid image raster with CRS is required')
    if lbl is not None:
        if lbl.crs is None:
            raise ValueError('A valid label raster with CRS is required')
        if not rasters.is_same_crs(img.crs, lbl.crs):
            raise ValueError(
                f'CRS does not match img!=lbl: {img.crs}!={lbl.crs}'
            )
    return str(img.crs)


def _check_raster_pixels(
    img: rasters.DatasetReader,
    lbl: rasters.DatasetReader | None,
) -> tuple[float, float]:
    '''Check if the input rasters have matching pixel sizes.'''
    # if both image and label provided
    if lbl is not None:
        # get the transform (Affine matrix) from the metadata
        transform_1 = img.transform
        transform_2 = lbl.transform

        # transform[0]: pixel size in the x direction (horizontal).
        # transform[4]: pixel size in the y direction (vertical).
        x1, y1 = transform_1[0], -transform_1[4]
        x2, y2 = transform_2[0], -transform_2[4]

        # check if the pixel sizes match
        if not _is_close((x1, y1), (x2, y2)):
            raise ValueError(
                f'DETAILS/ Input rasters have different pixel sizes: '
                f'Raster1: ({x1}, {-y1}), Raster2: ({x2}, {-y2})'
            )
    # or only image provided
    else:
        transform_1 = img.transform
        x1, y1 = transform_1[0], -transform_1[4]

    # assign value and log out
    return x1, y1


def _compute_overlap_extent(
    img: rasters.DatasetReader,
    lbl: rasters.DatasetReader | None,
) -> _OverlapBoundingBox:
    '''Compute overlapping spatial extent between input rasters.'''
    # if both image and label provided
    if lbl is not None:
        # get the bounding boxes
        b1 = img.bounds
        b2 = lbl.bounds

        # bounds(0-3) correspond to [left, bottom, right, top]
        lft = max(b1[0], b2[0]) # max of the left bounds
        btm = max(b1[1], b2[1]) # max of the bottom bounds
        rgt = min(b1[2], b2[2]) # min of the right bounds
        top = min(b1[3], b2[3]) # min of the top bounds

        # if the two do not overlop
        if lft >= rgt or btm >= top:
            raise ValueError('Input rasters must have overlapping extents')

        # get the overlapping extent if no error and retrun a summary
        bb = rasters.BoundingBox(lft, btm, rgt, top)
        return _OverlapBoundingBox(
            image_bbox=b1,
            label_bbox=b2,
            same_bbox=b1 == b2,
            inter_bbox=bb,
        )

    # or only image provided, retrun summary from image bounds
    return _OverlapBoundingBox(
        image_bbox=img.bounds,
        label_bbox=None,
        same_bbox=None,
        inter_bbox=img.bounds
    )


def _is_close(p1: tuple[float, float], p2: tuple[float, float]) -> bool:
    '''Check if two coordinate pairs are close within tolerance.'''
    px1, py1 = p1
    px2, py2 = p2
    return (
        math.isclose(px1, px2, rel_tol=1e-9, abs_tol=1e-9) and
        math.isclose(py1, py2, rel_tol=1e-9, abs_tol=1e-9)
    )
