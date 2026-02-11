'''
Validate and summarize raster geometry for image/label datasets.

This module ingests one or two rasters and validates core geometric
properties required for co-registration workflows. It checks that the
coordinate reference systems (CRS) match, verifies pixel sizes, and
computes the overlapping bounding box (intersection). A typed summary
(`GeometrySummary`) is returned for downstream processing.
'''

# standard imports
import typing
# third-party imports
import rasterio
import rasterio.coords
# local imports
import alias
import utils

# ---------------------------------Public Type---------------------------------
class GeometrySummary(typing.TypedDict):
    '''Typed dictionary to summarize the validated raster geometry.'''
    crs: str
    pixel_size: tuple[float, float]
    image_bbox: tuple[float, ...]
    label_bbox: tuple[float, ...] | None
    same_bbox: bool | None
    inter_bbox: tuple[float, ...]
    image_transform: rasterio.Affine
    label_transform: rasterio.Affine | None

# -------------------------------Public Function-------------------------------
def validate_geometry(
    image_fpath: str,
    label_fpath: str | None,
    logger: utils.Logger
) -> GeometrySummary:
    '''
    Ingest raster inputs and validate their alignment.

    When both image and label raster are provided, alignment between CRS
    and pixel size are validated before intersected extent computation
    and the return of a `GeometryMeta` dictionary. Otherwise if only
    image is provided (e.g., as inference data), the return will simply
    be composed from the image's transform.

    Args:
        image_fpath: Path to the image raster.
        label_fpath: Optional label raster path. If provided must be
            co-registered with the image raster.
        logger: Handles the logging for this module.

    Returns:
        GeometryMeta: A typed dictionary containing validation results.

    Raises:
        ValueError: when one of the following is encounter
            - Image and label rasters have difference CRS.
            - Image and label rasters have difference pixel size.
            - Image and label rasters do not have overlapping extent.

    Note: in final summary pixel size y is converted to positives.
    '''

    # init a meta dict
    summary = {}

    # execute pipeline
    with utils.open_rasters(image_fpath, label_fpath) as (_img, _lbl):
        # make sure at least image is present
        if _img is None:
            raise ValueError('A valid image raster is required')
        # assign raster handlers
        img: alias.RasterReader = _img
        lbl: alias.RasterReader | None = _lbl
        # get transforms
        summary['image_transform'] = img.transform
        if lbl is not None:
            summary['label_transform'] = lbl.transform
        else:
            summary['label_transform'] = None
        # check if both rasters have the same projection system
        summary['crs'] = _check_raster_proj(img, lbl, logger)
        # check if both rasters have the same squared pixels
        summary['pixel_size'] = _check_raster_pixels(img, lbl, logger)
        # get the overlapping extent from the input rasters
        bbox = _compute_overlap_extent(img, lbl, logger)
        summary.update(**bbox)

    # return a summary
    return typing.cast(GeometrySummary, summary)

# ------------------------------private  function------------------------------
def _check_raster_proj(
    img: alias.RasterReader,
    lbl: alias.RasterReader | None,
    logger: utils.Logger
) -> str:
    '''
    Check if the input rasters have the same CRS.

    Raises:
        ValueError: If CRSs disagree when both image and label rasters
            are provided.
    '''

    # if both image and label provided
    if lbl is not None:
        logger.log('DEBUG', ' | Both image & label rasters provided')
        # get projection names, raster.crs might return differently
        try:
            crs_1 = img.crs.to_string().split('"')[1]
            crs_2 = lbl.crs.to_string().split('"')[1]
        except IndexError:
            crs_1 = img.crs
            crs_2 = lbl.crs

        # check if the projection systems are the same
        if crs_1 != crs_2:
            m = f' | Projections do not match: \n1: {crs_1} != 2: {crs_2}'
            logger.log('ERROR', m)
            raise ValueError('The rasters must have the same projection')
        logger.log('DEBUG', f' | Matching projections: {crs_1}')
        return crs_1
    # or only image provided
    try:
        crs_1 = img.crs.to_string().split('"')[1]
    except IndexError:
        crs_1 = img.crs
    logger.log('INFO', f'CRS from image raster: {crs_1}')
    return crs_1

def _check_raster_pixels(
    img: alias.RasterReader,
    lbl: alias.RasterReader | None,
    logger: utils.Logger
) -> tuple[float, float]:
    '''
    Check if the input rasters have the same squared pixels.

    Raises:
        ValueError: If the pixel sizes are different when both image and
            label rasters are provided.
    '''

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
        if (x1, y1) != (x2, y2):
            m = f' | Input rasters have different pixel sizes: '\
                f'Raster1: ({x1}, {-y1}), Raster2: ({x2}, {-y2})'
            logger.log('ERROR', m)
            raise ValueError('Input rasters must have the same pixel size')
    # or only image provided
    else:
        transform_1 = img.transform
        x1, y1 = transform_1[0], -transform_1[4]

    # assign value and log out
    logger.log('DEBUG', f' | Image raster pixel size: {x1} x {-y1}')
    return x1, y1

def _compute_overlap_extent(
    img: alias.RasterReader,
    lbl: alias.RasterReader | None,
    logger: utils.Logger
) -> dict[str, typing.Any]:
    '''
    Get the overlapping extent of the input rasters.

    The extent is defined by:
    * max of the left bounds.
    * max of the bottom bounds.
    * min of the right bounds.
    * min of the top bounds.

    Raises:
        ValueError: If no overlapping extent can be computed when both
            image and label rasters are provided..
    '''

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
            logger.log('ERROR', ' | Input rasters have no overlaps')
            raise ValueError('Input rasters must have overlapping extents')

        # get the overlapping extent if no error and retrun a summary
        bb = rasterio.coords.BoundingBox(lft, btm, rgt, top)
        return {
            'image_bbox': b1,
            'label_bbox': b2,
            'same_bbox': b1 == b2,
            'inter_bbox': bb
        }

    # or only image provided, retrun summary from image bounds
    return {
        'image_bbox': img.bounds,
        'label_bbox': None,
        'same_bbox': None,
        'inter_bbox': img.bounds
    }
