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
Spatial geometry primitives and bounding box operations.

This module provides reusable primitives for validating coordinate
reference system alignment, checking pixel resolution consistency,
computing spatial bounding box intersections, and reprojecting extents.

Public APIs:
    - `check_raster_crs`: Verify coordinate reference system alignment.
    - `check_raster_resolution`: Verify pixel size consistency.
    - `compute_overlap_bounds`: Calculate intersected bounding box.
    - `reproject_bounds`: Reproject bounding box to destination CRS.
'''

# standard imports
import math
# third-party imports
import rasterio.warp
# local imports
import landseg.geopipe.rasters._types as t
import landseg.geopipe.rasters.warp as warp


# ----- public functions
def check_raster_crs(
    crs_a: t.CRS | str | None,
    crs_b: t.CRS | str | None = None,
) -> str:
    '''
    Verify coordinate reference system presence and alignment.

    Args:
        crs_a:
            Primary coordinate reference system.
        crs_b:
            Optional secondary coordinate reference system to compare.

    Returns:
        str:
            String representation of the validated primary CRS.

    Raises:
        ValueError:
            If primary CRS is missing or secondary CRS differs.
    '''
    if crs_a is None:
        raise ValueError('A valid primary CRS is required')
    if crs_b is not None:
        if not warp.is_same_crs(crs_a, crs_b):
            raise ValueError(
                f'CRS does not match primary!=secondary: {crs_a}!={crs_b}'
            )
    return str(crs_a)


def check_raster_resolution(
    transform_a: t.Affine,
    transform_b: t.Affine | None = None,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> tuple[float, float]:
    '''
    Verify pixel resolution consistency between affine transforms.

    Args:
        transform_a:
            Primary affine transformation matrix.
        transform_b:
            Optional secondary affine transformation matrix to compare.
        rel_tol:
            Relative tolerance for floating-point comparison.
        abs_tol:
            Absolute tolerance for floating-point comparison.

    Returns:
        tuple[float, float]:
            Pixel resolution in horizontal and vertical directions.

    Raises:
        ValueError:
            If pixel dimensions between transforms disagree.
    '''
    dx_a, dy_a = abs(transform_a.a), abs(transform_a.e)
    if transform_b is not None:
        dx_b, dy_b = abs(transform_b.a), abs(transform_b.e)
        if not (
            math.isclose(dx_a, dx_b, rel_tol=rel_tol, abs_tol=abs_tol)
            and math.isclose(dy_a, dy_b, rel_tol=rel_tol, abs_tol=abs_tol)
        ):
            raise ValueError(
                f'Input transforms have different pixel sizes: '
                f'({dx_a}, {dy_a}) != ({dx_b}, {dy_b})'
            )
    return dx_a, dy_a


def compute_overlap_bounds(
    bounds_a: t.BoundingBox | tuple[float, float, float, float],
    bounds_b: t.BoundingBox | tuple[float, float, float, float] | None = None,
) -> t.BoundingBox:
    '''
    Compute spatial intersection bounding box of two extents.

    Args:
        bounds_a:
            Primary bounding box coordinates (left, bottom, right, top).
        bounds_b:
            Optional secondary bounding box coordinates. If None,
            returns bounds_a.

    Returns:
        t.BoundingBox:
            Intersected bounding box (left, bottom, right, top).

    Raises:
        ValueError:
            If extents do not share overlapping spatial coverage.
    '''
    if bounds_b is None:
        return t.BoundingBox(*bounds_a)

    left = max(bounds_a[0], bounds_b[0])
    bottom = max(bounds_a[1], bounds_b[1])
    right = min(bounds_a[2], bounds_b[2])
    top = min(bounds_a[3], bounds_b[3])

    if left >= right or bottom >= top:
        raise ValueError('Input extents do not have overlapping spatial bounds')

    return t.BoundingBox(left, bottom, right, top)


def reproject_bounds(
    bounds: t.BoundingBox | tuple[float, float, float, float],
    src_crs: t.CRS | str,
    dst_crs: t.CRS | str,
    *,
    densify_pts: int = 21,
) -> t.BoundingBox:
    '''
    Reproject bounding box coordinates into target CRS.

    Args:
        bounds:
            Bounding box coordinates (left, bottom, right, top).
        src_crs:
            Source coordinate reference system.
        dst_crs:
            Destination coordinate reference system.
        densify_pts:
            Number of points used to densify each edge during transform.

    Returns:
        t.BoundingBox:
            Reprojected bounding box (left, bottom, right, top).
    '''
    left, bottom, right, top = bounds
    reprojected = rasterio.warp.transform_bounds(
        src_crs, dst_crs, left, bottom, right, top, densify_pts
    )
    return t.BoundingBox(*reprojected)
