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
Numeric utilities for geospatial feature engineering and analysis.

Provides shared masking, validity checking, normalized-difference
calculations, and topographic metric computations used throughout
the ingestion pipeline. Centralizes invalid-pixel semantics and
ensures consistent handling of NaN, infinity, nodata values, and
numerically unstable operations across spectral and terrain-derived
features.

Public APIs:
    - `entropy`: Compute Shannon entropy from class counts.
    - `get_px_group`: Extract neighbourhood pixel windows.
    - `mask`: Create masked-array views of raster bands.
    - `nbr`: Compute Normalized Burn Ratio.
    - `ndmi`: Compute Normalized Difference Moisture Index.
    - `ndvi`: Compute Normalized Difference Vegetation Index.
    - `slope_n_aspect`: Derive slope and aspect metrics from a DEM.
    - `tpi`: Derive Topographic Position Index from a DEM.
'''

# standard imports
import math
# third-party imports
import numpy


# ----- constants
ND_EPS = 1e-12


# ----- public functions
def mask(
    band: numpy.ndarray,
    nodata: float | None,
) -> numpy.ma.MaskedArray:
    '''
    Return a masked floating-point view of a raster band.

    Args:
        band:
            Input two-dimensional raster band array.
        nodata:
            Optional sentinel value representing invalid pixels.

    Returns:
        numpy.ma.MaskedArray:
            Floating-point masked array with invalid pixels masked.
    '''
    band_f64 = band.astype(numpy.float64)
    return numpy.ma.array(
        band_f64,
        mask=_invalid_mask(band_f64, nodata),
    )


def entropy(counts: numpy.ndarray | list[int]) -> float:
    '''
    Compute Shannon entropy from class pixel counts.

    Args:
        counts:
            Sequence or array of non-negative per-class counts.

    Returns:
        float:
            Base-2 Shannon entropy in bits, or `0.0` if total is zero.
    '''
    ss = sum(counts)
    if ss <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / ss
            ent -= p * math.log2(p)
    return ent


def ndvi(
    nir: numpy.ma.MaskedArray,
    red: numpy.ma.MaskedArray,
    nodata: float | None,
) -> numpy.ndarray:
    '''
    Compute Normalized Difference Vegetation Index (NDVI).

    Args:
        nir:
            Masked near-infrared band array.
        red:
            Masked red band array.
        nodata:
            Optional fill value for invalid output pixels.

    Returns:
        numpy.ndarray:
            Clipped NDVI array in `[-1.0, 1.0]` with invalid fills.
    '''
    return _normalized_difference(nir, red, nodata)


def ndmi(
    nir: numpy.ma.MaskedArray,
    swir1: numpy.ma.MaskedArray,
    nodata: float | None,
) -> numpy.ndarray:
    '''
    Compute Normalized Difference Moisture Index (NDMI).

    Args:
        nir:
            Masked near-infrared band array.
        swir1:
            Masked short-wave infrared 1 band array.
        nodata:
            Optional fill value for invalid output pixels.

    Returns:
        numpy.ndarray:
            Clipped NDMI array in `[-1.0, 1.0]` with invalid fills.
    '''
    return _normalized_difference(nir, swir1, nodata)


def nbr(
    nir: numpy.ma.MaskedArray,
    swir2: numpy.ma.MaskedArray,
    nodata: float | None,
) -> numpy.ndarray:
    '''
    Compute Normalized Burn Ratio (NBR).

    Args:
        nir:
            Masked near-infrared band array.
        swir2:
            Masked short-wave infrared 2 band array.
        nodata:
            Optional fill value for invalid output pixels.

    Returns:
        numpy.ndarray:
            Clipped NBR array in `[-1.0, 1.0]` with invalid fills.
    '''
    return _normalized_difference(nir, swir2, nodata)


def get_px_group(
    arr: numpy.ndarray,
    x: int,
    y: int,
    rr: int,
) -> numpy.ndarray:
    '''
    Extract a square pixel neighborhood centered at `(x, y)`.

    Args:
        arr:
            Two-dimensional input raster array.
        x:
            Column index of the window center.
        y:
            Row index of the window center.
        rr:
            Half-window radius in pixels.

    Returns:
        numpy.ndarray:
            Sliced array view of shape `(2 * rr + 1, 2 * rr + 1)`.
    '''
    return arr[slice(y - rr, y + rr + 1), slice(x - rr, x + rr + 1)]


def slope_n_aspect(
    arr: numpy.ndarray,
    nodata: float | None,
) -> tuple[float, float, float]:
    '''
    Compute slope and aspect trigonometry from a 3x3 DEM window.

    Args:
        arr:
            Three-by-three elevation array centered at target pixel.
        nodata:
            Optional sentinel value indicating invalid pixels.

    Returns:
        tuple[float, float, float]:
            Tuple of `(slope, cos_aspect, sin_aspect)`, or repeated
            invalid fill values if any input pixel is invalid.
    '''
    if _invalid_mask(arr, nodata).any():
        invalid = numpy.nan if nodata is None else nodata
        return invalid, invalid, invalid

    dz_dx = (
        (arr[0, 2] + 2 * arr[1, 2] + arr[2, 2]) -
        (arr[0, 0] + 2 * arr[1, 0] + arr[2, 0])
    ) / 8.0

    dz_dy = (
        (arr[2, 0] + 2 * arr[2, 1] + arr[2, 2]) -
        (arr[0, 0] + 2 * arr[0, 1] + arr[0, 2])
    ) / 8.0

    slope = numpy.sqrt(dz_dx ** 2 + dz_dy ** 2)

    if slope == 0.0:
        return 0.0, 1.0, 0.0

    aspect_rad = numpy.arctan2(dz_dy, -dz_dx)

    if aspect_rad < 0:
        aspect_rad += 2 * numpy.pi

    return (
        float(slope),
        float(numpy.cos(aspect_rad)),
        float(numpy.sin(aspect_rad)),
    )


def tpi(
    arr: numpy.ndarray,
    nodata: float | None,
) -> float | None:
    '''
    Compute Topographic Position Index (TPI) for a DEM window.

    Args:
        arr:
            Two-dimensional elevation window around the center pixel.
        nodata:
            Optional sentinel value indicating invalid pixels.

    Returns:
        float | None:
            Center elevation minus neighborhood mean, or `nodata` when
            center or all neighbors are invalid.
    '''
    h, w = arr.shape
    c_row, c_col = h // 2, w // 2

    centre = arr[c_row, c_col]

    if _invalid_mask(numpy.asarray([centre]), nodata)[0]:
        return nodata

    invalid_mask = _invalid_mask(arr, nodata)
    masked = numpy.ma.array(arr, mask=invalid_mask)

    if masked.count() <= 1:
        return nodata

    neighbour_mean = (masked.sum() - centre) / (masked.count() - 1)

    return float(centre - neighbour_mean)


# ----- private helpers
def _invalid_mask(
    arr: numpy.ndarray,
    nodata: float | None,
) -> numpy.ndarray:
    '''
    Return True where values should be considered invalid.

    Invalid values include NaN, +/- Inf, and explicit nodata values.
    '''
    _mask = ~numpy.isfinite(arr)
    if nodata is not None:
        if not (isinstance(nodata, float) and numpy.isnan(nodata)):
            _mask |= arr == nodata
    return _mask


def _normalized_difference(
    a: numpy.ma.MaskedArray,
    b: numpy.ma.MaskedArray,
    nodata: float | None,
) -> numpy.ndarray:
    '''
    Compute a normalized difference index `(a - b) / (a + b)`.

    Pixels are marked invalid when either input is masked, the
    denominator is non-finite, or its magnitude is below `ND_EPS`.
    '''
    numer = a - b
    denom = a + b

    valid = (
        ~numpy.ma.getmaskarray(a)
        & ~numpy.ma.getmaskarray(b)
        & numpy.isfinite(denom)
        & (numpy.abs(denom) > ND_EPS)
    )

    fill_value = numpy.nan if nodata is None else float(nodata)

    out = numpy.full(denom.shape, fill_value, dtype=numpy.float64)

    numpy.divide(
        numer,
        denom,
        out=out,
        where=valid,
    )

    # theoretical range for NDVI/NDMI/NBR.
    out[valid] = numpy.clip(out[valid], -1.0, 1.0)
    return out
