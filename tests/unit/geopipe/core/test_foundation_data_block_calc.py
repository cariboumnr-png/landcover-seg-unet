# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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

# pylint: disable=protected-access

'''Unit tests for DataBlock related calculator methods.'''

# standard imports
import numpy
# third-party imports
import pytest
# local imports
import landseg.geopipe.core.foundation_data_block as data_block

# aliases
rng = numpy.random.default_rng()


# ----- `_Calc` calculators
def test_mask_with_nodata():
    '''
    Given: A float32 band containing a specific nodata value.
    When: Running mask calculation.
    Then: Correctly flags matching elements in the mask.
    '''
    band = numpy.ones((256, 256), dtype=numpy.float32)
    nodata = -999.9

    idx = rng.choice(band.size, 999, replace=False)
    band.flat[idx] = nodata

    masked = data_block._Calc.mask(band, nodata)

    assert numpy.sum(masked.mask) == 999


def test_mask_without_nodata():
    '''
    Given: A band without any nodata filter value.
    When: Running mask calculation.
    Then: Return a masked array with no elements masked.
    '''
    band = numpy.ones((256, 256), dtype=numpy.float32)
    masked = data_block._Calc.mask(band, None)

    assert numpy.array_equal(band, masked.data)


def test_mask_no_matching_nodata():
    '''
    Given: A band where no elements match the nodata filter value.
    When: Running mask calculation.
    Then: Return an array with zero masked elements.
    '''
    band = numpy.ones((10, 10), dtype=numpy.float32)
    masked = data_block._Calc.mask(band, -999.0)

    assert not numpy.any(masked.mask)


def test_mask_all_nodata():
    '''
    Given: A band where all elements match the nodata value.
    When: Running mask calculation.
    Then: Return an array where all elements are masked.
    '''
    band = numpy.full((10, 10), -999.0, dtype=numpy.float32)
    masked = data_block._Calc.mask(band, -999.0)

    assert numpy.all(masked.mask)


def test_mask_float_nodata_isclose():
    '''
    Given: A double precision band containing values within tolerance
        of the target nodata value.
    When: Running mask calculation.
    Then: Mask all elements falling within the close-to-nodata range.
    '''
    band = numpy.ones((256, 256), dtype=numpy.float64)
    nodata = -999.9

    idx = rng.choice(band.size, 999, replace=False)
    tol = 1e-8 + 1e-5 * abs(nodata)
    band.flat[idx] = nodata + rng.uniform(-tol * 0.5, tol * 0.5)

    masked = data_block._Calc.mask(band, nodata)

    assert numpy.sum(masked.mask) == 999


def test_mask_does_not_modify_input():
    '''
    Given: A band array.
    When: Running mask calculation.
    Then: The original input array is not modified.
    '''
    band = numpy.ones((10, 10), dtype=numpy.float32)
    original = band.copy()

    data_block._Calc.mask(band, 1.0)

    numpy.testing.assert_array_equal(band, original)


def test_mask_preserves_shape():
    '''
    Given: A band array of shape (256, 256).
    When: Running mask calculation.
    Then: Preserve the exact shape in the output array.
    '''
    band = numpy.ones((256, 256), dtype=numpy.float32)
    masked = data_block._Calc.mask(band, None)

    assert band.shape == masked.data.shape


def test_mask_converts_dtype():
    '''
    Given: A float32 band array.
    When: Running mask calculation.
    Then: Convert the underlying output data dtype to float64.
    '''
    band = numpy.ones((10, 10), dtype=numpy.float32)
    masked = data_block._Calc.mask(band, None)

    assert masked.dtype == numpy.float64


def test_entropy_single_class():
    '''
    Given: An array of class counts with a single active category.
    When: Calculating Shannon entropy.
    Then: Return 0.0 entropy.
    '''
    counts = numpy.array([42])
    ent = data_block._Calc.entropy(counts)

    assert ent == pytest.approx(0.0)


def test_entropy_uniform_class_distribution():
    '''
    Given: Uniform class counts distribution.
    When: Calculating Shannon entropy.
    Then: Return the maximum possible entropy (e.g. 2.0).
    '''
    counts = numpy.array([1, 1, 1, 1])
    ent = data_block._Calc.entropy(counts)

    assert ent == pytest.approx(2.0)


def test_entropy_non_uniform_class_distribution():
    '''
    Given: Non-uniform class counts distribution.
    When: Calculating Shannon entropy.
    Then: Return correct expected entropy (e.g. 1.5).
    '''
    counts = [1, 1, 2]
    result = data_block._Calc.entropy(counts)

    assert result == pytest.approx(1.5)


def test_entropy_ignores_zero_counts():
    '''
    Given: Class counts array containing zero-count entries.
    When: Calculating Shannon entropy.
    Then: Ignore the zero entries and calculate entropy.
    '''
    counts = [1, 0, 1]
    result = data_block._Calc.entropy(counts)

    assert result == pytest.approx(1.0)


@pytest.mark.parametrize('method_name', ['ndvi', 'ndmi', 'nbr'])
def test_spectral_indices_calc(method_name):
    '''
    Given: Masked bands for NIR, red, and SWIR bands.
    When: Calculating a specific index (ndvi, ndmi, or nbr).
    Then: Compute the normalized difference ratio.
    '''
    nir = numpy.ma.array([0.8, 0.6, 0.2])
    other = numpy.ma.array([0.2, 0.3, 0.1])

    method = getattr(data_block._Calc, method_name)
    result = method(nir, other, nodata=-999.9)
    expected = (nir - other) / (nir + other)

    numpy.testing.assert_allclose(result, expected.filled(-999.9))


def test_spectral_indices_calc_fill_nodata():
    '''
    Given: Bands containing masked invalid elements.
    When: Calculating spectral indices.
    Then: Fill the masked indices with the designated nodata value.
    '''
    nir = numpy.ma.array([0.8, 0.6], mask=[False, True])
    red = numpy.ma.array([0.2, 0.3], mask=[False, True])

    method = data_block._Calc.ndvi
    result = method(nir, red, nodata=-999.9)
    expected = numpy.array([0.6, -999.9])

    numpy.testing.assert_allclose(result, expected)


def test_get_px_group_returns_centered_window():
    '''
    Given: A 2D array and target coordinates.
    When: Calling get_px_group with radius 1.
    Then: Extract the 3x3 window centered around target coordinates.
    '''
    arr = numpy.arange(25).reshape(5, 5)

    result = data_block._Calc.get_px_group(arr, x=2, y=2, rr=1)
    expected = numpy.array([
        [ 6,  7,  8],
        [11, 12, 13],
        [16, 17, 18],
    ])

    numpy.testing.assert_array_equal(result, expected)


def test_get_px_group_window_size():
    '''
    Given: A 2D array and target coordinates.
    When: Calling get_px_group with radius 2.
    Then: Return a window of shape (5, 5).
    '''
    arr = numpy.arange(100).reshape(10, 10)
    result = data_block._Calc.get_px_group(arr, x=5, y=5, rr=2)

    assert result.shape == (5, 5)


def test_slope_n_aspect_flat_surface():
    '''
    Given: A 3x3 elevation grid representing a flat surface.
    When: Calculating slope and aspect.
    Then: Return slope 0.0, cos 1.0, and sin 0.0.
    '''
    arr = numpy.full((3, 3), 42.0)
    slope, cos, sin = data_block._Calc.slope_n_aspect(arr, -999.9)

    assert slope == pytest.approx(0.0)
    assert cos == pytest.approx(1.0)
    assert sin == pytest.approx(0.0)


def test_slope_n_aspect_known_gradient():
    '''
    Given: A 3x3 elevation grid representing a west-east slope.
    When: Calculating slope and aspect.
    Then: Return correct gradients and orientation values.
    '''
    arr = numpy.array([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ], dtype=float)

    slope, cos, sin = data_block._Calc.slope_n_aspect(arr, -999.9)

    assert slope == pytest.approx(1.0)
    assert cos == pytest.approx(-1.0)
    assert sin == pytest.approx(0.0)


@pytest.mark.parametrize('invalid', [-999.9, numpy.nan])
def test_slope_n_aspect_any_invald_value_returns_nodata(invalid):
    '''
    Given: A 3x3 elevation grid containing invalid values.
    When: Calculating slope and aspect.
    Then: Return nodata markers for all three output channels.
    '''
    arr = numpy.full((3, 3), 42.0)
    idx = rng.choice(arr.size, 1, replace=False)
    arr.flat[idx] = invalid
    results = data_block._Calc.slope_n_aspect(arr, -999.9)

    assert results == (-999.9, -999.9, -999.9)


def test_tpi_center_above_neighbors():
    '''
    Given: A 3x3 grid where center cell is higher than neighbors.
    When: Calculating topographic position index.
    Then: Return a positive TPI value.
    '''
    arr = numpy.array([
        [10, 10, 10],
        [10, 20, 10],
        [10, 10, 10],
    ], dtype=float)
    tpi = data_block._Calc.tpi(arr, -999.9)

    assert tpi == pytest.approx(10.0)


def test_tpi_center_below_neighbors():
    '''
    Given: A 3x3 grid where center cell is lower than neighbors.
    When: Calculating topographic position index.
    Then: Return a negative TPI value.
    '''
    arr = numpy.array([
        [20, 20, 20],
        [20, 10, 20],
        [20, 20, 20],
    ], dtype=float)
    tpi = data_block._Calc.tpi(arr, -999.9)

    assert tpi == pytest.approx(-10.0)


def test_tpi_flat_surface():
    '''
    Given: A flat 3x3 elevation grid.
    When: Calculating topographic position index.
    Then: Return TPI of 0.0.
    '''
    arr = numpy.full((3, 3), 42.0)
    tpi = data_block._Calc.tpi(arr, -999.9)

    assert tpi == pytest.approx(0.0)


def test_tpi_ignores_invalid_neighbours():
    '''
    Given: A 3x3 grid with invalid neighbor cells.
    When: Calculating topographic position index.
    Then: Ignore the invalid neighbors and average over remaining cells.
    '''
    arr = numpy.array([
        [20, 20, 20],
        [20, 10, -999.9],
        [numpy.nan, 20, 20],
    ], dtype=float)
    tpi = data_block._Calc.tpi(arr, -999.9)

    assert tpi == pytest.approx(-10.0)


def test_tpi_all_invalid_neighbours_returns_nodata():
    '''
    Given: A 3x3 grid with all invalid neighbor cells.
    When: Calculating topographic position index.
    Then: Return the default nodata marker.
    '''
    arr = numpy.full((3, 3), numpy.nan)
    arr[1, 1] = 42
    tpi = data_block._Calc.tpi(arr, -999.9)

    assert tpi == -999.9


@pytest.mark.parametrize('invalid', [-999.9, numpy.nan])
def test_tpi_invalid_centre_returns_nodata(invalid):
    '''
    Given: A 3x3 grid with an invalid center cell.
    When: Calculating topographic position index.
    Then: Return the default nodata marker.
    '''
    arr = numpy.full((3, 3), 42.0)
    arr[1, 1] = invalid
    tpi = data_block._Calc.tpi(arr, -999.9)

    assert tpi == -999.9
