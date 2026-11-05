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

# pylint: disable=protected-access

'''
Unit tests for calc.py.
'''

# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.ingest.blocks.assembler.calc as calc


def test_invalid_mask_detects_nan_inf_and_nodata():
    '''
    Given: An array containing NaN, Inf, and nodata values.
    When: Building an invalid mask.
    Then: Mark all invalid values as True.
    '''
    arr = numpy.array([1.0, numpy.nan, numpy.inf, -numpy.inf, -9999.0])
    mask = calc._invalid_mask(arr, -9999.0)

    assert numpy.array_equal(mask, numpy.array([False, True, True, True, True]))


def test_invalid_mask_with_none_nodata():
    '''
    Given: An array containing only finite values.
    When: Building an invalid mask with nodata unset.
    Then: Only non-finite values are masked.
    '''
    arr = numpy.array([ 1.0, 2.0, numpy.nan])
    mask = calc._invalid_mask(arr, None)

    assert numpy.array_equal(mask, numpy.array([False, False, True]))


def test_mask_masks_invalid_values():
    '''
    Given: A raster band containing invalid values.
    When: Creating a masked array.
    Then: Invalid pixels are masked.
    '''
    band = numpy.array([1.0, numpy.nan, -9999.0])
    masked = calc.mask(band, -9999.0)

    assert masked.mask.tolist() == [ False, True, True]


def test_normalized_difference_returns_expected_values():
    '''
    Given: Valid input bands.
    When: Computing a normalized difference.
    Then: Return the expected index values.
    '''
    a = calc.mask(numpy.array([10.0, 20.0]), None)
    b = calc.mask(numpy.array([5.0, 10.0]), None)
    result = calc._normalized_difference(a, b, numpy.nan)
    expected = numpy.array([1 / 3, 1 / 3])

    numpy.testing.assert_allclose(result, expected)


def test_normalized_difference_replaces_zero_denominator():
    '''
    Given: Inputs with a zero denominator.
    When: Computing a normalized difference.
    Then: Return nodata for the invalid pixel.
    '''
    a = calc.mask(numpy.array([1.0]), None)
    b = calc.mask(numpy.array([-1.0]), None)
    result = calc._normalized_difference(a, b, -9999.0)

    assert result[0] == -9999.0


def test_normalized_difference_replaces_near_zero_denominator():
    '''
    Given: Inputs with a near-zero denominator.
    When: Computing a normalized difference.
    Then: Return nodata for the invalid pixel.
    '''
    a = calc.mask(numpy.array([1.0]), None)
    b = calc.mask(numpy.array([-1.0 + 1e-16]), None)
    result = calc._normalized_difference(a, b, -9999.0)

    assert result[0] == -9999.0


def test_ndvi_matches_expected_formula():
    '''
    Given: Red and NIR bands.
    When: Computing NDVI.
    Then: Return the expected NDVI values.
    '''
    nir = calc.mask(numpy.array([0.8]), None)
    red = calc.mask(numpy.array([0.2]), None)
    result = calc.ndvi(nir, red, numpy.nan)

    numpy.testing.assert_allclose(result, numpy.array([0.6]))


def test_entropy_returns_zero_for_single_class():
    '''
    Given: A single populated class.
    When: Computing entropy.
    Then: Return zero entropy.
    '''
    ent = calc.entropy([100, 0, 0])

    assert ent == 0.0


def test_entropy_returns_expected_value():
    '''
    Given: Two equally represented classes.
    When: Computing entropy.
    Then: Return one bit of entropy.
    '''
    ent = calc.entropy([10, 10])

    assert ent == pytest.approx(1.0)


def test_entropy_returns_zero_for_empty_or_zero_counts():
    '''
    Given: Empty, zero-sum, or negative-sum counts.
    When: Computing entropy.
    Then: Return zero entropy without zero division errors.
    '''
    assert calc.entropy([]) == 0.0
    assert calc.entropy([0, 0, 0]) == 0.0
    assert calc.entropy([-5, 5]) == 0.0
    assert calc.entropy([-10, 0]) == 0.0


def test_slope_n_aspect_returns_nodata_for_invalid_window():
    '''
    Given: A DEM window containing invalid values.
    When: Computing slope and aspect.
    Then: Return nodata outputs.
    '''
    arr = numpy.ones((3, 3))
    arr[1, 1] = numpy.nan
    slope, cos_a, sin_a = calc.slope_n_aspect(arr, -9999.0)

    assert slope == -9999.0
    assert cos_a == -9999.0
    assert sin_a == -9999.0


def test_slope_n_aspect_returns_flat_surface():
    '''
    Given: A flat DEM window.
    When: Computing slope and aspect.
    Then: Return a zero slope.
    '''
    arr = numpy.ones((3, 3))
    slope, cos_a, sin_a = calc.slope_n_aspect(arr, -9999.0)

    assert slope == 0.0
    assert cos_a == 1.0
    assert sin_a == 0.0


def test_tpi_returns_zero_for_flat_surface():
    '''
    Given: A flat DEM neighbourhood.
    When: Computing TPI.
    Then: Return zero.
    '''
    arr = numpy.full((5, 5), 10.0)
    result = calc.tpi(arr, -9999.0)

    assert result == pytest.approx(0.0)


def test_tpi_returns_positive_for_local_peak():
    '''
    Given: A neighbourhood with an elevated centre pixel.
    When: Computing TPI.
    Then: Return a positive value.
    '''
    arr = numpy.full((5, 5), 10.0)
    arr[2, 2] = 20.0
    result = calc.tpi(arr, -9999.0)

    assert result and result > 0.0
