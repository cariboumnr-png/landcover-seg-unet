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

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access

'''Unit tests for DataBlock related calculator methods.'''

# standard imports
import numpy
# thrid-party imports
import pytest
# local imports
import landseg.geopipe.core.foundation_data_block as data_block # direct import

# alises
rng = numpy.random.default_rng()

# ----- _Calc calculators
def test_mask_with_nodata():
    band = numpy.ones((256, 256), dtype=numpy.float32)
    nodata = -999.9

    idx = rng.choice(band.size, 999, replace=False)
    band.flat[idx] = nodata

    masked = data_block._Calc.mask(band, nodata)

    assert numpy.sum(masked.mask) == 999


def test_mask_without_nodata():
    band = numpy.ones((256, 256), dtype=numpy.float32)
    masked = data_block._Calc.mask(band, None)

    assert numpy.array_equal(band, masked.data) # no change


def test_mask_no_matching_nodata():
    band = numpy.ones((10, 10), dtype=numpy.float32)
    masked = data_block._Calc.mask(band, -999.0)

    assert not numpy.any(masked.mask)


def test_mask_all_nodata():
    band = numpy.full((10, 10), -999.0, dtype=numpy.float32)
    masked = data_block._Calc.mask(band, -999.0)

    assert numpy.all(masked.mask)


def test_mask_float_nodata_isclose():
    band = numpy.ones((256, 256), dtype=numpy.float64)
    nodata = -999.9

    idx = rng.choice(band.size, 999, replace=False)
    tol = 1e-8 + 1e-5 * abs(nodata) # default tolerance
    band.flat[idx] = nodata + rng.uniform(-tol * 0.5, tol * 0.5)

    masked = data_block._Calc.mask(band, nodata)

    assert numpy.sum(masked.mask) == 999


def test_mask_does_not_modify_input():
    band = numpy.ones((10, 10), dtype=numpy.float32)
    original = band.copy()

    data_block._Calc.mask(band, 1.0)

    numpy.testing.assert_array_equal(band, original)


def test_mask_preserves_shape():
    band = numpy.ones((256, 256), dtype=numpy.float32)
    masked = data_block._Calc.mask(band, None)

    assert band.shape == masked.data.shape


def test_mask_converts_dtype():
    band = numpy.ones((10, 10), dtype=numpy.float32)
    masked = data_block._Calc.mask(band, None)

    assert masked.dtype == numpy.float64


def test_entropy_single_class():
    counts = numpy.array([42])
    ent = data_block._Calc.entropy(counts)

    assert ent == pytest.approx(0.0)


def test_entropy_uniform_class_distribution():
    counts = numpy.array([1, 1, 1, 1]) # prob: [0.25, 0.25, 0.25, 0.25]
    ent = data_block._Calc.entropy(counts)

    assert ent == pytest.approx(2.0)


def test_entropy_non_uniform_class_distribution():
    counts = [1, 1, 2] # prob: [0.25, 0.25, 0.50]
    result = data_block._Calc.entropy(counts)

    assert result == pytest.approx(1.5)


def test_entropy_ignores_zero_counts():
    counts = [1, 0, 1] # prob: [0.50, 0.00, 0.50]
    result = data_block._Calc.entropy(counts)

    assert result == pytest.approx(1.0)


@pytest.mark.parametrize('method_name', ['ndvi', 'ndmi', 'nbr'])
def test_spectral_indices_calc(method_name):
    nir = numpy.ma.array([0.8, 0.6, 0.2])
    other = numpy.ma.array([0.2, 0.3, 0.1])

    method = getattr(data_block._Calc, method_name)
    result = method(nir, other, nodata=-999.9)
    expected = (nir - other) / (nir + other)

    numpy.testing.assert_allclose(result, expected.filled(-999.9))


def test_spectral_indices_calc_fill_nodata():
    nir = numpy.ma.array([0.8, 0.6], mask=[False, True])
    red = numpy.ma.array([0.2, 0.3], mask=[False, True])

    method = data_block._Calc.ndvi # test once
    result = method(nir, red, nodata=-999.9)
    expected = numpy.array([0.6, -999.9]) # (0.8-0.2) / (0.8+0.2) = 0.6

    numpy.testing.assert_allclose(result, expected)


def test_get_px_group_returns_centered_window():
    arr = numpy.arange(25).reshape(5, 5)

    result = data_block._Calc.get_px_group(arr, x=2, y=2, rr=1) # centre, r=1
    expected = numpy.array([
        [ 6,  7,  8],
        [11, 12, 13],
        [16, 17, 18],
    ])

    numpy.testing.assert_array_equal(result, expected)


def test_get_px_group_window_size():
    arr = numpy.arange(100).reshape(10, 10) # larger array
    result = data_block._Calc.get_px_group(arr, x=5, y=5, rr=2)

    assert result.shape == (5, 5)


def test_slope_n_aspect_flat_surface():
    arr = numpy.full((3, 3), 42.0)
    slope, cos, sin = data_block._Calc.slope_n_aspect(arr, -999.9)

    assert slope == pytest.approx(0.0)
    assert cos == pytest.approx(1.0)
    assert sin == pytest.approx(0.0)


def test_slope_n_aspect_known_gradient():
    arr = numpy.array([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ], dtype=float) # west-east slope

    slope, cos, sin = data_block._Calc.slope_n_aspect(arr, -999.9)

    assert slope == pytest.approx(1.0)
    assert cos == pytest.approx(-1.0)
    assert sin == pytest.approx(0.0)


@pytest.mark.parametrize('invalid', [-999.9, numpy.nan])
def test_slope_n_aspect_any_invald_value_returns_nodata(invalid):
    arr = numpy.full((3, 3), 42.0)
    idx = rng.choice(arr.size, 1, replace=False)
    arr.flat[idx] = invalid
    results = data_block._Calc.slope_n_aspect(arr, -999.9)

    assert results == (-999.9, -999.9, -999.9)


def test_tpi_center_above_neighbors():
    arr = numpy.array([
        [10, 10, 10],
        [10, 20, 10],
        [10, 10, 10],
    ], dtype=float)
    tpi = data_block._Calc.tpi(arr, -999.9)

    assert tpi == pytest.approx(10.0)


def test_tpi_center_below_neighbors():
    arr = numpy.array([
        [20, 20, 20],
        [20, 10, 20],
        [20, 20, 20],
    ], dtype=float)
    tpi = data_block._Calc.tpi(arr, -999.9)

    assert tpi == pytest.approx(-10.0)


def test_tpi_flat_surface():
    arr = numpy.full((3, 3), 42.0)
    tpi = data_block._Calc.tpi(arr, -999.9)

    assert tpi == pytest.approx(0.0)


def test_tpi_ignores_invalid_neighbours():
    arr = numpy.array([
        [20, 20, 20],
        [20, 10, -999.9],
        [numpy.nan, 20, 20],
    ], dtype=float)
    tpi = data_block._Calc.tpi(arr, -999.9)

    assert tpi == pytest.approx(-10.0) # with the rest six 20s


def test_tpi_all_invalid_neighbours_returns_nodata():
    arr = numpy.full((3, 3), numpy.nan)
    arr[1, 1] = 42
    tpi = data_block._Calc.tpi(arr, -999.9)

    assert tpi == -999.9


@pytest.mark.parametrize('invalid', [-999.9, numpy.nan])
def test_tpi_invalid_centre_returns_nodata(invalid):
    arr = numpy.full((3, 3), 42.0)
    arr[1, 1] = invalid
    tpi = data_block._Calc.tpi(arr, -999.9)

    assert tpi == -999.9
