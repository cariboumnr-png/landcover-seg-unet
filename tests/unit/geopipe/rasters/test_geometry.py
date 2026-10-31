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

'''Unit tests for raster spatial geometry primitives (geometry.py).'''

# third-party imports
import pytest
import rasterio.transform
# local imports
import landseg.geopipe.rasters as rasters


# ----- `check_raster_crs` tests
def test_check_raster_crs_matching():
    '''
    Given: Two matching CRS specifications.
    When: Validating CRS alignment.
    Then: Return string representation of the CRS.
    '''
    res = rasters.check_raster_crs('EPSG:3161', 'EPSG:3161')
    assert res == 'EPSG:3161'

    res_single = rasters.check_raster_crs('EPSG:4326')
    assert res_single == 'EPSG:4326'


def test_check_raster_crs_mismatch_raises():
    '''
    Given: Disagreeing CRS specifications.
    When: Validating CRS alignment.
    Then: Raise ValueError.
    '''
    with pytest.raises(ValueError, match='CRS does not match'):
        rasters.check_raster_crs('EPSG:3161', 'EPSG:4326')


def test_check_raster_crs_none_primary_raises():
    '''
    Given: None as primary CRS.
    When: Validating CRS alignment.
    Then: Raise ValueError.
    '''
    with pytest.raises(ValueError, match='valid primary CRS'):
        rasters.check_raster_crs(None)


# ----- `check_raster_resolution` tests
def test_check_raster_resolution_matching():
    '''
    Given: Transforms with identical pixel dimensions.
    When: Checking pixel resolution alignment.
    Then: Return horizontal and vertical pixel sizes.
    '''
    t1 = rasterio.transform.Affine(10.0, 0.0, 100.0, 0.0, -10.0, 500.0)
    t2 = rasterio.transform.Affine(10.0, 0.0, 200.0, 0.0, -10.0, 600.0)

    dx, dy = rasters.check_raster_resolution(t1, t2)
    assert dx == 10.0
    assert dy == 10.0


def test_check_raster_resolution_single():
    '''
    Given: A single transform without secondary transform.
    When: Checking pixel resolution alignment.
    Then: Return horizontal and vertical pixel sizes.
    '''
    t1 = rasterio.transform.Affine(10.0, 0.0, 100.0, 0.0, -10.0, 500.0)

    dx, dy = rasters.check_raster_resolution(t1)
    assert dx == 10.0
    assert dy == 10.0


def test_check_raster_resolution_mismatch_raises():
    '''
    Given: Transforms with differing pixel sizes.
    When: Checking resolution alignment.
    Then: Raise ValueError.
    '''
    t1 = rasterio.transform.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    t2 = rasterio.transform.Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0)

    with pytest.raises(ValueError, match='different pixel sizes'):
        rasters.check_raster_resolution(t1, t2)


# ----- `compute_overlap_bounds` tests
def test_compute_overlap_bounds_overlapping():
    '''
    Given: Two overlapping spatial bounding boxes.
    When: Computing overlap bounds.
    Then: Return intersected BoundingBox.
    '''
    b1 = (0.0, 0.0, 100.0, 100.0)
    b2 = (50.0, 50.0, 150.0, 150.0)

    inter = rasters.compute_overlap_bounds(b1, b2)
    assert inter == rasters.BoundingBox(50.0, 50.0, 100.0, 100.0)


def test_compute_overlap_bounds_single():
    '''
    Given: A single bounding box without secondary extent.
    When: Computing overlap bounds.
    Then: Return original BoundingBox.
    '''
    b = (10.0, 20.0, 30.0, 40.0)
    inter = rasters.compute_overlap_bounds(b)
    assert inter == rasters.BoundingBox(10.0, 20.0, 30.0, 40.0)


def test_compute_overlap_bounds_disjoint_raises():
    '''
    Given: Two spatially disjoint bounding boxes.
    When: Computing overlap bounds.
    Then: Raise ValueError.
    '''
    b1 = (0.0, 0.0, 10.0, 10.0)
    b2 = (20.0, 20.0, 30.0, 30.0)

    with pytest.raises(ValueError, match='overlapping spatial bounds'):
        rasters.compute_overlap_bounds(b1, b2)


# ----- `reproject_bounds` tests
def test_reproject_bounds():
    '''
    Given: Geographic coordinates in EPSG:4326.
    When: Reprojecting bounds to Web Mercator (EPSG:3857).
    Then: Return projected BoundingBox.
    '''
    src_bounds = (-80.0, 40.0, -79.0, 41.0)
    reprojected = rasters.reproject_bounds(
        src_bounds, 'EPSG:4326', 'EPSG:3857'
    )

    assert isinstance(reprojected, rasters.BoundingBox)
    assert reprojected.left < reprojected.right
    assert reprojected.bottom < reprojected.top
    assert round(reprojected.left) == -8905559
