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
Unit tests for spatial AOI raster block intersection and split
resolution.
'''

# standard imports
import os
# third-party imports
import numpy
import rasterio
import rasterio.transform
# local imports
import landseg.geopipe.prepare.data_partition.operations.aoi as split_aoi


# ----- test helper
def _create_dummy_geotiff(
    fpath: str,
    *,
    crs: str = 'EPSG:3161',
    transform: rasterio.transform.Affine,
    shape: tuple[int, int] = (256, 256),
) -> str:
    '''Create a simple single-band GeoTIFF file.'''
    os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
    data = numpy.ones(shape, dtype=numpy.uint8)

    with rasterio.open(
        fpath,
        'w',
        driver='GTiff',
        height=shape[0],
        width=shape[1],
        count=1,
        dtype=numpy.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return fpath


# ----- `intersect_aoi_raster` tests
def test_intersect_aoi_raster_basic(tmp_path):
    '''
    Given: An AOI GeoTIFF covering the top-left 256x256 pixel quadrant.
    When: Intersecting against a 2x2 grid of candidate blocks.
    Then: Only the intersecting block (0, 0) is selected.
    '''
    canvas_transform = rasterio.transform.from_origin(
        500000.0, 600000.0, 20.0, 20.0
    )
    aoi_path = str(tmp_path / 'test_aoi.tif')
    _create_dummy_geotiff(
        aoi_path,
        crs='EPSG:3161',
        transform=canvas_transform,
        shape=(256, 256),
    )

    candidates = [(0, 0), (0, 256), (256, 0), (256, 256)]
    matched = split_aoi.intersect_aoi_raster(
        aoi_path,
        candidate_blocks=candidates,
        block_size=(256, 256),
        canvas_crs='EPSG:3161',
        canvas_transform=canvas_transform,
        min_overlap=0.5,
    )

    assert matched == [(0, 0)]


def test_intersect_aoi_raster_reprojection(tmp_path):
    '''
    Given: An AOI raster in EPSG:4326 covering Southern Ontario.
    When: Intersecting against a canvas grid in EPSG:3161.
    Then: Coordinate reprojection succeeds and intersects the block.
    '''
    # EPSG:4326 transform around -80 deg lon, 45 deg lat
    aoi_transform = rasterio.transform.from_origin(-80.5, 45.5, 0.01, 0.01)
    aoi_path = str(tmp_path / 'aoi_4326.tif')
    _create_dummy_geotiff(
        aoi_path,
        crs='EPSG:4326',
        transform=aoi_transform,
        shape=(100, 100),
    )

    # canvas transform matching the reprojected area in EPSG:3161
    canvas_transform = rasterio.transform.from_origin(
        1281086.0, 12113833.0, 20.0, 20.0
    )
    candidates = [(0, 0), (10000, 10000)]


    matched = split_aoi.intersect_aoi_raster(
        aoi_path,
        candidate_blocks=candidates,
        block_size=(256, 256),
        canvas_crs='EPSG:3161',
        canvas_transform=canvas_transform,
        min_overlap=0.1,
    )

    assert (0, 0) in matched
    assert (10000, 10000) not in matched


# ----- `resolve_aoi_partitions` tests
def test_resolve_aoi_partitions_priority_conflict(tmp_path, mocker):
    '''
    Given: Overlapping test and validation AOIs covering the same block.
    When: Resolving partitions.
    Then: Priority rule assigns the block to test and logs a warning.
    '''
    canvas_transform = rasterio.transform.from_origin(
        500000.0, 600000.0, 20.0, 20.0
    )
    test_aoi = str(tmp_path / 'test_aoi.tif')
    val_aoi = str(tmp_path / 'val_aoi.tif')
    _create_dummy_geotiff(
        test_aoi, transform=canvas_transform, shape=(256, 256)
    )
    _create_dummy_geotiff(
        val_aoi, transform=canvas_transform, shape=(256, 256)
    )

    mock_logger = mocker.MagicMock()
    candidates = [(0, 0), (256, 256)]

    result = split_aoi.resolve_aoi_partitions(
        candidates,
        test_aoi=test_aoi,
        val_aoi=val_aoi,
        train_aoi=None,
        block_size=(256, 256),
        canvas_crs='EPSG:3161',
        canvas_transform=canvas_transform,
        logger=mock_logger,
    )

    assert result.test == [(0, 0)]
    assert not result.val
    assert not result.train
    assert result.unassigned == [(256, 256)]
    assert mock_logger.log.called


def test_resolve_aoi_partitions_scenario_1_three_splits(tmp_path):
    '''
    Given: Three distinct non-overlapping AOI rasters for test, val,
        train.
    When: Resolving partitions.
    Then: Each block is cleanly assigned to its corresponding split.
    '''
    t_test = rasterio.transform.from_origin(500000.0, 600000.0, 20.0, 20.0)
    t_val = rasterio.transform.from_origin(
        500000.0 + 256 * 20, 600000.0, 20.0, 20.0
    )
    t_train = rasterio.transform.from_origin(
        500000.0, 600000.0 - 256 * 20, 20.0, 20.0
    )

    test_path = str(tmp_path / 'test.tif')
    val_path = str(tmp_path / 'val.tif')
    train_path = str(tmp_path / 'train.tif')

    _create_dummy_geotiff(test_path, transform=t_test)
    _create_dummy_geotiff(val_path, transform=t_val)
    _create_dummy_geotiff(train_path, transform=t_train)

    candidates = [(0, 0), (0, 256), (256, 0), (256, 256)]

    result = split_aoi.resolve_aoi_partitions(
        candidates,
        test_aoi=test_path,
        val_aoi=val_path,
        train_aoi=train_path,
        block_size=(256, 256),
        canvas_crs='EPSG:3161',
        canvas_transform=t_test,
    )

    assert result.test == [(0, 0)]
    assert result.val == [(0, 256)]
    assert result.train == [(256, 0)]
    assert result.unassigned == [(256, 256)]
