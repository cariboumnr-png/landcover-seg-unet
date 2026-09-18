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

# pylint: disable=duplicate-code

'''
Unit tests for data partition pipeline under 3 spatial AOI scenarios.
'''


# standard imports
import os
# third-party imports
import numpy
import rasterio
import rasterio.transform
# local imports
import landseg.geopipe.prepare.data_partition.orchestration as orchestration


# ----- test helper
def _create_aoi_tiff(
    fpath: str,
    *,
    transform: rasterio.transform.Affine,
    shape: tuple[int, int] = (256, 256),
    crs: str = 'EPSG:3161',
) -> str:
    '''Create a synthetic AOI GeoTIFF.'''
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


# ----- scenario test cases
def test_scenario_1_fixed_test_aoi_with_auto_train_val(tmp_path):
    '''
    Scenario 1: User provides test_aoi.
    Given: A grid of 4 base blocks with block (0, 0) covered by
        test_aoi.
    When: Partitioning with val_ratio=0.5 and test_ratio=0.0.
    Then: Block (0, 0) is locked into test, and the remaining 3 blocks
        are partitioned into train and val.
    '''
    t_test = rasterio.transform.from_origin(500000.0, 600000.0, 20.0, 20.0)
    test_aoi = str(tmp_path / 'test_aoi.tif')
    _create_aoi_tiff(test_aoi, transform=t_test)

    blocks = {
        (0, 0): '/path/block_0_0.npz',
        (0, 256): '/path/block_0_256.npz',
        (256, 0): '/path/block_256_0.npz',
        (256, 256): '/path/block_256_256.npz',
    }
    base_counts = {
        (0, 0): [10, 20],
        (0, 256): [15, 25],
        (256, 0): [20, 30],
        (256, 256): [25, 35],
    }

    config = orchestration.PartitionParameters(
        val_test_ratios=(0.5, 0.0),
        buffer_step=0,
        reward_ratios={},
        scoring_alpha=1.0,
        scoring_beta=0.0,
        max_skew_rate=10.0,
        block_spec=(256, 256, 256, 256),
        test_aoi=test_aoi,
        canvas_crs='EPSG:3161',
        canvas_transform=t_test,
    )

    results = orchestration.create_blocks_partition(
        base_counts, base_counts, blocks, config
    )

    assert 'block_0_0' in results.partition_fpaths['test']
    assert len(results.partition_fpaths['test']) == 1
    assert len(results.partition_fpaths['val']) >= 1
    assert len(results.partition_fpaths['train']) >= 1


def test_scenario_2_train_val_only_zero_test_blocks():
    '''
    Scenario 2: User provides test_ratio=0.0 and no test_aoi.
    Given: 4 valid data blocks.
    When: Partitioning for training/validation only.
    Then: Test split is empty, and blocks are split between train and
        val.
    '''
    blocks = {
        (0, 0): '/path/b0.npz',
        (0, 256): '/path/b1.npz',
        (256, 0): '/path/b2.npz',
        (256, 256): '/path/b3.npz',
    }
    base_counts = {
        (0, 0): [10, 10],
        (0, 256): [10, 10],
        (256, 0): [10, 10],
        (256, 256): [10, 10],
    }

    config = orchestration.PartitionParameters(
        val_test_ratios=(0.25, 0.0),
        buffer_step=0,
        reward_ratios={},
        scoring_alpha=1.0,
        scoring_beta=0.0,
        max_skew_rate=10.0,
        block_spec=(256, 256, 256, 256),
        test_aoi=None,
    )

    results = orchestration.create_blocks_partition(
        base_counts, base_counts, blocks, config
    )

    assert not results.partition_fpaths['test']
    assert len(results.partition_fpaths['val']) == 1
    assert len(results.partition_fpaths['train']) == 3


def test_scenario_3_multi_zone_aoi_priority_and_buffering(tmp_path, mocker):
    '''
    Scenario 3: User provides test_aoi, val_aoi, and train_aoi with
        overlap.
    Given: Test and validation AOIs overlapping block (0, 0), and a
        train AOI covering (256, 256).
    When: Partitioning with buffer_step=1.
    Then: Overlap resolves to test > val > train with logged warnings.
    '''
    t_test = rasterio.transform.from_origin(500000.0, 600000.0, 20.0, 20.0)
    t_train = rasterio.transform.from_origin(
        500000.0 + 256 * 20, 600000.0 - 256 * 20, 20.0, 20.0
    )

    test_path = str(tmp_path / 'test.tif')
    val_path = str(tmp_path / 'val.tif')
    train_path = str(tmp_path / 'train.tif')

    _create_aoi_tiff(test_path, transform=t_test)
    _create_aoi_tiff(val_path, transform=t_test) # intentionally overlapping
    _create_aoi_tiff(train_path, transform=t_train)

    blocks = {
        (0, 0): '/path/b0.npz',
        (256, 256): '/path/b3.npz',
    }
    base_counts = {
        (0, 0): [10, 10],
        (256, 256): [10, 10],
    }

    mock_logger = mocker.MagicMock()

    config = orchestration.PartitionParameters(
        val_test_ratios=(0.0, 0.0),
        buffer_step=0,
        reward_ratios={},
        scoring_alpha=1.0,
        scoring_beta=0.0,
        max_skew_rate=10.0,
        block_spec=(256, 256, 256, 256),
        train_aoi=train_path,
        val_aoi=val_path,
        test_aoi=test_path,
        canvas_crs='EPSG:3161',
        canvas_transform=t_test,
    )

    results = orchestration.create_blocks_partition(
        base_counts,
        base_counts,
        blocks,
        config,
        logger=mock_logger,
    )

    assert 'b0' in results.partition_fpaths['test']
    assert 'b3' in results.partition_fpaths['train']
    assert not results.partition_fpaths['val']
    assert mock_logger.log.called
