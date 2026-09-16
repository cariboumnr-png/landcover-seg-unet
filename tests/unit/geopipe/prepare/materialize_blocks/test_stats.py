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

'''Unit tests for block image and label statistics aggregation.'''

# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.prepare.materialize_blocks.stats as mat_stats


# ----- `count_label` tests
def test_count_label_success():
    '''
    Given: Raw class counts mapping and selected block coordinate
        strings.
    When: Running count_label.
    Then: Correctly aggregate per-head class pixel counts.
    '''
    raw_counts = {
        (0, 0): {'head1': [10, 20], 'head2': [5]},
        (0, 1): {'head1': [30, 40], 'head2': [15]},
        (1, 0): {'head1': [100, 200], 'head2': [50]},
    }
    selected = ['row_000000_col_000000', 'row_000001_col_000000']

    result = mat_stats.count_label(raw_counts, selected)

    # head1: [10 + 30, 20 + 40] = [40, 60]
    # head2: [5 + 15] = [20]
    assert result == {'head1': [40, 60], 'head2': [20]}


def test_count_label_invalid_coordinate():
    '''
    Given: Block ID string with invalid coordinate format.
    When: Running count_label.
    Then: Raise ValueError.
    '''
    raw_counts = {(0, 0): {'head1': [10]}}
    with pytest.raises(ValueError, match='Invalid block coord string'):
        mat_stats.count_label(raw_counts, ['invalid_name.npz'])


# ----- `aggregate_image_stats` tests
def test_aggregate_image_stats_success(mocker):
    '''
    Given: A list of DataBlock paths with mock band metrics.
    When: Running aggregate_image_stats.
    Then: Correctly aggregate band means and standard deviations.
    '''
    mock_block = mocker.Mock()
    mock_block.data.image = numpy.random.rand(2, 64, 64).astype(
        numpy.float32
    )
    mock_block.manifest = {
        'image_stats': {
            'band_0': {
                'count': 100,
                'mean': 10.0,
                'm2': 400.0,
            },
            'band_1': {
                'count': 100,
                'mean': 20.0,
                'm2': 900.0,
            },
        }
    }

    mocker.patch(
        'landseg.geopipe.core.DataBlock.load', return_value=mock_block
    )

    input_blocks = {'block1.npz', 'block2.npz'}
    result = mat_stats.aggregate_image_stats(input_blocks)

    assert len(result) == 2
    assert 'band_0' in result
    assert 'band_1' in result
    assert result['band_0']['total_count'] == 200
    assert result['band_0']['current_mean'] == pytest.approx(10.0)
    assert result['band_0']['std'] > 0.0


def test_aggregate_image_stats_channel_indices(mocker):
    '''
    Given: Input blocks and explicit channel index selection.
    When: Running aggregate_image_stats with channel_indices.
    Then: Aggregate statistics only for selected channels remapped.
    '''
    mock_block = mocker.Mock()
    mock_block.data.image = numpy.random.rand(3, 64, 64).astype(
        numpy.float32
    )
    mock_block.manifest = {
        'image_stats': {
            'band_0': {'count': 100, 'mean': 10.0, 'm2': 400.0},
            'band_1': {'count': 100, 'mean': 20.0, 'm2': 900.0},
            'band_2': {'count': 100, 'mean': 30.0, 'm2': 1600.0},
        }
    }

    mocker.patch(
        'landseg.geopipe.core.DataBlock.load', return_value=mock_block
    )

    input_blocks = {'block1.npz'}
    result = mat_stats.aggregate_image_stats(
        input_blocks, channel_indices=[2, 0]
    )

    assert len(result) == 2
    # band_0 maps to orig band_2 (mean 30.0)
    assert result['band_0']['current_mean'] == pytest.approx(30.0)
    # band_1 maps to orig band_0 (mean 10.0)
    assert result['band_1']['current_mean'] == pytest.approx(10.0)


def test_aggregate_image_stats_empty():
    '''
    Given: An empty input block set.
    When: Running aggregate_image_stats.
    Then: Raise ValueError.
    '''
    with pytest.raises(ValueError, match='input_blocks cannot be empty'):
        mat_stats.aggregate_image_stats(set())
