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

'''Unit tests for normal blocks image statistics aggregation (stats.py).'''

# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.transform.normal_blocks.stats as norm_stats


# ----- `aggregate_image_stats` tests
def test_aggregate_image_stats_success(mocker):
    # mock geo_core.DataBlock.load to return a fake data block with sample data
    mock_block = mocker.Mock()
    mock_block.data.image = numpy.random.rand(2, 256, 256).astype(numpy.float32)

    # manifest stats for 2 bands
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
            }
        }
    }

    mocker.patch('landseg.geopipe.core.DataBlock.load', return_value=mock_block)

    input_blocks = {'block1.npz', 'block2.npz'}
    result = norm_stats.aggregate_image_stats(input_blocks)

    # verify 2 bands are calculated
    assert len(result) == 2
    assert 'band_0' in result
    assert 'band_1' in result
    assert result['band_0']['total_count'] == 200 # 2 blocks * 100
    assert result['band_0']['current_mean'] == pytest.approx(10.0)
    assert result['band_0']['std'] > 0.0


def test_aggregate_image_stats_empty():
    with pytest.raises(ValueError, match='input_blocks cannot be empty'):
        norm_stats.aggregate_image_stats(set())
