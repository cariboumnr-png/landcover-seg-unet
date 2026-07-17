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

'''Unit tests for data partition label statistics (stats.py).'''

# local imports
import landseg.geopipe.transform.data_partition.stats as part_stats


# ----- `count_label` tests
def test_count_label_success(mocker):
    '''
    Given: A list of DataBlock file paths.
    When: Running count_label to compute class statistics.
    Then: Correctly aggregate absolute class frequency counts across
        all blocks.
    '''
    mock_block = mocker.Mock()
    mock_block.manifest = {
        'label_count': {
            'head1': [10, 20],
        }
    }

    mocker.patch('landseg.geopipe.core.DataBlock.load', return_value=mock_block)

    block_files = ['file1.npz', 'file2.npz']
    result = part_stats.count_label(block_files)

    assert result == {'head1': [20, 40]}
