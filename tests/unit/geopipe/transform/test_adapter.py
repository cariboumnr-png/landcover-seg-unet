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

'''Unit tests for catalog data blocks adapter logic (adapter.py).'''

# local imports
import landseg.geopipe.transform.adapter as adapter


# ----- `data_blocks_adapter` tests
def test_data_blocks_adapter(mocker):
    # mock schema and catalog dicts
    mock_schema = {
        'tensor_shapes': {
            'image': {'H': 256, 'W': 256}
        },
        'labels': {
            'label_ignore_cls': ['unmapped']
        }
    }
    mock_catalog = {
        'row_000000_col_000000': {
            'row_col': [0, 0],
            'file_path': 'path/to/block_0.npz',
            'valid_px_ratios': {'image': 1.0},
            'class_count': {
                'unmapped': [100, 200]
            }
        },
        'row_000010_col_000010': {
            'row_col': [10, 10],
            'file_path': 'path/to/block_1.npz',
            'valid_px_ratios': {'image': 0.1},
            'class_count': {
                'unmapped': [50, 50]
            }
        }
    }

    # mock Controller.load_json_or_fail
    def mock_load(filepath):
        mock_ctrl = mocker.Mock()
        if 'schema' in filepath:
            mock_ctrl.fetch.return_value = mock_schema
        else:
            mock_ctrl.fetch.return_value = mock_catalog
        return mock_ctrl

    mocker.patch(
        'landseg.artifacts.Controller.load_json_or_fail',
        side_effect=mock_load
    )

    # config mock
    mock_config = mocker.Mock()
    mock_config.valid_pxs = {'image': 0.5}
    mock_config.focal_target = 'unmapped'
    mock_config.non_overlapping_test_grid = False

    result = adapter.data_blocks_adapter(
        dev_catalog='dev_cat.json',
        dev_schema='dev_schema.json',
        test_catalog='test_cat.json',
        config=mock_config
    )

    # check focal head matches
    assert result.focal_head == 'unmapped'

    # block_0 is valid (valid_px_ratios.image = 1.0 >= 0.5)
    # block_1 is invalid (valid_px_ratios.image = 0.1 < 0.5)
    # dev_blocks has coordinate (0, 0) for block_0
    assert (0, 0) in result.dev_blocks
    assert (10, 10) not in result.dev_blocks


def test_is_valid_block():
    thresholds = {'image': 0.9, 'label': 0.8}

    # case 1: all ratios met
    ratios_ok = {'image': 0.95, 'label': 0.85}
    assert adapter._is_valid_block(thresholds, ratios_ok) is True

    # case 2: one ratio below threshold
    ratios_bad = {'image': 0.85, 'label': 0.85}
    assert adapter._is_valid_block(thresholds, ratios_bad) is False
