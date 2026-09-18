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

'''Unit tests for DatasetCatalog and DatasetBlockMeta.'''

# standard imports
import json
# third-party imports
import pytest
# local imports
import landseg.geopipe.core as geo_core


# ----- fixtures
@pytest.fixture(name='sample_entry')
def sample_entry_fixture() -> geo_core.DatasetBlockMeta:
    '''Provide a sample DatasetBlockMeta dictionary.'''
    return {
        'block_name': 'row_000020_col_000010',
        'file_path': '/path/to/block.npz',
        'row_col': [20, 10],
        'valid_px_ratios': {'image': 1.0, 'label': 0.95},
        'class_count': {'landcover': [100, 200, 300]},
        'schema_version': '1.0.0',
        'creation_time': '2026-09-16T12:00:00',
        'sha_256': 'abc123def456',
        'aligned_grid': 'grid_utm17n',
        'source_image': '/path/to/img.tif',
        'source_image_sha_256': 'img_hash_123',
        'source_label': '/path/to/lbl.tif',
        'source_label_sha_256': 'lbl_hash_456',
    }


# ----- `DatasetCatalog` tests
def test_dataset_catalog_container_protocol(sample_entry):
    '''
    Given: An empty DatasetCatalog.
    When: Populating entries and checking dictionary interface.
    Then: Correctly gets, sets, iterates, and tracks length.
    '''
    catalog = geo_core.DatasetCatalog()
    assert len(catalog) == 0

    catalog[(10, 20)] = sample_entry
    assert len(catalog) == 1
    assert (10, 20) in catalog
    assert catalog[(10, 20)]['block_name'] == 'row_000020_col_000010'
    assert list(iter(catalog)) == [(10, 20)]


def test_dataset_catalog_from_dict(sample_entry):
    '''
    Given: A raw dictionary with row/col formatted string keys.
    When: Calling DatasetCatalog.from_dict.
    Then: Parses string keys to (x, y) coordinate tuples.
    '''
    raw = {'row_000020_col_000010': sample_entry}
    catalog = geo_core.DatasetCatalog.from_dict(raw)

    assert len(catalog) == 1
    # row 20, col 10 -> (col=10, row=20)
    assert (10, 20) in catalog
    assert catalog[(10, 20)]['sha_256'] == 'abc123def456'


def test_dataset_catalog_to_json_payload(sample_entry):
    '''
    Given: A populated DatasetCatalog.
    When: Calling to_json_payload.
    Then: Returns formatted JSON string matching original entries.
    '''
    catalog = geo_core.DatasetCatalog()
    catalog[(10, 20)] = sample_entry

    payload_str = catalog.to_json_payload()
    assert isinstance(payload_str, str)

    parsed = json.loads(payload_str)
    assert 'row_000020_col_000010' in parsed
    assert parsed['row_000020_col_000010']['block_name'] == (
        'row_000020_col_000010'
    )
