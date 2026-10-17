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

'''Unit tests for dataset manifest compiler (compiler.py).'''

# standard imports
import json
import pathlib
# third-party imports
import pytest
# local imports
import landseg.geopipe.harmonize.manifest.compiler as compiler


# ----- `compile_dataset_manifest` tests
def test_compile_dataset_manifest_success(tmp_path):
    '''
    Given: A valid root dataset manifest referencing sub-manifest files.
    When: `compile_dataset_manifest` is called.
    Then: Compile and return normalized ManifestEntry dictionaries.
    '''
    sub_manifest_fp = tmp_path / 's2_manifest.json'
    sub_entry = {
        'name': 's2_sample',
        'path': str(tmp_path / 's2.tif'),
        'band_mapping': {'1': 'blue', '2': 'green', '3': 'red'},
        'category': 'features',
    }
    with open(sub_manifest_fp, 'w', encoding='UTF-8') as f:
        json.dump(sub_entry, f)

    root_manifest_fp = tmp_path / 'dataset_manifest.json'
    root_manifest = [
        {
            'name': 's2_sample',
            'path': str(tmp_path / 's2.tif'),
            'manifest': str(sub_manifest_fp),
        }
    ]
    with open(root_manifest_fp, 'w', encoding='UTF-8') as f:
        json.dump(root_manifest, f)

    compiled = compiler.compile_dataset_manifest(root_manifest_fp)
    assert len(compiled) == 1
    assert compiled[0]['name'] == 's2_sample'
    assert compiled[0]['category'] == 'features'


def test_compile_dataset_manifest_not_a_list(tmp_path):
    '''
    Given: A root dataset manifest JSON containing a dictionary instead.
    When: `compile_dataset_manifest` is called.
    Then: Raise DatasetManifestError.
    '''
    invalid_root_fp = tmp_path / 'invalid_root.json'
    with open(invalid_root_fp, 'w', encoding='UTF-8') as f:
        json.dump({'name': 'not_a_list'}, f)

    with pytest.raises(
        compiler.DatasetManifestError,
        match='expected to read as a list',
    ):
        compiler.compile_dataset_manifest(invalid_root_fp)


def test_compile_dataset_manifest_invalid_sub_entry(tmp_path):
    '''
    Given: A root manifest referencing a non-existent or invalid entry.
    When: `compile_dataset_manifest` is called.
    Then: Raise DatasetManifestError.
    '''
    sub_manifest_fp = tmp_path / 'invalid_sub.json'
    with open(sub_manifest_fp, 'w', encoding='UTF-8') as f:
        json.dump({'invalid': 'content'}, f)

    root_manifest_fp = tmp_path / 'dataset_manifest.json'
    root_manifest = [
        {
            'name': 'invalid_entry',
            'path': '/nonexistent.tif',
            'manifest': str(sub_manifest_fp),
        }
    ]
    with open(root_manifest_fp, 'w', encoding='UTF-8') as f:
        json.dump(root_manifest, f)

    with pytest.raises(
        compiler.DatasetManifestError,
        match='Invalid manifest entry 0',
    ):
        compiler.compile_dataset_manifest(root_manifest_fp)
