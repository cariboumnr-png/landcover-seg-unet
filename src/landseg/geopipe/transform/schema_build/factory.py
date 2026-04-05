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

'''
Schema builders for dataset artifacts. Emits a dataset-wide JSON schema
from cached blocks and grid metadata, and can derive a minimal schema
from a single block (overfit mode).

Public APIs:
    - build_schema_full: Generate and write the dataset schema JSON.
    - build_schema_one_block: Build a minimal schema from one block.
'''

# standard imports
import os
# local imports
import landseg.geopipe.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.utils as utils

T_FORMAT = '%Y-%m-%dT%H:%M:%S'  # ISO-8601

# -------------------------------Public Function-------------------------------
def build_schema(root_dir: str) -> None:
    '''
    Generate and persist the dataset schema JSON from data and grid.

    Args:
        root_dir: Root folder for the dataset transform; The schema JSON
        file is written here.

    Raises:
        FileNotFoundError: If hash records for referenced artifacts are
            missing when resolving paths and checksums.
        ValueError: If artifacts hash value mismatch with the record.

    Note: this function does not return a schema dict, but write one as
    JSON to disk.
    '''

    # artifacts file paths
    _artifacts = {
        'block_source': f'{root_dir}/block_source.json',
        'block_splits': f'{root_dir}/block_splits.json',
        'label_stats': f'{root_dir}/label_stats.json',
        'image_stats': f'{root_dir}/image_stats.json'
    }

    # checksum the artifacts
    checksums = {
        'block_source': _resolve(_artifacts['block_source']),
        'block_splits': _resolve(_artifacts['block_splits']),
        'label_stats': _resolve(_artifacts['label_stats']),
        'image_stats': _resolve(_artifacts['image_stats'])
    }

    # read blocks splits
    block_splits: geo_core.BlocksPartition
    block_splits = utils.load_json(_artifacts['block_splits'])

    # read image stats
    image_stats: dict[str, geo_core.ImageBandStats]
    image_stats = utils.load_json(_artifacts['image_stats'])

    # read label stats
    label_stats: dict[str, list[int]]
    label_stats = utils.load_json(_artifacts['label_stats'])

    # populate schema dict
    schema: geo_core.TransformSchema = {
        'schema_version': '1.0',
        'creation_time': utils.get_timestamp(T_FORMAT),
        'artifacts': _artifacts,
        'checksums': checksums,
        'train_blocks': block_splits['train'],
        'val_blocks': block_splits['val'],
        'test_blocks': block_splits['test'],
        'label_stats': label_stats,
        'image_stats': image_stats,
        'image_array_key': 'image', # current convention
        'label_array_key': 'label', # current convention
    }

    # write schema to json
    artifacts.write_json_hash(f'{root_dir}/schema.json', schema)

# ------------------------------private  function------------------------------
def _resolve(fpath: str) -> str:
    '''Resolve an artifact path to its recorded SHA-256 in hash.json.'''

    # early exit if file does not exist
    if not os.path.exists(fpath):
        return ''

    # get file root and name
    root = os.path.dirname(fpath)
    fname = os.path.basename(fpath)

    # default hash record at root
    try:
        hash_records: dict[str, str] = utils.load_json(f'{root}/_hash.json')
    except FileNotFoundError as e:
        raise e
    # sanity checks
    if not 'root' in hash_records and hash_records['root'] == root:
        raise ValueError('Hash records root not matching with input root')
    if fname not in hash_records:
        raise ValueError('File hash not in record')

    # return a dict
    return hash_records[fname]
