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
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.transform.common as common
import landseg.utils as utils

T_FORMAT = '%Y-%m-%dT%H:%M:%S'  # ISO-8601

# typing aliases
PartitionCtrl = artifacts.Controller[geo_core.BlocksPartition]
ImageStatsCtrl = artifacts.Controller[dict[str, geo_core.ImageBandStats]]
LabelStatsCtrl = artifacts.Controller[dict[str, list[int]]]
SchemaCtrl = artifacts.Controller[geo_core.TransformSchema]

# -------------------------------Public Function-------------------------------
def build_schema(
    paths: common.TransformPaths,
    *,
    policy: artifacts.LifecyclePolicy
) -> None:
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

    # schema artifact controller
    schema_ctrl = SchemaCtrl(paths.schema, 'json', policy)
    schema = schema_ctrl.fetch()

    if not schema:
        # artifacts file paths
        collected_artifacts = {
            'block_source': paths.splits_source_blocks,
            'block_transform': paths.splits_transformed_blocks,
            'label_stats': paths.label_stats,
            'image_stats': paths.image_stats
        }

        # checksum the artifacts
        checksums = {
            'block_source': _resolve(paths.splits_source_blocks),
            'block_transform': _resolve(paths.splits_transformed_blocks),
            'label_stats': _resolve(paths.label_stats),
            'image_stats': _resolve(paths.image_stats)
        }

        # read blocks splits
        ctrl = PartitionCtrl.load_json_or_fail(paths.splits_source_blocks)
        block_splits = ctrl.fetch()
        assert block_splits # typing assertion

        # read label stats
        ctrl = LabelStatsCtrl.load_json_or_fail(paths.label_stats)
        label_stats = ctrl.fetch()
        assert label_stats # typing assertion

        # read image stats
        ctrl = ImageStatsCtrl.load_json_or_fail(paths.image_stats)
        image_stats = ctrl.fetch()
        assert image_stats # typing assertion

        # populate schema dict
        schema = {
            'schema_version': geo_core.transform_types.SCHEMA_ID,
            'creation_time': utils.get_timestamp(T_FORMAT),
            'artifacts': collected_artifacts,
            'checksums': checksums,
            'train_blocks': block_splits['train'],
            'val_blocks': block_splits['val'],
            'test_blocks': block_splits['test'],
            'label_stats': label_stats,
            'image_stats': image_stats,
            'image_array_key': 'image', # current convention
            'label_array_key': 'label', # current convention
        }
        schema_ctrl.persist(schema)

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
