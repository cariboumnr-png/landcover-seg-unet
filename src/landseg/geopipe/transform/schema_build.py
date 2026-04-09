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
import datetime
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core

T_FORMAT = '%Y-%m-%dT%H:%M:%S'  # ISO-8601

# typing aliases
PartitionCtrl = artifacts.Controller[geo_core.BlocksPartition]
ImageStatsCtrl = artifacts.Controller[dict[str, geo_core.ImageBandStats]]
LabelStatsCtrl = artifacts.Controller[dict[str, list[int]]]
SchemaCtrl = artifacts.Controller[geo_core.TransformSchema]
Resolver = artifacts.Controller.load_json_or_fail

# -------------------------------Public Function-------------------------------
def build_schema(
    paths: artifacts.TransformPaths,
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
    schema_ctrl = SchemaCtrl(paths.schema, policy)
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
            'block_source': Resolver(paths.splits_source_blocks).sha256,
            'block_transform': Resolver(paths.splits_transformed_blocks).sha256,
            'label_stats': Resolver(paths.label_stats).sha256,
            'image_stats': Resolver(paths.image_stats).sha256
        }

        # read blocks splits
        ctrl = PartitionCtrl.load_json_or_fail(paths.splits_transformed_blocks)
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
            'schema_version': geo_core.transform_types.TRANSFORM_SCHEMA_ID,
            'creation_time': datetime.datetime.now().strftime(T_FORMAT),
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
