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

# pylint: disable=missing-function-docstring

'''
Schema builders for prepared dataset artifacts.

Emits a dataset-wide JSON schema from materialized blocks, split
partitions, image statistics, and label statistics.

Public APIs:
    - build_schema: generate and persist dataset preparation schema.
'''

# standard imports
import datetime
import time
import typing
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.geopipe.contracts.preparation as contracts
import landseg.geopipe.prepare as prepare


# ----- typing aliases
PartitionCtrl = artifacts.Controller[contracts.BlocksPartition]
ImageStatsCtrl = artifacts.Controller[dict[str, contracts.ImageBandStats]]
LabelStatsCtrl = artifacts.Controller[dict[str, list[int]]]
SchemaCtrl = artifacts.Controller[contracts.PreparedSchema]


# ----- private types
class _PipelinePaths(typing.Protocol):
    @property
    def schema(self) -> str: ...
    @property
    def splits_source_blocks(self) -> str: ...
    @property
    def splits_prepared_blocks(self) -> str: ...
    @property
    def label_stats(self) -> str: ...
    @property
    def image_stats(self) -> str: ...


# ----- public functions
def build_schema(
    paths: _PipelinePaths,
    context: prepare.DatasetContext,
    *,
    policy: artifacts.LifecyclePolicy,
    logger: prepare.PreparationLogger,
) -> None:
    '''
    Generate and persist dataset preparation schema JSON.

    Combines artifact hashes, train/val/test split manifests, per-band
    image statistics, label class statistics, and target head
    reclassification hierarchy into a single verified preparation
    schema JSON artifact.

    Args:
        paths:
            preparation paths container.
        context:
            dataset preparation context with target heads topology.
        policy:
            lifecycle policy guiding rebuild behavior.
        logger:
            logger for progress and diagnostic output.
    '''
    start_time = time.perf_counter()

    # schema artifact controller
    schema_ctrl = SchemaCtrl(paths.schema, policy)
    schema = schema_ctrl.fetch()
    loaded = schema is not None

    if not schema:
        # artifacts file paths
        collected_artifacts = {
            'block_source': paths.splits_source_blocks,
            'block_prepared': paths.splits_prepared_blocks,
            'label_stats': paths.label_stats,
            'image_stats': paths.image_stats
        }

        # checksum the artifacts
        load = artifacts.Controller.load_json_or_fail
        checksums = {
            'block_source': load(paths.splits_source_blocks).sha256,
            'block_prepared': load(paths.splits_prepared_blocks).sha256,
            'label_stats': load(paths.label_stats).sha256,
            'image_stats': load(paths.image_stats).sha256
        }

        # read blocks splits
        ctrl = PartitionCtrl.load_json_or_fail(paths.splits_prepared_blocks)
        block_splits = ctrl.fetch()

        # read label stats
        ctrl = LabelStatsCtrl.load_json_or_fail(paths.label_stats)
        label_stats = ctrl.fetch()

        # read image stats
        ctrl = ImageStatsCtrl.load_json_or_fail(paths.image_stats)
        image_stats = ctrl.fetch()

        # target heads reclass hierarchy
        heads_schema: contracts.TargetHeadsSchema = {
            'head_names': list(context.targets.head_names),
            'head_parent': dict(context.targets.head_parent),
            'head_parent_cls': dict(context.targets.head_parent_cls),
            'num_classes': dict(context.targets.num_classes),
            'class_names': {
                k: list(v) for k, v in context.targets.class_names.items()
            },
            'ignore_classes': {
                k: list(v) for k, v in context.targets.ignore_classes.items()
            },
        }

        # populate schema dict
        schema = {
            'schema_version': contracts.PREPARED_SCHEMA_ID,
            'creation_time': datetime.datetime.now().strftime(c.TF_ISO8601),
            'artifacts': collected_artifacts,
            'checksums': checksums,
            'train_blocks': block_splits['train'],
            'val_blocks': block_splits['val'],
            'test_blocks': block_splits['test'],
            'label_stats': label_stats,
            'image_stats': image_stats,
            'image_array_key': 'image', # current convention
            'label_array_key': 'label', # current convention
            'heads': heads_schema,
        }
        schema_ctrl.persist(schema)
        logger.log('INFO', '[CHECKPOINT] Created dataset prepared schema')
    else:
        logger.log('INFO', '[CHECKPOINT] Loaded dataset prepared schema')

    # compile report
    duration = time.perf_counter() - start_time

    report: contracts.SchemaReport = {
        'status': 'loaded' if loaded else 'created',
        'duration_sec': duration,
        'schema_filepath': paths.schema,
        'classes_mapped': list(schema['label_stats'].keys()),
    }
    logger.set_schema_report(report)
