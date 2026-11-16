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
import pathlib
import time
import typing
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.geopipe.contracts.preparation as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.prepare.logger as prepare_logger
import landseg.geopipe.prepare.dataset as prepare_dataset


# ----- private types
class _PipelinePaths(typing.Protocol):
    @property
    def root(self) -> pathlib.Path: ...
    @property
    def schema(self) -> pathlib.Path: ...
    @property
    def splits_source_blocks(self) -> pathlib.Path: ...
    @property
    def splits_prepared_blocks(self) -> pathlib.Path: ...
    @property
    def label_stats(self) -> pathlib.Path: ...
    @property
    def image_stats(self) -> pathlib.Path: ...


# ----- public functions
def build_schema(
    paths: _PipelinePaths,
    context: prepare_dataset.DatasetView,
    ingestion_schema_fpath: pathlib.Path,
    *,
    policy: artifacts.LifecyclePolicy,
    logger: prepare_logger.PreparationLogger,
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
            dataset preparation view with target heads topology.
        ingestion_schema_fpath:
            resolved ingestion schema artifact path.
        policy:
            lifecycle policy guiding rebuild behavior.
        logger:
            logger for progress and diagnostic output.
    '''
    start_time = time.perf_counter()

    # schema artifact controller
    # - version handled explicitly so a legacy schema can be upgraded
    #   under an explicit rebuild instead of blocking in-place
    Ctrl = artifacts.Controller[geo_core.PreparedSchema]
    schema_ctrl = Ctrl(paths.schema, policy=policy)
    cur_schema = schema_ctrl.fetch(missing_ok=True)
    if (cur_schema is not None and
        cur_schema.get('schema_id') != geo_core.prepared_schema.SCHEMA_ID
    ):
        if policy != artifacts.LifecyclePolicy.REBUILD:
            raise artifacts.ArtifactError(
                f'Prepared schema at {paths.schema} has version '
                f'{cur_schema.get("schema_id")!r}; re-run data-prepare with '
                'rebuild enabled to upgrade it.'
            )
        cur_schema = None # upgrade in place

    if not cur_schema:
        # artifacts file paths
        collected_artifacts = {
            'block_source': str(paths.splits_source_blocks.resolve()),
            'block_prepared': str(paths.splits_prepared_blocks.resolve()),
            'label_stats': str(paths.label_stats.resolve()),
            'image_stats': str(paths.image_stats.resolve()),
        }

        # checksum the artifacts
        # TODO might not be necessary as _hash.json is referred when loading
        Ctrl = artifacts.Controller[dict[str, str]]
        checksums = {
            'block_source': Ctrl(paths.splits_source_blocks).sha256,
            'block_prepared': Ctrl(paths.splits_prepared_blocks).sha256,
            'label_stats': Ctrl(paths.label_stats).sha256,
            'image_stats': Ctrl(paths.image_stats).sha256
        }

        # forward ingestion lineage (single-upstream contract)
        ingestion_schema = pathlib.Path(ingestion_schema_fpath)
        collected_artifacts['ingestion_schema'] = str(ingestion_schema)
        checksums['ingestion_schema'] = Ctrl(ingestion_schema).sha256

        # read blocks splits
        Ctrl = artifacts.Controller[contracts.BlocksPartition]
        block_splits = Ctrl(paths.splits_prepared_blocks).fetch()

        # read label stats
        Ctrl = artifacts.Controller[dict[str, list[int]]]
        label_stats = Ctrl(paths.label_stats).fetch()

        # read image stats
        Ctrl = artifacts.Controller[dict[str, geo_core.ImageBandStats]]
        image_stats = Ctrl(paths.image_stats).fetch()

        # target heads reclass hierarchy
        # TODO shallow copy might not be necessary here
        heads_schema: geo_core.TargetHeadsSchema = {
            'head_names': list(context.targets.head_names),
            'head_parent': dict(context.targets.head_parent),
            'head_parent_cls': dict(context.targets.head_parent_cls),
            'num_classes': dict(context.targets.num_classes),
            'class_names': dict(context.targets.class_names),
            'ignore_classes': dict(context.targets.ignore_classes),
        }

        # populate schema dict
        schema: geo_core.prepared_schema.PreparedSchema = {
            'schema_id': geo_core.prepared_schema.SCHEMA_ID,
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
            'image_band_mapping': context.features.image_band_mapping,
            'heads': heads_schema,
        }
        schema_ctrl.persist(schema)
        logger.info('[CHECKPOINT] Created dataset prepared schema')
    else:
        schema = cur_schema
        logger.info('[CHECKPOINT] Loaded dataset prepared schema')

    # compile report
    duration = time.perf_counter() - start_time

    report: contracts.SchemaReport = {
        'status': 'loaded' if cur_schema is not None else 'created',
        'duration_sec': duration,
        'schema_filepath': str(paths.schema.resolve()),
        'classes_mapped': list(schema['label_stats'].keys()),
        'ingestion_schema_filepath': schema['artifacts'].get('ingestion_schema'),
    }
    logger.set_schema(report)
