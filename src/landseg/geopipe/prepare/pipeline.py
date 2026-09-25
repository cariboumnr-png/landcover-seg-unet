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

'''
Data preparation (experiment-materialized) pipeline.

Splits raw blocks into train/val(/test), computes train-only band
statistics, normalizes all splits, and emits the final dataset schema.
'''

# standard imports
from __future__ import annotations
# local imports
import landseg.artifacts as artifacts
import landseg.artifacts.paths as paths
import landseg.geopipe.contracts as contracts
import landseg.geopipe.prepare.context as prepare_context
import landseg.geopipe.prepare.dataset as prepare_dataset
import landseg.geopipe.prepare.logger as prepare_logger
import landseg.geopipe.prepare.materialize as prepare_materialize
import landseg.geopipe.prepare.partition as prepare_partition


# ----- public functions
def run_data_preparation(
    artifact_paths: paths.ArtifactPaths,
    config: contracts.PreparationPipelineConfig,
    tile_specs_tuple: tuple[int, int, int, int], # need to canonalize
    *,
    policy: artifacts.LifecyclePolicy,
    logger: prepare_logger.PreparationLogger,
) -> None:
    '''Run the preparation pipeline for an experiment.'''
    # resolve preparation context
    prep_context = prepare_context.build_preparation_context(
        artifact_paths.data_ingestion.data_blocks.catalog,
        artifact_paths.data_ingestion.data_blocks.schema,
    )

    # build dataset view
    dataset_params = prepare_dataset.DatasetViewParameters(
        valid_pxs=config.catalog.valid_pxs,
        focal_target=config.catalog.focal_target,
        test_catalog=config.catalog.test_catalog,
        non_overlapping_test_grid=config.catalog.non_overlapping_test_grid,
        features=config.features,
        targets=config.targets,
    )
    dataset_view = prepare_dataset.build_dataset_view(
        prep_context.catalog_fpath,
        prep_context.schema,
        parameters=dataset_params,
        canvas_crs=prep_context.canvas_crs,
        canvas_transform=prep_context.canvas_transform,
    )

    # datablocks partition
    logger.log('INFO', '[START] Dataset partitioning splits')
    # data preparation config aliases
    partition = config.partition
    scoring = config.scoring
    hydration = config.hydration
    # partition config
    partition_config = prepare_partition.PartitionParameters(
        val_test_ratios=(partition.val_ratio, partition.test_ratio),
        buffer_step=partition.buffer_step,
        reward_ratios=scoring.reward,
        scoring_alpha=scoring.alpha,
        scoring_beta=scoring.beta,
        max_skew_rate=hydration.max_skew_rate,
        block_spec=tile_specs_tuple,
        train_aoi=partition.train_aoi,
        val_aoi=partition.val_aoi,
        test_aoi=partition.test_aoi,
        aoi_min_overlap=partition.aoi_min_overlap,
        canvas_crs=dataset_view.crs,
        canvas_transform=dataset_view.transform,
    )
    prepare_partition.run_datablocks_partition(
        dataset_view,
        artifact_paths.data_preparation,
        partition_config,
        policy=policy,
        logger=logger,
    )
    assert logger.summary
    assert logger.summary['data_partition']
    d = logger.summary['data_partition']['duration_sec']
    logger.log('INFO', f'[COMPLETE] Dataset partitioning splits (D_{d:.2f}s)')

    # materialize
    logger.log('INFO', '[START] Block normalization')
    prepare_materialize.run_materialize_blocks(
        artifact_paths.data_preparation,
        dataset_view,
        policy=policy,
        logger=logger
    )
    assert logger.summary['normalization']
    d = logger.summary['normalization']['duration_sec']
    logger.log('INFO', f'[COMPLETE] Block normalization (D_{d:.2f}s)')
