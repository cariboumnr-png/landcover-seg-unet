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
    context: prepare_context.PreparationContext,
    prep_paths: paths.PreparationPaths,
    config: contracts.PreparationPipelineConfig,
    *,
    policy: artifacts.LifecyclePolicy,
    logger: prepare_logger.PreparationLogger,
) -> None:
    '''Run the preparation pipeline for an experiment.'''
    # build dataset view
    dataset_view_inputs = prepare_dataset.DataBlocksManifestInputs(
        schema=context.schema,
        catalog=context.catalog,
        test_catalog=context.test_catalog,
    )
    dataset_view_config = prepare_dataset.DatasetViewConfig(
        head_pxs_thres=config.datasetview.valid_pxs,
        focal_head=config.datasetview.focal_target,
        non_overlapping_test_grid=config.datasetview.non_overlapping_test_grid,
        features=config.datasetview.features,
        targets=config.datasetview.targets,
    )
    dataset_view = prepare_dataset.build_dataset_view(
        dataset_view_inputs,
        dataset_view_config,
        canvas_crs=context.canvas_crs,
        canvas_transform=context.canvas_transform,
    )

    # datablocks partition
    logger.info('[START] Dataset partitioning splits')
    # data preparation config aliases
    partition = config.partition
    scoring = config.scoring
    hydration = config.hydration
    # partition config
    aoi_config = (
        prepare_partition.AOIConfig(
            train_aoi=partition.train_aoi,
            val_aoi=partition.val_aoi,
            test_aoi=partition.test_aoi,
            min_overlap=partition.aoi_min_overlap,
            canvas_crs=dataset_view.crs,
            canvas_transform=dataset_view.transform,
        )
        if partition.train_aoi or partition.val_aoi or partition.test_aoi
        else None
    )
    hydration_config = (
        prepare_partition.HydrationConfig(
            reward_ratios=scoring.reward,
            scoring_alpha=scoring.alpha,
            scoring_beta=scoring.beta,
            max_skew_rate=hydration.max_skew_rate,
        )
        if bool(scoring.reward)
        else None
    )
    partition_config = prepare_partition.PartitionConfig(
        val_test_ratios=(partition.val_ratio, partition.test_ratio),
        block_spec=context.block_specs,
        buffer_step=partition.buffer_step,
        aoi=aoi_config,
        hydration=hydration_config,
    )
    prepare_partition.run_datablocks_partition(
        dataset_view,
        prep_paths,
        partition_config,
        policy=policy,
        logger=logger,
    )
    assert logger.summary
    assert logger.summary['data_partition']
    d = logger.summary['data_partition']['duration_sec']
    logger.info(f'[COMPLETE] Dataset partitioning splits (D_{d:.2f}s)')

    # materialize
    logger.info('[START] Block normalization')
    prepare_materialize.run_materialize_blocks(
        prep_paths,
        dataset_view,
        policy=policy,
        logger=logger
    )
    assert logger.summary['normalization']
    d = logger.summary['normalization']['duration_sec']
    logger.info(f'[COMPLETE] Block normalization (D_{d:.2f}s)')
