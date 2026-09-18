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
Data preparation (experiment-materialized) pipeline.

Splits raw blocks into train/val(/test), computes train-only band
statistics, normalizes all splits, and emits the final dataset schema.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.artifacts.paths as paths
import landseg.geopipe.prepare.data_context as prepare_context
import landseg.geopipe.prepare.data_partition as prepare_partition
import landseg.geopipe.prepare.logger as prepare_logger
import landseg.geopipe.prepare.materialize_blocks as prepare_materialize


# ----- private types
class _CatalogViewConfig(typing.Protocol):
    @property
    def valid_pxs(self) -> dict[str, float]: ...

    @property
    def focal_target(self) -> str | None: ...

    @property
    def test_catalog(self) -> str | None: ...

    @property
    def non_overlapping_test_grid(self) -> bool: ...


class _PartitionConfig(typing.Protocol):
    @property
    def val_ratio(self) -> float: ...

    @property
    def test_ratio(self) -> float: ...

    @property
    def buffer_step(self) -> int: ...

    @property
    def train_aoi(self) -> str | None: ...

    @property
    def val_aoi(self) -> str | None: ...

    @property
    def test_aoi(self) -> str | None: ...

    @property
    def aoi_min_overlap(self) -> float: ...


class _ScoringConfig(typing.Protocol):
    @property
    def reward(self) -> dict[int, float]: ...

    @property
    def alpha(self) -> float: ...

    @property
    def beta(self) -> float: ...


class _HydrationConfig(typing.Protocol):
    @property
    def max_skew_rate(self) -> float: ...


class _PreparationPipelineConfig(typing.Protocol):
    @property
    def features(self) -> dict[str, typing.Any]: ...

    @property
    def targets(self) -> dict[str, typing.Any]: ...

    @property
    def catalog(self) -> _CatalogViewConfig: ...

    @property
    def partition(self) -> _PartitionConfig: ...

    @property
    def scoring(self) -> _ScoringConfig: ...

    @property
    def hydration(self) -> _HydrationConfig: ...

    @property
    def rebuild(self) -> bool: ...

    @property
    def output_dpath(self) -> str: ...


def run_data_preparation(
    artifact_paths: paths.ArtifactPaths,
    config: _PreparationPipelineConfig,
    tile_specs_tuple: tuple[int, int, int, int], # need to canonalize
    *,
    policy: artifacts.LifecyclePolicy,
    logger: prepare_logger.PreparationLogger,
) -> None:
    '''Run the preparation pipeline for an experiment.'''
    # build dataset context
    dataset_context = prepare_context.build_dataset_context(
        artifact_paths.data_ingestion.data_blocks.catalog,
        artifact_paths.data_ingestion.data_blocks.schema,
        config=config.catalog,
        user_features=config.features,
        user_targets=config.targets
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
        canvas_crs=dataset_context.crs,
        canvas_transform=dataset_context.transform,
    )
    prepare_partition.run_datablocks_partition(
        dataset_context,
        artifact_paths.data_preparation,
        partition_config,
        policy=policy,
        logger=logger,
    )
    assert logger.summary
    assert logger.summary['data_partition']
    d = logger.summary['data_partition']['duration_sec']
    logger.log('INFO', f'[COMPLETE] Dataset partitioning splits (D_{d:.2f}s)')

    # materiazlie
    logger.log('INFO', '[START] Block normalization')
    prepare_materialize.run_materialize_blocks(
        artifact_paths.data_preparation,
        dataset_context,
        policy=policy,
        logger=logger
    )
    assert logger.summary['normalization']
    d = logger.summary['normalization']['duration_sec']
    logger.log('INFO', f'[COMPLETE] Block normalization (D_{d:.2f}s)')
