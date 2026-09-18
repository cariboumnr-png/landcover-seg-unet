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
Data ingestion pipeline.

Prepares the world grid, materializes domain knowledge, and builds
the immutable raw block catalogue for later experiments.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.artifacts.paths as paths
import landseg.geopipe.ingest.context as ingest_context
import landseg.geopipe.ingest.data_blocks as ingest_data_blocks
import landseg.geopipe.ingest.domain_maps as ingest_domain_maps
import landseg.geopipe.ingest.logger as ingest_logger


# ----- private types
class _IngestionPipelineConfig(typing.Protocol):
    @property
    def domains(self) -> _DomainsConfig: ...

    @property
    def datablocks(self) -> _DataBlocksConfig: ...

    @property
    def rebuild(self) -> bool: ...

    @property
    def harmonization_run(self) -> int | str | None: ...

    @property
    def output_dpath(self) -> str: ...


class _DomainsConfig(typing.Protocol):
    @property
    def valid_threshold(self) -> float: ...

    @property
    def target_variance(self) -> float: ...


class _DataBlocksConfig(typing.Protocol):
    @property
    def ignore_index(self) -> int: ...

    @property
    def image_dem_pad(self) -> int: ...

    @property
    def add_topo(self) -> list[str] | None: ...

    @property
    def add_spectral(self) -> list[str] | None: ...


# ----- public functions
def run_data_ingestion(
    artifact_paths: paths.ArtifactPaths,
    config: _IngestionPipelineConfig,
    *,
    policy: artifacts.LifecyclePolicy,
    logger: ingest_logger.IngestionLogger
) -> None:
    '''Run the ingestion pipeline.'''
    assert logger.summary

    # build ingestion context from harmonization
    context = ingest_context.build_ingestion_context(
        artifact_paths.data_harmonization,
        config.harmonization_run
    )

    # ----- canonical world grid reference
    world_grid = context.grid
    gid = world_grid.gid
    logger.log('INFO', f'[COMPLETE] World grid loaded: {gid}')

    # ----- materialize domain maps
    if context.domains:
        logger.log('INFO', '[START] Domain maps preparation')
        domain_paths = artifact_paths.data_ingestion.domains
        domain_configs = [
            ingest_domain_maps.DomainBuildingParameters(
                input_fpath=path,
                domain_fpath=domain_paths.domain_map_fpath(name),
                tiles_fpath=domain_paths.mapped_tiles_fpath(name, gid),
                valid_threshold=config.domains.valid_threshold,
                target_variance=config.domains.target_variance,
            ) for name, path in context.domains.items()
        ]
        ingest_domain_maps.prepare_domain_maps(
            world_grid,
            domain_configs,
            policy=policy,
            logger=logger,
        )

        d = sum(dm['duration_sec'] for dm in logger.summary['domain_maps'])
        logger.log('INFO', f'[COMPLETE] Domain maps preparation (D_{d:.2f}s)')
    else:
        logger.log('INFO', '[NOTE] No domain knowledge layers provided')

    # ----- build canonical data blocks if provided
    if not context.has_data:
        logger.log('INFO', 'Harmonized feature/label rasters not provided')
    else:
        logger.log('INFO', '[START] Canonical data blocks building')
        assert context.features
        data_blocks_config = ingest_data_blocks.BlockBuildingParameters(
            image_fpath=context.features,
            label_fpath=context.labels,
            dem_pad=config.datablocks.image_dem_pad,
            ignore_index=config.datablocks.ignore_index,
            add_spectral=config.datablocks.add_spectral,
            add_topo=config.datablocks.add_topo,
        )
        ingest_data_blocks.run_blocks_building(
            world_grid,
            artifact_paths.data_ingestion.data_blocks,
            data_blocks_config,
            policy=policy,
            logger=logger,
        )

        assert logger.summary['data_blocks'] # typing
        d = logger.summary['data_blocks']['duration_sec']
        logger.log('INFO', f'[COMPLETE] Canonical data blocks preparation (D_{d:.2f}s)')
