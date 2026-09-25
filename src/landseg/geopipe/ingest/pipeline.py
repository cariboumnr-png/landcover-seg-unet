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
Data ingestion pipeline.

Prepares the world grid, materializes domain knowledge, and builds
the immutable raw block catalogue for later experiments.
'''

# standard imports
from __future__ import annotations
# local imports
import landseg.artifacts as artifacts
import landseg.artifacts.paths as paths
import landseg.geopipe.contracts as contracts
import landseg.geopipe.ingest.blocks as ingest_blocks
import landseg.geopipe.ingest.context as ingest_context
import landseg.geopipe.ingest.domains as ingest_domains
import landseg.geopipe.ingest.logger as ingest_logger


# ----- public functions
def run_data_ingestion(
    artifact_paths: paths.ArtifactPaths,
    harmonization_record: contracts.HarmonizationRunRecord,
    config: contracts.IngestionPipelineConfig,
    *,
    policy: artifacts.LifecyclePolicy,
    logger: ingest_logger.IngestionLogger,
) -> None:
    '''Run the ingestion pipeline.'''
    assert logger.summary

    # build ingestion context
    logger.log('INFO', '[START] Building data ingestion context')
    context = ingest_context.build_ingestion_context(
        artifact_paths.data_harmonization,
        harmonization_record,
        artifact_paths.data_ingestion.runs_manifest,
        artifact_paths.data_ingestion.data_blocks.schema,
        config
    )
    logger.set_harmonization_reference(
        run_uid=context.run_uid,
        run_id=context.harmonization_run_id,
    )
    logger.set_identity(context.current_run_identity)

    # early exit
    if context.collided_run_uid is not None:
        logger.set_summary_status('SKIPPED')
        logger.log(
            'INFO',
            f'[COMPLETE] Ingestion run with the same inputs and configs '
            f'already done (run uid: {context.collided_run_uid}), skipped'
        )
        return
    logger.log('INFO', '[COMPLETE] Data ingestion context built')

    # ----- materialize domain maps
    if context.domains:
        logger.log('INFO', '[START] Domain maps preparation')
        domain_paths = artifact_paths.data_ingestion.domains
        domain_configs = [
            ingest_domains.DomainBuildingParameters(
                input_fpath=path,
                domain_fpath=domain_paths.domain_map_fpath(name),
                tiles_fpath=domain_paths.mapped_tiles_fpath(name, context.grid.gid),
                valid_threshold=config.domains.valid_threshold,
                target_variance=config.domains.target_variance,
            ) for name, path in context.domains.items()
        ]
        ingest_domains.prepare_domain_maps(
           context.grid,
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
        data_blocks_config = ingest_blocks.BlockBuildingParameters(
            image_fpath=context.features,
            label_fpath=context.labels,
            dem_pad=config.datablocks.image_dem_pad,
            ignore_index=config.datablocks.ignore_index,
            add_spectral=config.datablocks.add_spectral,
            add_topo=config.datablocks.add_topo,
        )
        ingest_blocks.run_blocks_building(
            context.grid,
            artifact_paths.data_ingestion.data_blocks,
            data_blocks_config,
            policy=policy,
            logger=logger,
        )

        assert logger.summary['data_blocks'] # typing
        d = logger.summary['data_blocks']['duration_sec']
        logger.log('INFO', f'[COMPLETE] Canonical data blocks preparation (D_{d:.2f}s)')
