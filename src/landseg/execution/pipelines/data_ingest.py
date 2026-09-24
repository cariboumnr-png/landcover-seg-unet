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
import landseg.configs as configs
import landseg.geopipe.contracts.harmonization as harm_contracts
import landseg.geopipe.ingest as ingest
import landseg.utils as utils

# ----- typing aliases
ConfigController = artifacts.Controller[dict]


# ----- public functions
def exec_ingest_data(config: configs.RootConfig) -> None:
    '''
    Run data ingestion pipeline across planned harmonization batches.

    Discovers available upstream harmonization runs and matches them
    against the downstream ingestion ledger to resolve pending batches.
    Executes each planned batch sequentially into the block pool.

    Args:
        config:
            Root execution configuration containing data and ingestion
            settings.
    '''
    initial_artifact_paths = artifacts.ArtifactPaths.from_config(config)
    harm_paths = initial_artifact_paths.data_harmonization
    ingest_paths = initial_artifact_paths.data_ingestion

    planned_batches = ingest.resolve_pending_ingestion_batches(
        harmonization_paths=harm_paths,
        ingestion_paths=ingest_paths,
        target=config.data.ingestion.harmonization_run,
        rebuild=config.data.ingestion.rebuild,
    )

    if not planned_batches:
        logger = utils.Logger(name='data-ingest', enable_file_log=False)
        logger.log_sep()
        logger.log(
            'INFO',
            'No pending harmonization batches to ingest. '
            'Ingestion pool is up to date.'
        )
        logger.log_sep()
        logger.close()
        return

    for batch_record in planned_batches:
        current_artifact_paths = artifacts.ArtifactPaths.from_config(config)
        _exec_single_ingestion_batch(
            artifact_paths=current_artifact_paths,
            config=config,
            batch_record=batch_record,
        )


# ----- private helpers
def _exec_single_ingestion_batch(
    artifact_paths: artifacts.ArtifactPaths,
    config: configs.RootConfig,
    batch_record: harm_contracts.HarmonizationRunRecord,
) -> None:
    '''Execute ingestion for a single resolved harmonization batch.'''
    paths = artifact_paths.data_ingestion.init_pipeline_folders()

    logger = ingest.IngestionLogger(
        name='data-ingest',
        log_file=paths.report,
        enable_file_log=False
    )
    logger.init_summary(run_id=paths.run_id)

    try:
        logger.log_sep()
        logger.log(
            'INFO',
            f'Ingesting harmonization batch [{batch_record["run_id"]}] '
            f'({batch_record["run_uid"]}) into run [{paths.run_id}]'
        )

        policy = (
            artifacts.LifecyclePolicy.REBUILD
            if config.data.ingestion.rebuild
            else artifacts.LifecyclePolicy.BUILD_IF_MISSING
        )

        ingest.run_data_ingestion(
            artifact_paths,
            config.data.ingestion,
            policy=policy,
            logger=logger,
            harmonization_batch=batch_record,
        )

        artifacts.Controller[dict](paths.config).persist(config.as_dict)

    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log(
            'ERROR', f'Ingestion pipeline failed: {e}', exc_info=True
        )
        raise e

    finally:
        logger.update_runs_manifest(
            paths.runs_manifest,
            paths.effective_run_folder
        )
        logger.log_sep()
        logger.close()
