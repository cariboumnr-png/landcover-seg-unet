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

# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.geopipe.ingest as ingest
import landseg.utils as utils


# ----- public functions
def execute_batch_ingest(config: configs.RootConfig) -> None:
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
    artifact_paths = artifacts.ArtifactPaths.from_config(config)

    planned_batches = ingest.resolve_pending_ingestion_batches(
        artifact_paths.data_harmonization.runs_manifest,
        artifact_paths.data_ingestion.runs_manifest,
        target=config.data.ingestion.harmonization_run,
        rebuild=config.data.ingestion.rebuild,
    )

    if not planned_batches:
        logger = utils.Logger(name='data-ingest', enable_file_log=False)
        logger.log_sep()
        logger.info('No pending batches to ingest, ingestion pool is up to date')
        logger.log_sep()
        logger.close()
        return

    for batch_record in planned_batches:
        pp = pipelines.DataIngestion(config)
        pp.run(batch_record)
