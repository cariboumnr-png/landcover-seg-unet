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
import landseg.geopipe.ingest as ingest

# aliases
ConfigController = artifacts.Controller[dict]


def exec_ingest_data(config: configs.RootConfig) -> None:
    '''Run the ingestion pipeline.'''
    artifact_paths = artifacts.ArtifactPaths.from_config(config)
    paths = artifact_paths.data_ingestion

    logger = ingest.IngestionLogger(
        name='data-ingest',
        log_file=paths.report,
        enable_file_log=False
    )
    logger.init_summary(run_id='ingest')

    try:
        logger.log_sep()

        # resolve lifecycle policy dynamically
        policy = (
            artifacts.LifecyclePolicy.REBUILD
            if config.data.ingestion.rebuild
            else artifacts.LifecyclePolicy.BUILD_IF_MISSING
        )

        # run pipeline
        ingest.run_data_ingestion(
            artifact_paths,
            config.data.ingestion,
            policy=policy,
            logger=logger
        )

        # persist the whole config dict
        artifacts.Controller[dict](paths.config).persist(config.as_dict)

    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Ingestion pipeline failed: {e}', exc_info=True)
        raise e

    finally:
        logger.log_sep()
        logger.close()
