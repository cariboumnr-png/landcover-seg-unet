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
import landseg.configs as configs
import landseg.geopipe.prepare as geopipe_prepare


def exec_prepare_data(config: configs.RootConfig):
    '''Run the preparation pipeline for an experiment.'''
    artifact_paths = artifacts.ArtifactPaths.from_config(config)
    paths = artifact_paths.data_preparation.init_pipeline_folders()

    logger = geopipe_prepare.PreparationLogger(
        name='data-prep',
        log_file=paths.report,
        enable_file_log=False
    )
    logger.init_summary(run_id='prepare')

    try:
        logger.log_sep()

        # resolve lifecycle policy dynamically
        policy = (
            artifacts.LifecyclePolicy.REBUILD
            if config.data.preparation.rebuild
            else artifacts.LifecyclePolicy.BUILD_IF_MISSING
        )

        # run pipeline
        geopipe_prepare.run_data_preparation(
            artifact_paths,
            config.data.preparation,
            config.data.world_grid.tile_specs_tuple,
            policy=policy,
            logger=logger,
        )

        # persist the whole config dict
        artifacts.Controller[dict](paths.config).persist(config.as_dict)

    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Preparation pipeline failed: {e}', exc_info=True)
        raise e
    finally:
        logger.log_sep()
        logger.close()
