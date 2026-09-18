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
Data harmonization pipeline command implementation.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe.harmonize as harmonize


# ----- public functions
def exec_harmonize_data(config: configs.RootConfig) -> None:
    '''
    Execute the data-harmonize pipeline.

    Args:
        config:
            Resolved root configuration object.
    '''
    root_paths = artifacts.ArtifactPaths.from_config(config)

    paths = root_paths.data_harmonization
    paths.init()

    logger = harmonize.HarmonizationLogger(
        name='data-harmonize',
        log_file=paths.report,
        enable_file_log=False
    )
    logger.init_summary(run_id=paths.run_id)

    try:
        logger.log_sep()

        # load canonical world grid from upstream grid pipeline report
        logger.log('INFO', '[START] Loading world grid from grid report')
        context = harmonize.build_harmonization_context(
            config.data.world_grid.output_dpath
        )
        logger.set_grid_reference(context.grid_id, context.grid_fpath)
        logger.log('INFO', f'[COMPLETE] World grid loaded: {context.grid_id}')

        logger.log(
            'INFO', f'[START] Harmonizing data onto grid: {context.grid_id}'
        )
        harmonize.data_harmonization_pipeline(
            paths,
            config.data.harmonization,
            context.grid,
            logger=logger
        )
        logger.log('INFO', '[COMPLETE] Harmonization finished')

        # persist the whole config dict
        artifacts.Controller[dict](paths.config).persist(config.as_dict)

    except Exception as err:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Data harmonization failed: {err}')
        raise

    finally:
        logger.log_sep()
        logger.close()
