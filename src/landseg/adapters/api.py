# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
Programmati API entry
'''

# third-party imports
import hydra
import hydra.core.global_hydra
import omegaconf
# local imports
import landseg.execution as execution
import landseg.utils as utils

#
def get_default_config(pipeline: str) -> omegaconf.DictConfig:
    '''
    Return a base Hydra-composed configuration for the given pipeline.

    Intended as a convenience helper for programmatic and notebook usage.
    Users are expected to further modify the returned config interactively.
    '''

    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    with hydra.initialize(config_path='./configs'):
        cfg = hydra.compose('config', overrides=[f'pipeline={pipeline}'])
    return cfg

#
def run(config: omegaconf.DictConfig):
    '''
    Run the selected pipeline using a pre-composed Hydra configuration.

    This API is intended for non-CLI use cases (e.g. notebooks, tests,
    programmatic runners). It mirrors the CLI execution flow but does not
    terminate the interpreter on failure.
    '''

    logger = utils.Logger('api', './api.log')

    try:
        root_config = execution.resolve_configs(config)
        return execution.execute_pipeline(root_config)
    except KeyboardInterrupt:
        logger.log('INFO', 'Execution interrupted')
        raise
    except Exception:
        logger.log(
            'CRITICAL',
            'Unhandled exception occurred during API execution',
            exc_info=True,
        )
        raise
