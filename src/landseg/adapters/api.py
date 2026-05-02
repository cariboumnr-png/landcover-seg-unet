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

# standard imports
import pathlib
import typing
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

    # resolve absolute path to the config/ folder
    cfg_dir = str(pathlib.Path(__file__).resolve().parents[1] / 'configs')

    # clear global hydra state
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    # compose config from existing configs from config/
    with hydra.initialize_config_dir(config_dir=cfg_dir,version_base='1.3'):
        # here we apply preconfigured pipeline overrides
        cfg = hydra.compose('config', overrides=[f'pipeline={pipeline}'])

    # return
    return cfg

def add_custom_config(
    cfg: omegaconf.DictConfig,
    custom_config: dict[str, typing.Any] | None = None
) -> omegaconf.DictConfig:
    '''Add custom config dictionary to root config.'''

    # early exit - no ops
    if not custom_config:
        return cfg
    # create omega config dict from custom dict
    custom = omegaconf.OmegaConf.create(custom_config)
    # naively merge and we rely on downstream resolver for truthiness
    with omegaconf.open_dict(cfg):
        merged = omegaconf.OmegaConf.merge(cfg, custom)
    assert isinstance(merged, omegaconf.DictConfig) # typing sanity
    return merged

def run(config: omegaconf.DictConfig):
    '''
    Run the selected pipeline using a pre-composed Hydra configuration.

    This API is intended for non-CLI use cases (e.g. notebooks, tests,
    programmatic runners). It mirrors the CLI execution flow but does not
    terminate the interpreter on failure.
    '''

    logger = utils.Logger('api', './api.log')

    try:
        # here we rely on external caller to provide the only source of truth
        cfg = execution.resolve_configs(config, use_additional_settings=False)
        return execution.execute_pipeline(cfg)
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
