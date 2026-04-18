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

# pylint: disable=no-value-for-parameter

'''
CLI entry that resolves Hydra configs, selects a pipeline, dispatches the
mapped command, and manages logging and error handling.
'''

# standard imports
import os
import sys
import typing
# third-party imports
import hydra
import hydra.core.hydra_config
import hydra.utils
import omegaconf
# local imports
import landseg.cli.pipelines as pipelines
import landseg.configs as configs
import landseg.utils as utils

command_registry = {
    'default': pipelines.default_action,
    'ingest-data': pipelines.ingest,
    'prepare-data': pipelines.prepare,
    'train-model': pipelines.train,
    'train-overfit': pipelines.overfit
}

# main process
@hydra.main('pkg://landseg/configs', 'config', version_base='1.3')
def main(config: omegaconf.DictConfig) -> None:
    '''Run the selected CLI pipeline with resolved configuration.'''

    # resolve config
    root_config = _resolve_configs(config)

    # master logger
    log_fpath = os.path.join(root_config.execution.exp_root, 'cli.log')
    logger = utils.Logger('cli', log_fpath)

    # run specified mode with exceptions handling
    try:
        # get running pipeline
        get = hydra.core.hydra_config.HydraConfig.get()
        pipe = get.runtime.choices['pipeline']
        # get command from pipeline
        command = command_registry[pipe]
        # run command
        logger.log('INFO', f'Runing pipeline: {pipe} start')
        command(root_config)
        logger.log('INFO', f'Runing pipeline: {pipe} finish')
    # manual keyboard interruption
    except KeyboardInterrupt:
        logger.log('INFO', '\nExperiment manually interrupted, exiting...')
        sys.exit(130)
    # capture others and log
    except Exception: # pylint: disable=broad-exception-caught
        logger.log('CRITICAL', 'Unhandled exception occurred', exc_info=True)
        sys.exit(1)

def _resolve_configs(config: omegaconf.DictConfig) -> configs.RootConfig:
    '''Resolve configs from difference sources'''

        # list of configs to resolve
    config_list: list = []

    # add schema
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    config_list.append(schema)

    # add Hydra-composed runtime config
    config_list.append(config)

    # get user settings at root (with safer CWD fetching)
    user = os.path.join(hydra.utils.get_original_cwd(), 'settings.yaml')
    if os.path.exists(user):
        user_settings = omegaconf.OmegaConf.load(user)
        assert isinstance(user_settings, omegaconf.DictConfig)
        config_list.append(user_settings)

    # get dev settings (untracked)
    dev = config['execution'].get('dev_settings')
    if dev and os.path.exists(dev):
        dev_settings = omegaconf.OmegaConf.load(dev)
        assert isinstance(dev_settings, omegaconf.DictConfig)
        config_list.append(dev_settings)

    # merging overrides resolve
    with omegaconf.open_dict(config):
        merged = omegaconf.OmegaConf.merge(*config_list)
    cfg = typing.cast(omegaconf.DictConfig, merged)
    omegaconf.OmegaConf.resolve(cfg)

    # construct and cast config dataclass
    root = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(cfg))

    # final validation checks before returning
    root.validate_all()
    return root

if __name__ == '__main__':
    main()
