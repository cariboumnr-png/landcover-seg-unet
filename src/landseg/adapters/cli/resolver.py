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
Hydra configs resolver
'''

# standard imports
import os
import pathlib
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs

# register omegaconf.OmegaConf.resolvers
omegaconf.OmegaConf.register_new_resolver('concat', lambda x, y: x + y)


def resolve_configs(
    config: omegaconf.DictConfig,
    use_additional_settings: bool = True
) -> configs.RootConfig:
    '''Resolve configs from difference sources'''

    # list of configs to resolve
    config_list: list = []

    # add schema - dataclass scaffolding with default dummy values
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    config_list.append(schema)

    # add Hydra-composed config
    # - as the single source of truth in API mode
    # - might override by additional settings (*yaml) below in CLI mode
    config_list.append(config)

    # resolve command recipe
    cmd = omegaconf.OmegaConf.select(config, 'command', default=None)
    if cmd and cmd != 'default' and use_additional_settings:
        recipe_filename = f"{cmd.replace('-', '_')}.yaml"
        root_dpath = pathlib.Path(__file__).resolve().parents[4]
        recipe_path = root_dpath / 'configs' / recipe_filename
        if not recipe_path.exists():
            raise FileNotFoundError(
                f'Tracked recipe not found for command "{cmd}" at: '
                f'{recipe_path}'
            )
        recipe_cfg = omegaconf.OmegaConf.load(recipe_path)
        assert isinstance(recipe_cfg, omegaconf.DictConfig)
        config_list.append(recipe_cfg)

    # add dev settings (optional and untracked)
    dev = omegaconf.OmegaConf.select(config, 'execution.dev_cfg', default=None)
    if dev and use_additional_settings:
        if os.path.exists(dev):
            dev_cfg = omegaconf.OmegaConf.load(dev)
            assert isinstance(dev_cfg, omegaconf.DictConfig)
            config_list.append(dev_cfg)
        else:
            raise FileNotFoundError(
                f'Developer override configuration file not found at: {dev}'
            )

    # merging configs in order (last wins)
    # dev -> recipe -> hydra defaults -> schema defaults
    with omegaconf.open_dict(config):
        merged = omegaconf.OmegaConf.merge(*config_list)
    cfg = typing.cast(omegaconf.DictConfig, merged)
    omegaconf.OmegaConf.resolve(cfg)

    # construct and cast config dataclass
    root = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(cfg))
    root.execution.cli_mode = True

    return root # note here configs are not yet validated
