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

# aliases
omega = omegaconf.OmegaConf

# register omega resolvers
omega.register_new_resolver("concat", lambda x, y: x + y)

def resolve_configs(
    config: omegaconf.DictConfig,
    use_additional_settings: bool = True
) -> configs.RootConfig:
    '''Resolve configs from difference sources'''

    # list of configs to resolve
    config_list: list = []

    # add schema - dataclass scaffolding with default dummy values
    schema = omega.structured(configs.RootConfig)
    config_list.append(schema)

    # add Hydra-composed config
    # - as the single source of truth in API mode
    # - might override by additional settings (*yaml) below in CLI mode
    config_list.append(config)

    # add user settings - this should contain the complete config values
    # resolve absolute path to the user settings at root
    # root/src/landseg/execution/resolver.py -> the 4th parent
    user = pathlib.Path(__file__).resolve().parents[3] / 'settings.yaml'
    if os.path.exists(user) and use_additional_settings:
        user_settings = omega.load(user)
        assert isinstance(user_settings, omegaconf.DictConfig)
        config_list.append(user_settings)

    # add dev settings (optional and untracked)
    dev = omega.select(config, 'execution.dev_settings', default=None)
    if dev and os.path.exists(dev) and use_additional_settings:
        dev_settings = omega.load(dev)
        assert isinstance(dev_settings, omegaconf.DictConfig)
        config_list.append(dev_settings)

    # merging configs in order (last wins)
    # dev -> user -> hydra defaults -> schema defaults
    with omegaconf.open_dict(config):
        merged = omega.merge(*config_list)
    cfg = typing.cast(omegaconf.DictConfig, merged)
    omega.resolve(cfg)

    # construct and cast config dataclass
    root = typing.cast(configs.RootConfig, omega.to_object(cfg))

    # final validation checks before returning
    root.validate_all()
    return root
