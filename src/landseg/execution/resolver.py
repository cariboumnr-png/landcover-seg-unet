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
import typing
# third-party imports
import hydra
import hydra.utils
import omegaconf
# local imports
import landseg.configs as configs

# register omega resolvers
omegaconf.OmegaConf.register_new_resolver("concat", lambda x, y: x + y)

def resolve_configs(config: omegaconf.DictConfig) -> configs.RootConfig:
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
