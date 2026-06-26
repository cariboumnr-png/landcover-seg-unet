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

# register omegaconf.OmegaConf.resolvers
omegaconf.OmegaConf.register_new_resolver("concat", lambda x, y: x + y)

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

    # add user settings - this contains the essesion I/O to start the program
    # resolve absolute path to the user settings at root/configs
    # root/src/landseg/adapters/cli/resolver.py -> the 5th parent (parents[4])
    user = pathlib.Path(__file__).resolve().parents[4]/'configs'/'user.yaml'
    if os.path.exists(user) and use_additional_settings:
        user_settings = omegaconf.OmegaConf.load(user)
        assert isinstance(user_settings, omegaconf.DictConfig)
        translated_settings = _translate_user_config(user_settings)
        config_list.append(translated_settings)

    # add dev settings (optional and untracked)
    dev = omegaconf.OmegaConf.select(config, 'execution.dev_settings', default=None)
    if dev and os.path.exists(dev) and use_additional_settings:
        dev_settings = omegaconf.OmegaConf.load(dev)
        assert isinstance(dev_settings, omegaconf.DictConfig)
        config_list.append(dev_settings)

    # merging configs in order (last wins)
    # dev -> user -> hydra defaults -> schema defaults
    with omegaconf.open_dict(config):
        merged = omegaconf.OmegaConf.merge(*config_list)
    cfg = typing.cast(omegaconf.DictConfig, merged)
    omegaconf.OmegaConf.resolve(cfg)

    # construct and cast config dataclass
    root = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(cfg))

    # final validation checks before returning
    root.validate_all()
    return root


def _translate_user_config(
    user_raw: omegaconf.DictConfig
) -> omegaconf.DictConfig:
    '''Translate user.yaml DictConfig into RootConfig overrides.'''
    translated: dict[str, typing.Any] = {
        'execution': {},
        'foundation': {
            'grid': {
                'extent': {},
                'tile_specs': {},
            },
            'domains': {},
            'datablocks': {
                'filepaths': {},
            },
        },
        'transform': {
            'catalog': {},
            'partition': {},
            'scoring': {},
        },
        'dataspecs': {},
        'models': {},
        'session': {
            'data_loader': {},
            'orchestration': {
                'curriculum': {
                    'single': {
                        'phases': [{}],
                    },
                },
            },
        },
    }

    # Helper function to safely set nested keys
    def _set_path(d: dict, path: str, val: typing.Any):
        parts = path.split('.')
        curr = d
        for part in parts[:-1]:
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        curr[parts[-1]] = val

    # Map data-ingest settings
    if 'data-ingest' in user_raw:
        fdn = user_raw['data-ingest']
        # grid
        if 'grid_mode' in fdn:
            _set_path(
                translated, 
                'foundation.grid.mode', 
                fdn['grid_mode']
            )
        if 'grid_crs' in fdn:
            _set_path(
                translated, 
                'foundation.grid.crs', 
                fdn['grid_crs']
            )
        if 'grid_extent_path' in fdn:
            _set_path(
                translated,
                'foundation.grid.extent.filepath',
                fdn['grid_extent_path']
            )
        if 'tile_size' in fdn:
            _set_path(
                translated,
                'foundation.grid.tile_specs.size_row',
                fdn['tile_size']
            )
            _set_path(
                translated,
                'foundation.grid.tile_specs.size_col',
                fdn['tile_size']
            )
        if 'tile_overlap' in fdn:
            _set_path(
                translated,
                'foundation.grid.tile_specs.overlap_row',
                fdn['tile_overlap']
            )
            _set_path(
                translated,
                'foundation.grid.tile_specs.overlap_col',
                fdn['tile_overlap']
            )

        # domains
        if 'domain_files' in fdn:
            _set_path(
                translated, 
                'foundation.domains.files', 
                fdn['domain_files']
            )
        if 'domain_ids_name' in fdn:
            _set_path(
                translated,
                'dataspecs.domain_ids_name',
                fdn['domain_ids_name']
            )
        if 'domain_vec_name' in fdn:
            _set_path(
                translated,
                'dataspecs.domain_vec_name',
                fdn['domain_vec_name']
            )

        # datablocks
        if 'dataset_name' in fdn:
            _set_path(
                translated,
                'foundation.datablocks.name',
                fdn['dataset_name']
            )
        if 'dev_image' in fdn:
            _set_path(
                translated,
                'foundation.datablocks.filepaths.dev_image',
                fdn['dev_image']
            )
        if 'dev_label' in fdn:
            _set_path(
                translated,
                'foundation.datablocks.filepaths.dev_label',
                fdn['dev_label']
            )
        if 'test_image' in fdn:
            _set_path(
                translated,
                'foundation.datablocks.filepaths.test_image',
                fdn['test_image']
            )
        if 'test_label' in fdn:
            _set_path(
                translated,
                'foundation.datablocks.filepaths.test_label',
                fdn['test_label']
            )
        if 'dataset_config' in fdn:
            _set_path(
                translated,
                'foundation.datablocks.filepaths.config',
                fdn['dataset_config']
            )

    # Map data-prepare settings
    if 'data-prepare' in user_raw:
        tf = user_raw['data-prepare']
        if 'val_ratio' in tf:
            _set_path(
                translated,
                'transform.partition.val_ratio',
                tf['val_ratio']
            )
        if 'test_ratio' in tf:
            _set_path(
                translated,
                'transform.partition.test_ratio',
                tf['test_ratio']
            )
        if 'target_head' in tf:
            _set_path(
                translated,
                'transform.catalog.focal_target',
                tf['target_head']
            )
        if 'reward_classes' in tf:
            _set_path(
                translated,
                'transform.scoring.reward',
                tf['reward_classes']
            )

    # Map model-train settings
    if 'model-train' in user_raw:
        rt = user_raw['model-train']
        if 'exp_root' in rt:
            _set_path(
                translated, 
                'execution.exp_root',
                rt['exp_root']
            )
        if 'model_body' in rt:
            _set_path(
                translated, 
                'models.model_body',
                rt['model_body']
            )
        if 'bottleneck' in rt:
            _set_path(
                translated, 
                'models.bottleneck',
                rt['bottleneck']
            )
        if 'conditioners' in rt:
            _set_path(
                translated, 
                'models.conditioners',
                rt['conditioners']
            )
        if 'patch_size' in rt:
            _set_path(
                translated,
                'session.data_loader.patch_size',
                rt['patch_size']
            )
        if 'batch_size' in rt:
            _set_path(
                translated,
                'session.data_loader.batch_size',
                rt['batch_size']
            )
        if 'epochs' in rt:
            _set_path(
                translated,
                'session.orchestration.curriculum.single.phases',
                [{'name': 'demo_train', 'num_epochs': rt['epochs']}]
            )

    return omegaconf.OmegaConf.create(translated)
