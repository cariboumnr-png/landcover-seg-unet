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
User settings translation helper.
'''

# standard imports
import typing
# third-party imports
import omegaconf

def translate_user_config(
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

    if 'data-ingest' in user_raw:
        _translate_data_ingest(user_raw['data-ingest'], translated)

    if 'data-prepare' in user_raw:
        _translate_data_prepare(user_raw['data-prepare'], translated)

    if 'model-train' in user_raw:
        _translate_model_train(user_raw['model-train'], translated)

    return omegaconf.OmegaConf.create(translated)

def _set_path(
    d: dict,
    path: str,
    val: typing.Any
) -> None:
    '''Heper to set nested key-value pairs.'''
    parts = path.split('.')
    curr = d
    for part in parts[:-1]:
        if part not in curr:
            curr[part] = {}
        curr = curr[part]
    curr[parts[-1]] = val


def _translate_data_ingest(
    fdn: omegaconf.DictConfig,
    translated: dict
) -> None:
    '''Map data-ingest settings to foundation fields.'''
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


def _translate_data_prepare(
    tf: omegaconf.DictConfig,
    translated: dict
) -> None:
    '''Map data-prepare settings to transform fields.'''
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


def _translate_model_train(
    rt: omegaconf.DictConfig,
    translated: dict
) -> None:
    '''Map model-train settings to models and session fields.'''
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
