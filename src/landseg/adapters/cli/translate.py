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

def translate_user_config(raw: omegaconf.DictConfig) -> omegaconf.DictConfig:
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

    if 'data-ingest' in raw:
        _translate_data_ingest(raw['data-ingest'], translated)

    if 'data-prepare' in raw:
        _translate_data_prepare(raw['data-prepare'], translated)

    if 'model-train' in raw:
        _translate_model_train(raw['model-train'], translated)

    return omegaconf.OmegaConf.create(translated)

def _translate_data_ingest(
    fdn: omegaconf.DictConfig,
    translated: dict
) -> None:
    '''Map data-ingest settings to foundation fields.'''

    mapping = {
        'grid_mode': ['foundation.grid.mode'],
        'grid_crs': ['foundation.grid.crs'],
        'grid_extent_path': ['foundation.grid.extent.filepath'],
        'tile_size': [
            'foundation.grid.tile_specs.size_row',
            'foundation.grid.tile_specs.size_col'
        ],
        'tile_overlap': [
            'foundation.grid.tile_specs.overlap_row',
            'foundation.grid.tile_specs.overlap_col'
        ],
        'domain_files': ['foundation.domains.files'],
        'domain_ids_name': ['dataspecs.domain_ids_name'],
        'domain_vec_name': ['dataspecs.domain_vec_name'],
        'dataset_name': ['foundation.datablocks.name'],
        'dev_image': ['foundation.datablocks.filepaths.dev_image'],
        'dev_label': ['foundation.datablocks.filepaths.dev_label'],
        'test_image': ['foundation.datablocks.filepaths.test_image'],
        'test_label': ['foundation.datablocks.filepaths.test_label'],
        'dataset_config': ['foundation.datablocks.filepaths.config'],
    }
    _apply_mapping(fdn, translated, mapping)

def _translate_data_prepare(
    tf: omegaconf.DictConfig,
    translated: dict
) -> None:
    '''Map data-prepare settings to transform fields.'''

    mapping = {
        'val_ratio': ['transform.partition.val_ratio'],
        'test_ratio': ['transform.partition.test_ratio'],
        'target_head': ['transform.catalog.focal_target'],
        'reward_classes': ['transform.scoring.reward'],
    }
    _apply_mapping(tf, translated, mapping)

def _translate_model_train(
    rt: omegaconf.DictConfig,
    translated: dict
) -> None:
    '''Map model-train settings to models and session fields.'''
    
    mapping = {
        'exp_root': ['execution.exp_root'],
        'model_body': ['models.model_body'],
        'bottleneck': ['models.bottleneck'],
        'conditioners': ['models.conditioners'],
        'patch_size': ['session.data_loader.patch_size'],
        'batch_size': ['session.data_loader.batch_size'],
    }
    _apply_mapping(rt, translated, mapping)

    if 'epochs' in rt:
        _set_paths(
            translated,
            ['session.orchestration.curriculum.single.phases'],
            [{'name': 'demo_train', 'num_epochs': rt['epochs']}]
        )

def _apply_mapping(
    src: omegaconf.DictConfig,
    translated: dict,
    mapping: dict[str, list[str]]
) -> None:
    '''Apply mapping from source DictConfig to translated dictionary.'''
    for src_key, dest_paths in mapping.items():
        if src_key in src:
            _set_paths(translated, dest_paths, src[src_key])

def _set_paths(
    translated: dict,
    paths: list[str],
    val: typing.Any
) -> None:
    '''Set value at multiple target paths.'''

    def _set_path(
        d: dict,
        path: str,
        val: typing.Any
    ) -> None:
        parts = path.split('.')
        curr = d
        for part in parts[:-1]:
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        curr[parts[-1]] = val

    for path in paths:
        _set_path(translated, path, val)
