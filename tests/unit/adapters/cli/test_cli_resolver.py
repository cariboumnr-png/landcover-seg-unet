# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

# pylint: disable=protected-access

'''
Unit tests for `landseg.adapters.cli.resolver`.
'''

# third-party imports
import omegaconf
import pytest
# local imports
import landseg.adapters.cli.resolver as resolver_mod


# ----- `resolve_configs` tests
def test_resolve_configs_base(tmp_path):
    '''
    Given: A minimal `DictConfig` with valid mock file paths.
    When: Calling `resolve_configs` with
        `use_additional_settings=False`.
    Then: Resolve OmegaConf structure, set `cli_mode=True`,
        and validate.
    '''
    dev_img = tmp_path / 'dev_img.tif'
    dev_lbl = tmp_path / 'dev_lbl.tif'
    cfg_json = tmp_path / 'cfg.json'
    for f in (dev_img, dev_lbl, cfg_json):
        f.write_text('data')

    cfg_dict = omegaconf.OmegaConf.create({
        'data': {
            'harmonization': {
                'dataset_manifest': str(cfg_json),
            },
            'world_grid': {
                'mode': 'ref',
                'params': {
                    'ref_fpath': str(dev_img),
                    'crs_string': 'EPSG:32617',
                },
            },
        },

        'session': {
            'orchestration': {
                'curriculum': {
                    'single': {
                        'phases': [{'num_epochs': 5}],
                    },
                },
            },
        },
    })

    root = resolver_mod.resolve_configs(
        config=cfg_dict,
        use_additional_settings=False,
    )

    assert root.execution.cli_mode is True
    assert root.session.orchestration.single_phase.num_epochs == 5


def test_resolve_configs_missing_dev_file():
    '''
    Given: Non-existent dev config path in `execution.dev_cfg`.
    When: `resolve_configs` executes with
        `use_additional_settings=True`.
    Then: Raise a FileNotFoundError indicating missing dev config file.
    '''
    cfg_dict = omegaconf.OmegaConf.create({
        'execution': {'dev_cfg': '/path/missing_dev.yaml'},
    })

    with pytest.raises(FileNotFoundError, match='configuration file not found'):
        resolver_mod.resolve_configs(
            config=cfg_dict,
            use_additional_settings=True,
        )


def test_resolve_configs_overfit_recipe_applied():
    '''
    Given: A Hydra configuration targeting 'diagnose-overfit'.
    When: Calling `resolve_configs` with task command.
    Then: Automatically discover and apply the overfit recipe overrides.
    '''
    cfg_dict = omegaconf.OmegaConf.create({
        'command': 'diagnose-overfit',
    })

    root = resolver_mod.resolve_configs(
        config=cfg_dict,
        use_additional_settings=True,
    )

    assert root.command == 'diagnose-overfit'
    assert root.session.engine_exec.use_amp is False
    assert root.session.engine_optim.lr == 1e-3
    assert root.session.engine_tasks.loss_configs.focal.gamma == 0.0


def test_resolve_configs_command_recipe_applied():
    '''
    Given: A Hydra configuration targeting 'data-harmonize'.
    When: Calling `resolve_configs` with command.
    Then: Load configs directly from `data_harmonize.yaml`.
    '''
    cfg_dict = omegaconf.OmegaConf.create({
        'command': 'data-harmonize',
    })

    root = resolver_mod.resolve_configs(
        config=cfg_dict,
        use_additional_settings=True,
    )

    assert root.command == 'data-harmonize'
    assert root.data.world_grid.output_dpath == (
        './experiment/artifacts/world_grids'
    )
    assert root.data.harmonization.output_dpath == (
        './experiment/artifacts/harmonized_data'
    )
    assert root.data.harmonization.resampling_continuous == 'bilinear'
    # verify recipe values applied correctly
    assert root.data.preparation.partition.val_ratio != 0.20


def test_resolve_configs_missing_recipe():
    '''
    Given: A Hydra configuration targeting an unknown command.
    When: Calling `resolve_configs` with
        `use_additional_settings=True`.
    Then: Raise a FileNotFoundError indicating missing tracked recipe.
    '''
    cfg_dict = omegaconf.OmegaConf.create({
        'command': 'non-existent-cmd',
    })

    with pytest.raises(
        FileNotFoundError, match='Tracked recipe not found for command'
    ):
        resolver_mod.resolve_configs(
            config=cfg_dict,
            use_additional_settings=True,
        )
