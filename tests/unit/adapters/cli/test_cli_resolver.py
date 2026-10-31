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

'''Unit tests for `landseg.adapters.cli.resolver`.'''

# standard imports
import pathlib
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
            'harmonization': {'dataset_manifest': str(cfg_json)},
            'world_grid': {'mode': 'ref', 'ref_fpath': str(dev_img)},
        },

        'session': {
            'orchestration': {
                'curriculum': {'single': {'phases': [{'num_epochs': 5}]}},
            },
        },
    })

    root = resolver_mod.resolve_configs(config=cfg_dict)

    assert root.session.orchestration.single_phase.num_epochs == 5

def test_resolve_configs_overfit_recipe_applied():
    '''
    Given: A Hydra configuration targeting 'diagnose-overfit'.
    When: Calling `resolve_configs` with task command.
    Then: Automatically discover and apply the overfit recipe overrides.
    '''
    cfg_dict = omegaconf.OmegaConf.create({'command': 'diagnose-overfit'})

    root = resolver_mod.resolve_configs(config=cfg_dict)

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
    cfg_dict = omegaconf.OmegaConf.create({'command': 'data-harmonize'})

    root = resolver_mod.resolve_configs(config=cfg_dict)

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
    cfg_dict = omegaconf.OmegaConf.create({'command': 'non-existent-cmd'})

    with pytest.raises(
        FileNotFoundError, match='Tracked recipe not found for command'
    ):
        resolver_mod.resolve_configs(config=cfg_dict)


def test_resolve_configs_custom_recipe_dir(tmp_path):
    '''
    Given: A custom recipe directory passed via `execution.recipe_dir`.
    When: Calling `resolve_configs` with the custom recipe directory.
    Then: Load overrides from custom recipe instead of default.
    '''
    mock_ref = tmp_path / 'mock_ref.tif'
    mock_ref.write_text('dummy')

    custom_recipe = tmp_path / 'world_grid.yaml'
    custom_recipe.write_text(
        'data:\n'
        '  world_grid:\n'
        f'    ref_fpath: {mock_ref.as_posix()}\n'
        '    tile_size: [512, 512]\n'
        '    tile_stride: [256, 256]\n'
    )

    cfg_dict = omegaconf.OmegaConf.create({
        'command': 'world-grid',
        'execution': {'recipe_dir': str(tmp_path)},
    })

    root = resolver_mod.resolve_configs(config=cfg_dict)

    assert root.data.world_grid.tile_size == [512, 512]
    assert root.data.world_grid.tile_stride == [256, 256]


def test_resolve_configs_custom_recipe_dir_missing(tmp_path):
    '''
    Given: An `execution.recipe_dir` missing the required recipe file.
    When: Calling `resolve_configs`.
    Then: Raise FileNotFoundError indicating recipe is missing.
    '''
    cfg_dict = omegaconf.OmegaConf.create({
        'command': 'world-grid',
        'execution': {'recipe_dir': str(tmp_path / 'non_existent_dir')},
    })

    with pytest.raises(
        FileNotFoundError, match='Tracked recipe not found for command'
    ):
        resolver_mod.resolve_configs(config=cfg_dict)


@pytest.mark.parametrize(
    'cmd',
    [
        'world-grid',
        'data-harmonize',
        'data-ingest',
        'data-prepare',
        'diagnose-overfit',
        'model-evaluate',
        'model-train',
        'study-analysis',
        'study-sweep',
        'e2e-intake',
        'e2e-experiment',
    ],
)
def test_all_tracked_recipes_resolve_and_validate(cmd: str):
    '''
    Given: Each tracked command in sectional and end-to-end recipes.
    When: Calling `resolve_configs` for the command.
    Then: The configuration resolves cleanly and validates without error.
    '''
    ckpt_dir = pathlib.Path('./experiment/results/checkpoints')
    ckpt_file = ckpt_dir / 'best_model.pth'
    created = False
    if cmd == 'model-evaluate' and not ckpt_file.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file.write_text('dummy')
        created = True
    try:
        cfg_dict = omegaconf.OmegaConf.create({'command': cmd})
        root = resolver_mod.resolve_configs(config=cfg_dict)

        assert root.command == cmd
    finally:
        if created and ckpt_file.exists():
            ckpt_file.unlink()
