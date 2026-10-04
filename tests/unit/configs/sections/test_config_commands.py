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
Unit tests for `landseg.configs.schema.sections.commands`.
'''

# local imports
import landseg.configs.schema.sections.commands as commands


# ----- `CommandConfig` tests
def test_command_config_defaults():
    '''
    Given: Default instantiation parameters for `CommandConfig`.
    When: Instantiating `CommandConfig` without arguments.
    Then: Initialize default command name and sweep trial counts.
    '''
    cfg = commands.CommandConfig()
    assert cfg.name == 'default'
    assert isinstance(cfg.model_train, commands._TrainModel)
    assert isinstance(cfg.model_evaluate, commands._EvaluateModel)
    assert isinstance(cfg.study_sweep, commands._StudySweep)

    assert cfg.model_evaluate.checkpoint is None
    assert cfg.model_evaluate.split == 'test'
    assert cfg.model_evaluate.export_previews is False

    assert cfg.study_sweep.study_name == 'study_test'
    assert cfg.study_sweep.preset_name == 'base'
    assert cfg.study_sweep.n_trials == 50


def test_command_config_custom_initialization():
    '''
    Given: Custom `_EvaluateModel` and `_StudySweep` sub-configs.
    When: Instantiating `CommandConfig` with custom sub-configurations.
    Then: Store specified sub-configurations on attributes.
    '''
    eval_cfg = commands._EvaluateModel(
        checkpoint='/path/to/ckpt.pt',
        split='val',
        export_previews=True,
    )
    sweep_cfg = commands._StudySweep(
        study_name='custom_study',
        n_trials=100,
    )
    cfg = commands.CommandConfig(
        name='experiment_1',
        model_evaluate=eval_cfg,
        study_sweep=sweep_cfg,
    )

    assert cfg.name == 'experiment_1'
    assert cfg.model_evaluate.checkpoint == '/path/to/ckpt.pt'
    assert cfg.model_evaluate.split == 'val'
    assert cfg.model_evaluate.export_previews is True
    assert cfg.study_sweep.study_name == 'custom_study'
    assert cfg.study_sweep.n_trials == 100
