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
    Then: Initialize default command name and model_train sub-config.
    '''
    cfg = commands.CommandConfig()
    assert cfg.name == 'default'
    assert isinstance(cfg.model_train, commands._TrainModel)


def test_command_config_custom_initialization():
    '''
    Given: Custom `_TrainModel` sub-config.
    When: Instantiating `CommandConfig` with custom sub-configuration.
    Then: Store specified sub-configuration on attribute.
    '''
    train_cfg = commands._TrainModel()
    cfg = commands.CommandConfig(
        name='experiment_1',
        model_train=train_cfg,
    )

    assert cfg.name == 'experiment_1'
    assert cfg.model_train is train_cfg

