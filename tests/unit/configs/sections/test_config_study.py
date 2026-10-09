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
Unit tests for `landseg.configs.schema.study`.
'''

# local imports
import landseg.configs.schema.study as study
import landseg.configs.schema.study.architecture as architecture_sec
import landseg.configs.schema.study.objectives as objectives_sec
import landseg.configs.schema.study.optimization as optimization_sec
import landseg.configs.schema.study.sweep as sweep_sec


# ----- `StudyConfig` tests
def test_study_config_default_instantiation():
    '''
    Given: Default `StudyConfig` instantiation parameters.
    When: Instantiating `StudyConfig` without arguments.
    Then: Initialize sub-objects and hyperparameter search space tuples.
    '''
    cfg = study.StudyConfig()

    assert isinstance(
        cfg.optimization, optimization_sec.OptimizationSearchConfig
    )
    assert isinstance(
        cfg.architecture, architecture_sec.ArchitectureSearchConfig
    )
    assert isinstance(
        cfg.objectives, objectives_sec.ObjectivesSearchConfig
    )
    assert isinstance(
        cfg.sweep, sweep_sec.StudySweepConfig
    )

    # range definitions
    assert cfg.optimization.base.learning_rate == (1e-5, 1e-1)
    assert cfg.optimization.optimizer.weight_decay == (1e-6, 1e-2)
    assert (
        cfg.architecture.architecture.model_body
        == architecture_sec.MODEL_BODIES
    )
    assert (
        cfg.architecture.architecture.bottleneck
        == architecture_sec.BOTTLENECKS
    )


def test_study_config_custom_objective():
    '''
    Given: A custom `ArchitectureSearchSpace` search space definition.
    When: Passing custom architecture to `StudyConfig`.
    Then: Store specified model bodies and base channel choices.
    '''
    custom_arch_space = architecture_sec.ArchitectureSearchSpace(
        model_body=['unet'],
        base_channel=(32, 64, 32),
    )
    custom_arch = architecture_sec.ArchitectureSearchConfig(
        architecture=custom_arch_space
    )
    cfg = study.StudyConfig(architecture=custom_arch)

    assert cfg.architecture.architecture.model_body == ['unet']
    assert cfg.architecture.architecture.base_channel == (32, 64, 32)
