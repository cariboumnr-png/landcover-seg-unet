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


'''
Unit tests for `landseg.configs.schema.study`.
'''

# local imports
import landseg.configs.schema.study as study
import landseg.configs.schema.study.architecture as architecture
import landseg.configs.schema.study.objectives as objectives
import landseg.configs.schema.study.optimization as optimization
import landseg.configs.schema.study.sweep as sweep


# ----- `StudyConfig` tests
def test_study_config_default_instantiation():
    '''
    Given: Default `StudyConfig` instantiation parameters.
    When: Instantiating `StudyConfig` without arguments.
    Then: Initialize sub-objects and hyperparameter search space tuples.
    '''
    cfg = study.StudyConfig()

    assert isinstance(cfg.optimization, optimization.OptimizationSearchConfig)
    assert isinstance(cfg.architecture, architecture.ArchitectureSearchConfig)
    assert isinstance(cfg.objectives, objectives.ObjectivesSearchConfig)
    assert isinstance(cfg.sweep, sweep.StudySweepConfig)

    # range definitions
    assert cfg.optimization.base.learning_rate == (1e-5, 1e-1)
    assert cfg.optimization.optimizer.weight_decay == (1e-6, 1e-2)
    assert cfg.architecture.architecture.model_body == architecture.MODEL_BODIES
    assert cfg.architecture.architecture.bottleneck == architecture.BOTTLENECKS


def test_study_config_custom_objective():
    '''
    Given: A custom `ArchitectureSearchSpace` search space definition.
    When: Passing custom architecture to `StudyConfig`.
    Then: Store specified model bodies and base channel choices.
    '''
    custom_arch_space = architecture.Architecture(
        model_body=['unet'],
        base_channel=(32, 64, 32),
    )
    custom_arch = architecture.ArchitectureSearchConfig(
        architecture=custom_arch_space
    )
    cfg = study.StudyConfig(architecture=custom_arch)

    assert cfg.architecture.architecture.model_body == ['unet']
    assert cfg.architecture.architecture.base_channel == (32, 64, 32)
