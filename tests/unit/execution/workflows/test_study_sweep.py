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
# pylint: disable=missing-class-docstring

'''
Unit tests for study sweep pipeline (study_sweep.py).
'''

# standard imports
import dataclasses
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs
import landseg.execution.workflows.study_sweep as sweep_pipeline


# ----- `execute_study_sweep` workflow test
def test_sweep_pipeline(tmp_path, monkeypatch):
    '''
    Given: A RootConfig instance.
    When: `execute_study_sweep` workflow executes.
    Then: Call `study.run_sweep` and return best result summary.
    '''
    @dataclasses.dataclass
    class MockStudyResult:
        best_value: float = 0.92
        best_params: dict = dataclasses.field(
            default_factory=lambda: {'lr': 0.01}
        )

    def mock_run_sweep(_builder, _config):
        return MockStudyResult()

    monkeypatch.setattr(sweep_pipeline.study, 'run_sweep', mock_run_sweep)

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = str(tmp_path)
    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    result = sweep_pipeline.execute_study_sweep(config)

    assert result == {
        'best_value': 0.92,
        'best_params': {'lr': 0.01},
    }


def test_runner_builder(tmp_path):
    '''
    Given: A RootConfig instance.
    When: `_runner_builder` is called.
    Then: Return step results path and callable wrapper.
    '''
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = str(tmp_path)
    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    step_results, wrapper = sweep_pipeline._runner_builder(config)

    assert isinstance(step_results, str)
    assert callable(wrapper)
