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
Unit tests for study analysis pipeline (study_analysis.py).
'''

# standard imports
import json
import os
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs
import landseg.execution.workflows.study_analysis as analysis_pipeline


# ----- `analyze` pipeline test
def test_analyze_pipeline(tmp_path, monkeypatch):
    '''
    Given: A RootConfig instance with study_sweep settings.
    When: `analyze` is called.
    Then: Rank completed trials and persist analysis JSON artifact.
    '''
    mock_ranked = [{'trial_id': 1, 'value': 0.95, 'params': {'lr': 0.001}}]

    def mock_rank_trials(
        study_name: str,
        storage: str,
        top_k: int = 5,
        ascending: bool = False,
    ):
        _ = study_name, storage, top_k, ascending
        return mock_ranked

    monkeypatch.setattr(
        analysis_pipeline.study, 'rank_trials', mock_rank_trials
    )

    exp_root = str(tmp_path / 'exp')
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = exp_root
    schema.pipeline.study_sweep.study_name = 'test_study'
    schema.pipeline.study_sweep.storage = 'sqlite:///test.db'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    analysis_pipeline.analyze(config)

    analysis_fpath = f'{exp_root}/analysis/test_study.json'
    assert os.path.exists(analysis_fpath)
    with open(analysis_fpath, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    assert saved_data == mock_ranked
