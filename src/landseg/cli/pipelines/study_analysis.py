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
Study-level trial analysis.

This module performs lightweight, post hoc analysis over completed sweep
trials. It operates strictly on persisted Optuna study metadata and
materializes simple ranked summaries for downstream inspect
'''

# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.study as study


def analyze(config: configs.RootConfig):
    '''
    Analyze and rank completed sweep trials.

    This function loads an existing Optuna study using the configured
    study name and storage backend, ranks completed trials according to
    their objective values, and persists a small ranked summary as a
    study-level artifact.
    '''

    # load and rank
    sweep_config = config.pipeline.study_sweep
    ranked = study.rank_trials(
        sweep_config.study_name,
        sweep_config.storage,
        top_k=5,
        ascending=False
    )

    # persist artifacts
    analysis_json = f'{config.execution.exp_root}/analysis/{sweep_config.study_name}.json'
    ctrl = artifacts.Controller[list[dict]](analysis_json)
    ctrl.persist(ranked)

    print(ranked)
