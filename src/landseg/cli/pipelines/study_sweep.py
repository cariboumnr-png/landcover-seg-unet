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
Run one study trial.
'''

# local imports
import landseg.cli.pipelines as pipelines
import landseg.configs as configs
import landseg.tuning as tuning


def sweep(config: configs.RootConfig):
    '''doc'''

    study = tuning.run_study(trial, config)

    return {
        'best_value': study.best_value,
        'best_params': study.best_params,
    }


def trial(config: configs.RootConfig) -> float:
    '''
    Run one trial with training and evaluation.

    Args:
        config: RootConfig with model, trainer, and runner settings.
    '''

    # train
    meta = pipelines.train(config)

    # return the best value
    best = meta['summary'].get('best_value') # tighten later
    assert isinstance(best, float)
    return best
