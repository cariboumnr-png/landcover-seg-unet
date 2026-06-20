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

'''Data geometry preset objectives.'''

# third-party imports
import optuna
# local imports
import landseg.study.sweep as sweep

def obj_data_geometry(
    trial_cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    Data geometry mutations:
      - Patch size (`int`)
      - Batch size (`int`)
    '''

    study_cfg = trial_cfg.study.data_geometry

    trial_cfg.set_data_patch_size(
        patch_size=trial.suggest_int(
            name='data.patch_size',
            low=study_cfg.patch_size[0],
            high=study_cfg.patch_size[1],
            step=study_cfg.patch_size[2],
        )
    )

    trial_cfg.set_data_batch_size(
        batch_size=trial.suggest_int(
            name='data.batch_size',
            low=study_cfg.batch_size[0],
            high=study_cfg.batch_size[1],
            step=study_cfg.batch_size[2],
        )
    )

    return trial_cfg

def obj_context_window(
    trial_cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    Context window mutation:
      - Patch size (`int`)
    '''

    study_cfg = trial_cfg.study.context_window

    trial_cfg.set_data_patch_size(
        patch_size=trial.suggest_int(
            name='data.patch_size',
            low=study_cfg.patch_size[0],
            high=study_cfg.patch_size[1],
            step=study_cfg.patch_size[2],
        )
    )

    return trial_cfg
