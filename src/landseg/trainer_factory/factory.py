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

'''Factory to build a trainer class'''

# third-party imports
import torch
# local imports
import landseg.configs as configs
import landseg.core as core
import landseg.trainer_components as components
import landseg.trainer_engine as engine
import landseg.utils as utils

def build_trainer(
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: configs.TrainerCfg,
    logger: utils.Logger,
    **kwargs
) -> engine.MultiHeadTrainer:
    '''
    Build a trainer from model, components, and config.

    Args:
        kwargs:
            runtime convenience flags
            - skip_log: bool
            - enable_train_la: bool
            - enable_val_la: bool
            - enable_test_la: bool
    '''

    # gather trainer components
    runtime_components = components.build_trainer_components(
        dataspecs,
        model,
        config,
        logger
    )

    # get runtime traine config
    runtime_config = engine.get_config(config.runtime)

    # get currently avalaible device
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = engine.MultiHeadTrainer(
        model,
        runtime_components,
        runtime_config,
        available_device,
        **kwargs
    )

    return trainer
