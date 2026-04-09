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

# pylint: disable=missing-function-docstring

'''Factory to build the trainer components.'''

# standard imports
import dataclasses
# local imports
import landseg.core as core
import landseg.trainer.components.callback as callback
import landseg.trainer.components.data as data
import landseg.trainer.components.optim as optim
import landseg.trainer.components.task as task
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class TrainerComponents:
    '''Collection of trainer components.'''
    callbacks: callback.CallbackSet
    dataloaders: data.DataLoaders
    headspecs: task.HeadSpecs
    headlosses: task.HeadLosses
    headmetrics: task.HeadMetrics
    optimization: optim.Optimization

# -------------------------------Public Function-------------------------------
def build_trainer_components(
    *,
    data_specs: core.DataSpecs,
    model: core.MultiheadModelLike,
    data_config: data.LoaderConfig,
    task_config: task.TaskConfig,
    optim_config: optim.OptimConfig,
    logger: utils.Logger,
 ) -> TrainerComponents:
    '''Builder trainer.'''

    # data
    data_loaders = data.build_dataloaders(
        data_specs,
        data_config,
        logger=logger,
    )

    # task
    # heads specifications
    headspecs = task.build_headspecs(
        data_specs,
        alpha_fn=task_config.alpha_fn,
        en_beta=task_config.en_beta
    )
    # heads loss modules
    headlosses = task.build_headlosses(
        headspecs,
        config=task_config.types,
        ignore_index=data_specs.meta.ignore_index,
    )
    # heads metric modules
    headmetrics = task.build_headmetrics(
        headspecs,
        ignore_index=data_specs.meta.ignore_index
    )

    # optimizer and scheduler
    optimization = optim.build_optimization(model, optim_config)

    # callback system
    callbacks = callback.build_callbacks(logger)

    # collect components
    return TrainerComponents(
        dataloaders=data_loaders,
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,
        optimization=optimization,
        callbacks=callbacks,
    )
