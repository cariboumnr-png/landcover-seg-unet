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
# pylint: disable=too-few-public-methods

'''Factory to build the trainer components.'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.core as core
import landseg.trainer.components.callback as callback
import landseg.trainer.components.data as data
import landseg.trainer.components.optim as optim
import landseg.trainer.components.task as task
import landseg.utils as utils

#
class _LoaderConfig(typing.Protocol):
    '''doc'''
    @property
    def batch_size(self) -> int: ...
    @property
    def patch_size(self) -> int: ...

class _LossConfig(typing.Protocol):
    '''doc'''
    @property
    def alpha_fn(self) -> str: ...
    @property
    def en_beta(self) -> float: ...
    @property
    def types(self) -> typing.Any: ...

class _OptimConfig(typing.Protocol):
    '''doc'''
    @property
    def opt_cls(self) -> str: ...
    @property
    def lr(self) -> float: ...
    @property
    def weight_decay(self) -> float: ...
    @property
    def sched_cls(self) -> str | None: ...
    @property
    def sched_args(self) -> dict[str, typing.Any]: ...

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
    logger: utils.Logger,
    *,
    data_specs: core.DataSpecs,
    model: core.MultiheadModelLike,
    loader_config: _LoaderConfig,
    loss_config: _LossConfig,
    optim_config: _OptimConfig
 ) -> TrainerComponents:
    '''Builder trainer.'''

    # generate callback instances
    callbacks = callback.build_callbacks(logger)

    # compile data loaders
    data_loaders = data.build_dataloaders(
        data_specs,
        loader_config.batch_size,
        loader_config.patch_size,
        logger
    )

    # compile training heads basic specifications
    headspecs = task.build_headspecs(
        data_specs,
        loss_config.alpha_fn,
        en_beta=loss_config.en_beta
    )

    # compile training heads loss compute modules
    headlosses = task.build_headlosses(
        headspecs,
        loss_config.types,
        data_specs.meta.ignore_index,
    )

    # compile training heads metric compute modules
    headmetrics = task.build_headmetrics(
        headspecs,
        data_specs.meta.ignore_index
    )

    # build optimizer and scheduler
    optimization = optim.build_optimization(
        model,
        optim_config.opt_cls,
        optim_config.lr,
        optim_config.weight_decay,
        optim_config.sched_cls,
        **optim_config.sched_args
    )

    # collect components
    return TrainerComponents(
        dataloaders=data_loaders,
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,
        optimization=optimization,
        callbacks=callbacks,
    )
