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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods

'''Factory to build the session-owned components.'''

# standard imports
from __future__ import annotations
import dataclasses
import typing
# local imports
import landseg.core as core
import landseg.session.engine.runtime.tasks.heads as heads
import landseg.session.engine.runtime.tasks.loss as loss
import landseg.session.engine.runtime.tasks.metrics as metrics

# ---------------------------------Public Type---------------------------------
class TaskConfig(typing.Protocol):
    @property
    def alpha_fn(self) -> str: ...
    @property
    def en_beta(self) -> float: ...
    @property
    def excluded_cls(self) -> dict[str, list[int]] | None: ...
    @property
    def loss_types(self) -> _LossTypes: ...

# --------------------------------private  type--------------------------------
class _LossTypes(typing.Protocol):
    @property
    def focal(self) -> _FocalLoss: ...
    @property
    def dice(self) -> _DiceLoss: ...
    @property
    def spectral(self) -> _SpectralLoss: ...
    @property
    def tv(self) -> _TotalVariationLoss: ...

class _FocalLoss(typing.Protocol):
    @property
    def weight(self) -> float: ...
    @property
    def gamma(self) -> float: ...
    @property
    def reduction(self) -> str: ...

class _DiceLoss(typing.Protocol):
    @property
    def weight(self) -> float: ...
    @property
    def smooth(self) -> float: ...

class _SpectralLoss(typing.Protocol):
    @property
    def weight(self) -> float: ...
    @property
    def alpha(self) -> float: ...
    @property
    def neighbour(self) -> int: ...

class _TotalVariationLoss(typing.Protocol):
    @property
    def weight(self) -> float: ...


# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class EngineTasks:
    '''A simple container of the session components.'''
    headspecs: heads.HeadSpecs
    headlosses: loss.HeadLosses
    headmetrics: metrics.HeadMetrics

# -------------------------------Public Function-------------------------------
def build_engine_tasks(
    data_specs: core.DataSpecs,
    config: TaskConfig,
 ) -> EngineTasks:
    '''Builder session components from data shape and configs.'''

    # task - heads specifications
    headspecs = heads.build_headspecs(
        data_specs,
        alpha_fn=config.alpha_fn,
        en_beta=config.en_beta,
        excluded_cls=config.excluded_cls
    )
    # task - heads loss modules
    headlosses = loss.build_headlosses(
        headspecs,
        config=config,
        ignore_index=data_specs.meta.label_specs.ignore_index,
        spectral_band_indices=data_specs.meta.image_specs.spec_channels
    )
    # task - heads metric modules
    headmetrics = metrics.build_headmetrics(
        headspecs,
        ignore_index=data_specs.meta.label_specs.ignore_index
    )

    # collect components
    return EngineTasks(
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,

    )
