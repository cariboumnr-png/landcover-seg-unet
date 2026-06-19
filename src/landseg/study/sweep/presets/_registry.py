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
Sweep objective presets registry and resolver.
'''

# standard imports
from __future__ import annotations
import typing
# third-party imports
import optuna
# local imports
import landseg.study.sweep as sweep
import landseg.study.sweep.presets as presets

PresetFn = typing.Callable[
    [sweep.RootConfigShape, optuna.Trial],
    sweep.RootConfigShape,
]

_REGISTRY: dict[str, PresetFn] = {
    "base": presets.base_objectives,
    "optimizer": presets.optimizer_objectives,
    "throughput": presets.throughput_objectives,
    "data_geometry": presets.data_geometry_objectives,
    "context_window": presets.context_window_objectives,
    "architecture": presets.architecture_objectives,
    "bottleneck": presets.bottleneck_objectives,
    "conditioning": presets.conditioning_objectives,
    "loss_balance": presets.loss_balance_objectives,
    "regularization": presets.regularization_objectives,
    "mtl_consistency": presets.mtl_consistency_objectives,
    "head_weights": presets.head_weights_objectives,
    "mtl_joint": presets.mtl_joint_objectives,
    "hierarchy": presets.hierarchy_objectives,
    "quick": presets.quick_objectives,
    "capacity": presets.capacity_objectives,
    "mtl_quality": presets.mtl_quality_objectives,
    "production_candidate": presets.production_candidate_objectives,
}

#
def resolve(name: str) -> PresetFn:
    '''Resolve presets from name'''
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown preset: {name}") from e
