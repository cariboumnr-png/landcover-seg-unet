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

'''Sweep objective presets registry and resolver.'''

# standard imports
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
    "base": presets.obj_base,
    "optimizer": presets.obj_optimizer,
    "throughput": presets.obj_throughput,
    "data_geometry": presets.obj_data_geometry,
    "context_window": presets.obj_context_window,
    "architecture": presets.obj_architecture,
    "bottleneck": presets.obj_bottleneck,
    "conditioning": presets.obj_conditioning,
    "loss_balance": presets.obj_loss_balance,
    "regularization": presets.obj_loss_aux,
    "mtl_consistency": presets.obj_regularization,
    "head_weights": presets.obj_head_weights,
    "mtl_joint": presets.obj_mtl_joint,
    "hierarchy": presets.obj_hierarchy,
    "quick": presets.comp_quick,
    "capacity": presets.comp_capacity,
    "mtl_quality": presets.comp_mtl_quality,
    "production_candidate": presets.comp_candidate,
}

#
def resolve(name: str) -> PresetFn:
    '''Resolve presets from name'''
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown preset: {name}") from e
