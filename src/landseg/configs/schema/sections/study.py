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

'''
Study schema.
'''

# standard imports
import dataclasses

# alias
field = dataclasses.field
# Constants
MODEL_BODIES = ['unet', 'unetpp', 'unetppp']
BOTTLENECKS = ['conv', 'transformer', 'hybrid']

# --------------------------------STUDY CONFIGS--------------------------------
@dataclasses.dataclass
class _BaseObj:
    learning_rate: tuple[float, float] = (1e-5, 1e-1)
    batch_size: tuple[int, int, int] = (16, 64, 16)

@dataclasses.dataclass
class _OptimizerObj:
    learning_rate: tuple[float, float] = (1e-5, 1e-1)
    weight_decay: tuple[float, float] = (1e-6, 1e-2)

@dataclasses.dataclass
class _ThroughputObj:
    batch_size: tuple[int, int, int] = (16, 64, 16)
    use_amp: list[bool] = field(default_factory=lambda: [True, False])

@dataclasses.dataclass
class _DataGeometryObj:
    patch_size: tuple[int, int, int] = (64, 128, 64)
    batch_size: tuple[int, int, int] = (16, 64, 16)

@dataclasses.dataclass
class _ContextWindowObj:
    patch_size: tuple[int, int, int] = (64, 128, 64)

@dataclasses.dataclass
class _ArchitectureObj:
    model_body: list[str] = field(default_factory=lambda: MODEL_BODIES)
    base_channel: tuple[int, int, int] = (16, 64, 16)
    bottleneck: list[str] = field(default_factory=lambda: BOTTLENECKS)

@dataclasses.dataclass
class _BottleneckObj:
    bottleneck: list[str] = field(default_factory=lambda: BOTTLENECKS)
    num_conv_blocks: tuple[int, int, int] = (1, 4, 1)
    num_transformer_blocks: tuple[int, int, int] = (1, 4, 1)
    num_heads: list[int] = field(default_factory=lambda: [2, 4, 8])
    mlp_ratio: tuple[float, float] = (1.0, 4.0)
    dropout: tuple[float, float] = (0.0, 0.5)
    attn_dropout: tuple[float, float] = (0.0, 0.5)

@dataclasses.dataclass
class _ConditioningObj:
    conditioners: list[list[str]] = field(
        default_factory=lambda: [[], ['film'], ['concat'], ['film', 'concat']]
    )

@dataclasses.dataclass
class _LossBalanceObj:
    focal_weight: tuple[float, float] = (0.0, 1.0)
    dice_weight: tuple[float, float] = (0.0, 1.0)

@dataclasses.dataclass
class _RegularizationObj:
    spectral_weight: tuple[float, float] = (0.0, 1e-2)
    tv_weight: tuple[float, float] = (0.0, 1e-3)

@dataclasses.dataclass
class _MtlConsistencyObj:
    consistency_lambda: tuple[float, float] = (0.0, 1.0)

@dataclasses.dataclass
class _HeadWeightsObj:
    logit_adjust_alpha: tuple[float, float] = (0.0, 2.0)

@dataclasses.dataclass
class _MtlJointObj:
    consistency_lambda: tuple[float, float] = (0.0, 1.0)
    logit_adjust_alpha: tuple[float, float] = (0.0, 2.0)

@dataclasses.dataclass
class _HierarchyObj:
    consistency_lambda: tuple[float, float] = (0.0, 1.0)
    consistency_reduction: list[str] = field(default_factory=lambda: ['mean', 'sum'])

@dataclasses.dataclass
class StudyConfig:
    base: _BaseObj = field(default_factory=_BaseObj)
    optimizer: _OptimizerObj = field(default_factory=_OptimizerObj)
    throughput: _ThroughputObj = field(default_factory=_ThroughputObj)
    data_geometry: _DataGeometryObj = field(default_factory=_DataGeometryObj)
    context_window: _ContextWindowObj = field(default_factory=_ContextWindowObj)
    architecture: _ArchitectureObj = field(default_factory=_ArchitectureObj)
    bottleneck: _BottleneckObj = field(default_factory=_BottleneckObj)
    conditioning: _ConditioningObj = field(default_factory=_ConditioningObj)
    loss_balance: _LossBalanceObj = field(default_factory=_LossBalanceObj)
    regularization: _RegularizationObj = field(default_factory=_RegularizationObj)
    mtl_consistency: _MtlConsistencyObj = field(default_factory=_MtlConsistencyObj)
    head_weights: _HeadWeightsObj = field(default_factory=_HeadWeightsObj)
    mtl_joint: _MtlJointObj = field(default_factory=_MtlJointObj)
    hierarchy: _HierarchyObj = field(default_factory=_HierarchyObj)
