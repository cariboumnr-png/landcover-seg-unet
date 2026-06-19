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
import typing

# alias
field = dataclasses.field

# --------------------------------STUDY CONFIGS--------------------------------
@dataclasses.dataclass
class _BaseObjectives:
    learning_rate: tuple[float, float] = (1e-5, 1e-1)   # low, high
    weight_decay: tuple[float, float] = (1e-6, 1e-2)    # low, high
    patch_size: tuple[int, int, int] = (64, 128, 64)    # low, high, step
    batch_size: tuple[int, int, int] = (16, 64, 16)     # low, high, step

@dataclasses.dataclass
class _OptimizerObjectives:
    learning_rate: tuple[float, float] = (1e-5, 1e-1)
    weight_decay: tuple[float, float] = (1e-6, 1e-2)

@dataclasses.dataclass
class _ThroughputObjectives:
    batch_size: tuple[int, int, int] = (16, 64, 16)
    use_amp: list[bool] = field(default_factory=lambda: [True, False])

@dataclasses.dataclass
class _DataGeometryObjectives:
    patch_size: tuple[int, int, int] = (64, 128, 64)
    batch_size: tuple[int, int, int] = (16, 64, 16)

@dataclasses.dataclass
class _ContextWindowObjectives:
    patch_size: tuple[int, int, int] = (64, 128, 64)

@dataclasses.dataclass
class _ArchitectureObjectives:
    model_body: list[str] = field(
        default_factory=lambda: ['unet', 'unetpp', 'unetppp']
    )
    base_channel: tuple[int, int, int] = (16, 64, 16)
    bottleneck: list[str] = field(
        default_factory=lambda: ['conv', 'transformer', 'hybrid']
    )

@dataclasses.dataclass
class _BottleneckObjectives:
    bottleneck: list[str] = field(
        default_factory=lambda: ['conv', 'transformer', 'hybrid']
    )
    num_conv_blocks: tuple[int, int, int] = (1, 4, 1)
    num_transformer_blocks: tuple[int, int, int] = (1, 4, 1)
    num_heads: list[int] = field(default_factory=lambda: [2, 4, 8])
    mlp_ratio: tuple[float, float] = (1.0, 4.0)
    dropout: tuple[float, float] = (0.0, 0.5)
    attn_dropout: tuple[float, float] = (0.0, 0.5)

@dataclasses.dataclass
class _ConditioningObjectives:
    conditioners: list[list[str]] = field(
        default_factory=lambda: [[], ['film'], ['concat']]
    )

@dataclasses.dataclass
class _LossBalanceObjectives:
    focal_weight: tuple[float, float] = (0.0, 1.0)
    dice_weight: tuple[float, float] = (0.0, 1.0)

@dataclasses.dataclass
class _RegularizationObjectives:
    spectral_weight: tuple[float, float] = (0.0, 1e-2)
    tv_weight: tuple[float, float] = (0.0, 1e-3)

@dataclasses.dataclass
class _MtlConsistencyObjectives:
    consistency_lambda: tuple[float, float] = (0.0, 1.0)

@dataclasses.dataclass
class _HeadWeightsObjectives:
    logit_adjust_alpha: tuple[float, float] = (0.0, 2.0)

@dataclasses.dataclass
class _MtlJointObjectives:
    consistency_lambda: tuple[float, float] = (0.0, 1.0)
    logit_adjust_alpha: tuple[float, float] = (0.0, 2.0)

@dataclasses.dataclass
class _HierarchyObjectives:
    consistency_lambda: tuple[float, float] = (0.0, 1.0)
    consistency_reduction: list[str] = field(
        default_factory=lambda: ['mean', 'sum']
    )

@dataclasses.dataclass
class StudyConfig:
    base: _BaseObjectives = field(default_factory=_BaseObjectives)
    optimizer: _OptimizerObjectives = field(
        default_factory=_OptimizerObjectives
    )
    throughput: _ThroughputObjectives = field(
        default_factory=_ThroughputObjectives
    )
    data_geometry: _DataGeometryObjectives = field(
        default_factory=_DataGeometryObjectives
    )
    context_window: _ContextWindowObjectives = field(
        default_factory=_ContextWindowObjectives
    )
    architecture: _ArchitectureObjectives = field(
        default_factory=_ArchitectureObjectives
    )
    bottleneck: _BottleneckObjectives = field(
        default_factory=_BottleneckObjectives
    )
    conditioning: _ConditioningObjectives = field(
        default_factory=_ConditioningObjectives
    )
    loss_balance: _LossBalanceObjectives = field(
        default_factory=_LossBalanceObjectives
    )
    regularization: _RegularizationObjectives = field(
        default_factory=_RegularizationObjectives
    )
    mtl_consistency: _MtlConsistencyObjectives = field(
        default_factory=_MtlConsistencyObjectives
    )
    head_weights: _HeadWeightsObjectives = field(
        default_factory=_HeadWeightsObjectives
    )
    mtl_joint: _MtlJointObjectives = field(
        default_factory=_MtlJointObjectives
    )
    hierarchy: _HierarchyObjectives = field(
        default_factory=_HierarchyObjectives
    )


