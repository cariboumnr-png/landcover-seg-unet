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
Session schema
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing

# alias
field = dataclasses.field

# -------------------------------SESSION CONFIGS-------------------------------
# ----- COMPONENTS
@dataclasses.dataclass
class _LoaderConfig:
    patch_size: int = 128
    batch_size: int = 16

@dataclasses.dataclass
class _FocalLossConfig:
    weight: float = 0.5
    gamma: float = 2.0
    reduction: str = 'mean'

@dataclasses.dataclass
class _DiceLossConfig:
    weight: float = 0.5
    smooth: float = 1.0

@dataclasses.dataclass
class _SpectralLossConfig:
    weight: float = 1e-3
    alpha: float = 1.0
    neighbour: int = 4

@dataclasses.dataclass
class _TVLossConfig:
    weight: float = 1e-4

@dataclasses.dataclass
class _LossTypesConfig:
    focal: _FocalLossConfig = field(default_factory=_FocalLossConfig)
    dice: _DiceLossConfig = field(default_factory=_DiceLossConfig)
    spectral: _SpectralLossConfig = field(default_factory=_SpectralLossConfig)
    tv: _TVLossConfig = field(default_factory=_TVLossConfig)

@dataclasses.dataclass
class _LossConfig:
    alpha_fn: str = 'effective_n'
    en_beta: float = 0.999
    excluded_cls: dict[str, list[int]] | None = None
    types: _LossTypesConfig = field(default_factory=_LossTypesConfig)

@dataclasses.dataclass
class _OptimConfig:
    opt_cls: str = 'AdamW'
    lr: float = 1e-4
    weight_decay: float = 1e-3
    sched_cls: str | None = 'CosAnneal'
    sched_args: dict[str, typing.Any] = field(
        default_factory=lambda: {'T_max': 50}
    )

@dataclasses.dataclass
class _ComponentsCfg:
    loader: _LoaderConfig = field(default_factory=_LoaderConfig)
    task: _LossConfig = field(default_factory=_LossConfig)
    optimization: _OptimConfig = field(default_factory=_OptimConfig)

    def validate(self) -> None:
        # Example: scheduler-specific requirements
        if self.optimization.sched_cls == 'CosAnneal':
            if 'T_max' not in self.optimization.sched_args:
                raise ValueError('missing T_max for CosAnneal')

# ----- RUNTIME
@dataclasses.dataclass
class _Schedule:
    max_epoch: int = 50
    max_step: int = 1_000_000
    log_every: int = 50
    val_every: int = 1
    ckpt_every: int = 5

@dataclasses.dataclass
class _Monitor:
    metric_name: str = 'iou'
    track_heads: list[str] = field(default_factory=lambda: ['base'])
    track_mode: str = 'max'
    allow_early_stop: bool = True
    patience: int = 10
    min_delta: float = 0.0005

@dataclasses.dataclass
class _Precision:
    use_amp: bool = True

@dataclasses.dataclass
class _Optimization:
    grad_clip_norm: float | None = 1.0

@dataclasses.dataclass
class _LogitAdjustConfig:
    logit_adjust_alpha: float = 1.0
    enable_train_logit_adjust: bool = False
    enable_eval_logit_adjust: bool = False

@dataclasses.dataclass
class _RuntimeConfig:
    schedule: _Schedule = field(default_factory=_Schedule)
    monitor: _Monitor = field(default_factory=_Monitor)
    precision: _Precision = field(default_factory=_Precision)
    optimization: _Optimization = field(default_factory=_Optimization)
    logit_adjust: _LogitAdjustConfig = field(default_factory=_LogitAdjustConfig)

# ----- PHASES
@dataclasses.dataclass
class _Phase:
    name: str = 'phase_0'
    num_epochs: int = 0
    start_epoch: int = 1 # 1-based
    lr_scale: float | None = 1.0
    active_heads: list[str] | None = None
    frozen_heads: list[str] | None = None

@dataclasses.dataclass
class _DefaultPhases:
    name: str = 'default'
    phases: list[_Phase] = field(default_factory=lambda: [_Phase()])

@dataclasses.dataclass
class _BaselinePhases:
    name: str = 'baseline'
    select_children: list[str] = field(default_factory=list)
    phases: list[_Phase] = field(default_factory=lambda: [_Phase()])

@dataclasses.dataclass
class _PhaseProfiles:
    default: _DefaultPhases = field(default_factory=_DefaultPhases)
    baseline: _BaselinePhases = field(default_factory=_BaselinePhases)

# session composite
@dataclasses.dataclass
class SessionConfig:
    '''doc'''
    resume_from_last: bool = False
    mode: str = 'continuous'
    phase_schema: str = 'default'
    components: _ComponentsCfg = field(default_factory=_ComponentsCfg)
    runtime: _RuntimeConfig = field(default_factory=_RuntimeConfig)
    phases: _PhaseProfiles = field(default_factory=_PhaseProfiles)

    @property
    def single_phase(self) -> _Phase:
        '''Single phase from runtime configs.'''
        return _Phase(
            name='single_phase',
            num_epochs=self.runtime.schedule.max_epoch,
            start_epoch=1,
            lr_scale=1.0,
            active_heads=['base'],
            frozen_heads=None
        )

    @property
    def curriculum(self) -> list[_Phase]:
        '''List of phases by configs.'''
        registry = {
            'default': self.phases.default.phases,
            'baseline': self.phases.baseline.phases
        }
        schema = registry.get(self.phase_schema)
        if schema is None:
            raise ValueError(f'Invalid phase schema: {self.phase_schema}')
        return schema
