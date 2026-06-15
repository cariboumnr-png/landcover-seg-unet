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
import dataclasses
import typing
# local imports
import landseg.configs.schema.utils as utils

# alias
field = dataclasses.field

# -------------------------------SESSION CONFIGS-------------------------------
# ----- data loader
@dataclasses.dataclass
class _DataLoaderConfig:
    patch_size: int = 128
    batch_size: int = 16

    def validate(self):
        utils.must_within(self.patch_size, 'data patch size', 0)
        utils.must_within(self.batch_size, 'data batch size', 0)

# ----- engione executor
@dataclasses.dataclass
class _EngineExecConfig:
    use_amp: bool = True
    logit_adjust_alpha: float = 1.0

    def validate(self):
        utils.must_within(self.logit_adjust_alpha, 'logit adjust alpha', 0)

# ----- engine optim
@dataclasses.dataclass
class _OptimConfig:
    opt_cls: str = 'AdamW'
    lr: float = 1e-4
    weight_decay: float = 1e-3
    sched_cls: str | None = 'CosAnneal'
    sched_args: dict[str, typing.Any] = field(default_factory=lambda: {'T_max': 50})
    grad_clip_norm: float | None = 1.0

    def validate(self):
        utils.must_within(self.lr, 'learning rate', 0)
        utils.must_within(self.weight_decay, 'weight decay', 0)
        utils.must_within(self.grad_clip_norm, 'gradient norm clipping', 0)
        # scheduler specific requirements
        if self.sched_cls == 'CosAnneal':
            if 'T_max' not in self.sched_args:
                raise ValueError('missing T_max for CosAnneal')

# ----- engine tasks
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
class _MTLConstraints:
    name: str = ''
    source_head: str = ''
    trigger_val: int = 0
    target_head: str = ''
    forbidden: list[int] = field(default_factory=list)

@dataclasses.dataclass
class _TasksConfig:
    alpha_fn: str = 'effective_n'
    en_beta: float = 0.999
    excluded_cls: dict[str, list[int]] | None = None
    loss_types: _LossTypesConfig = field(default_factory=_LossTypesConfig)
    constraints: list[_MTLConstraints] | None = None

    def validate(self):
        match self.alpha_fn:
            case 'effective_n': utils.must_within(self.en_beta, 'EN beta', 0, 1)
            case 'inverse': pass
            case _: raise ValueError('Invalid loss alpha function')
        utils.must_within(self.loss_types.focal.weight, 'focal loss weight', 0)
        utils.must_within(self.loss_types.dice.weight, 'dice loss weight', 0)
        utils.must_within(self.loss_types.spectral.weight, 'spectral loss weight', 0)
        utils.must_within(self.loss_types.tv.weight, 'tv loss weight', 0)

# ----- orchestration
@dataclasses.dataclass
class _Schedule:
    val_every_n_epoch: int = 1
    infer_every_n_epoch: int = 1
    ckpt_every_n_epoch: int = 5
    update_loss_every_n_batch: int = 50
    resume_from_last: bool = False

    def validate(self):
        utils.must_within(self.val_every_n_epoch, 'validation frequency', 1)
        utils.must_within(self.infer_every_n_epoch, 'inference frequency', 1)
        utils.must_within(self.ckpt_every_n_epoch, 'Saving frequency', 1)
        utils.must_within(self.update_loss_every_n_batch, 'loss update frequency', 1)

@dataclasses.dataclass
class _Monitor:
    metric_name: str = 'iou'
    track_heads: dict[str, float] | None = None
    track_mode: str = 'max'
    allow_early_stop: bool = True
    patience: int | None = 10
    min_delta: float | None = 0.0005

    def validate(self):
        utils.must_within(self.patience, 'patience epoch', 0)
        utils.must_within(self.min_delta, 'patience delta', 0)

@dataclasses.dataclass
class _Phase:
    name: str = 'phase_0'
    num_epochs: int = 0
    start_epoch: int = 1 # 1-based
    lr_scale: float | None = 1.0
    active_heads: list[str] | None = None
    frozen_heads: list[str] | None = None

    def validate(self):
        utils.must_within(self.num_epochs, f'{self.name}: number of epochs', 0)
        utils.must_within(self.start_epoch, f'{self.name}:starting epoch', 0)
        utils.must_within(self.lr_scale, 'LR scale', 0)
        if self.start_epoch > self.num_epochs:
            raise ValueError(
                f'Starting epochs {self.num_epochs} for phase {self.name} is '
                f'larger than the max number of epochs {self.num_epochs}'
            )

@dataclasses.dataclass
class _SinglePhase:
    name: str = 'single'
    phases: list[_Phase] = field(default_factory=lambda: [_Phase()])

@dataclasses.dataclass
class _BaselinePhases:
    name: str = 'baseline'
    phases: list[_Phase] = field(default_factory=lambda: [_Phase()])

@dataclasses.dataclass
class _CustomPhases:
    name: str = 'custom'
    phases: list[_Phase] = field(default_factory=lambda: [_Phase()])

@dataclasses.dataclass
class _Curriculum:
    schema: str = 'single'
    single: _SinglePhase = field(default_factory=_SinglePhase)
    baseline: _BaselinePhases = field(default_factory=_BaselinePhases)
    custom: _CustomPhases = field(default_factory=_CustomPhases)

@dataclasses.dataclass
class _OrchestrationConfig:
    schedule: _Schedule = field(default_factory=_Schedule)
    monitor: _Monitor = field(default_factory=_Monitor)
    curriculum: _Curriculum = field(default_factory=_Curriculum)

    @property
    def single_phase(self) -> _Phase:
        '''Single phase from runtime configs.'''
        return self.curriculum.single.phases[0]

    @property
    def multi_phases(self) -> list[_Phase]:
        '''List of phases by configs.'''
        # currently supported pre-configured phases
        schema = self.curriculum.schema
        match schema:
            case 'baseline': return self.curriculum.baseline.phases
            case 'custom': return self.curriculum.custom.phases
            case _: raise ValueError(f'Invalid multi-phases schema: {schema}')

    def validate(self):
        self.schedule.validate()
        self.monitor.validate()
        if self.curriculum.schema == 'single':
            self.single_phase.validate()
        else:
            for phase in self.multi_phases:
                phase.validate()

# session composite
@dataclasses.dataclass
class SessionConfig:
    mode: str = 'continuous'
    data_loader: _DataLoaderConfig = field(default_factory=_DataLoaderConfig)
    engine_exec: _EngineExecConfig = field(default_factory=_EngineExecConfig)
    engine_optim: _OptimConfig = field(default_factory=_OptimConfig)
    engine_tasks: _TasksConfig = field(default_factory=_TasksConfig)
    orchestration: _OrchestrationConfig = field(default_factory=_OrchestrationConfig)

    def __post_init__(self):
        # allow_early_stop=True is invalid for curriculum
        if self.mode == 'curriculum':
            self.orchestration.monitor.allow_early_stop = False

    def validate(self):
        # mode specific requirements
        if self.mode == 'continuous':
            if self.orchestration.curriculum.schema != 'single':
                raise ValueError(
                    '[curriculum.schema] must be "single" for a continuous '
                    'training session'
                )
        elif self.mode == 'curriculum':
            if self.orchestration.curriculum.schema == 'single':
                raise ValueError(
                    '[curriculum.schema] must not be "single" for a curriculum'
                    '-based training session; expected: "baseline" or "custom"'
                )
        else:
            raise ValueError(
                f'Invalid mode: {self.mode}, '
                f'must be "continuous" or "curriculum"'
            )
        # sections validation
        self.data_loader.validate()
        self.engine_exec.validate()
        self.engine_optim.validate()
        self.engine_tasks.validate()
        self.orchestration.validate()
