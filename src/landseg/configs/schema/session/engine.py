# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''
Session schema
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.configs.schema.base as base

# alias
field = dataclasses.field


@dataclasses.dataclass
class EngineExec(base.BaseConfigSection):
    use_amp: bool = True
    logit_adjust_alpha: float = 1.0

    def validate(self):
        self.require_attr_type_range('logit_adjust_alpha', float, (0.0, None))


@dataclasses.dataclass
class EngineOptim(base.BaseConfigSection):
    opt_cls: str = 'AdamW'
    lr: float = 1e-4
    weight_decay: float = 1e-3
    sched_cls: str | None = 'CosAnneal'
    sched_args: dict[str, typing.Any] = field(default_factory=lambda: {'T_max': 50})
    grad_clip_norm: float | None = 1.0

    def validate(self):
        self.require_attr_type_range('lr', float, (0.0, None))
        self.require_attr_type_range('weight_decay', float, (0.0, None))
        self.require_attr_type_range('grad_clip_norm', float, (0.0, None))

        if self.sched_cls == 'CosAnneal':
            if 'T_max' not in self.sched_args:
                raise base.ConfigValidationError('missing T_max for CosAnneal')

@dataclasses.dataclass
class EngineSchedule(base.BaseConfigSection):
    val_every_n_epoch: int = 1
    infer_every_n_epoch: int = 1
    ckpt_every_n_epoch: int = 5
    update_loss_every_n_batch: int = 50

    def validate(self):
        self.require_attr_type_range('val_every_n_epoch', int, (1, None))
        self.require_attr_type_range('infer_every_n_epoch', int, (1, None))
        self.require_attr_type_range('ckpt_every_n_epoch', int, (1, None))
        self.require_attr_type_range('update_loss_every_n_batch', int, (1, None))


@dataclasses.dataclass
class FocalLoss(base.BaseConfigSection):
    weight: float = 0.5
    gamma: float = 2.0
    reduction: str = 'mean'

    def validate(self) -> None:
        self.require_attr_type_range('weight', float, (0.0, None))
        self.require_attr_type_range('gamma', float, (0.0, None))


@dataclasses.dataclass
class DiceLoss(base.BaseConfigSection):
    weight: float = 0.5
    smooth: float = 1.0

    def validate(self) -> None:
        self.require_attr_type_range('weight', float, (0.0, None))
        self.require_attr_type_range('smooth', float, (0.0, None))


@dataclasses.dataclass
class SpectralLoss(base.BaseConfigSection):
    weight: float = 1e-3
    alpha: float = 1.0
    neighbour: int = 4

    def validate(self) -> None:
        self.require_attr_type_range('weight', float, (0.0, None))
        self.require_attr_type_range('alpha', float, (0.0, None))
        self.require_attr_type_range('neighbour', int, (4, 4))
        self.require_attr_type_range('neighbour', int, (8, 8))
        # pixel neighbourhood can only be 4 or 8


@dataclasses.dataclass
class TVLoss(base.BaseConfigSection):
    weight: float = 1e-4

    def validate(self) -> None:
        self.require_attr_type_range('weight', float, (0.0, None))


@dataclasses.dataclass
class EcologicalLoss(base.BaseConfigSection):
    weight: float = 0.0
    profile: str | None = 'ontario_tree_species_grouped_profiles'

    def validate(self) -> None:
        self.require_attr_type_range('weight', float, (0.0, None))


@dataclasses.dataclass
class LossConfigs(base.BaseConfigSection):
    focal: FocalLoss = field(default_factory=FocalLoss)
    dice: DiceLoss = field(default_factory=DiceLoss)
    spectral: SpectralLoss = field(default_factory=SpectralLoss)
    tv: TVLoss = field(default_factory=TVLoss)
    ecological: EcologicalLoss = field(default_factory=EcologicalLoss)

    def validate(self) -> None:
        self.focal.validate()
        self.dice.validate()
        self.spectral.validate()
        self.tv.validate()
        self.ecological.validate()


@dataclasses.dataclass
class MTLConstraints(base.BaseConfigSection):
    name: str = ''
    source_head: str = ''
    trigger_val: int = 0
    target_head: str = ''
    forbidden: list[int] = field(default_factory=list)


@dataclasses.dataclass
class MTLRegularization(base.BaseConfigSection):
    consistency_lambda: float = 0.05
    consistency_reduction: str = 'mean'


@dataclasses.dataclass
class EngineTask(base.BaseConfigSection):
    alpha_fn: str = 'effective_n'
    en_beta: float = 0.999
    excluded_cls: dict[str, list[int]] | None = None
    head_weights: dict[str, float] | None = None
    loss_configs: LossConfigs = field(default_factory=LossConfigs)
    mtl_constraints: list[MTLConstraints] | None = None
    mtl_reg_configs: MTLRegularization = field(default_factory=MTLRegularization)

    def validate(self):
        match self.alpha_fn:
            case 'effective_n':
                self.require_attr_type_range('en_beta', float, (0, 1))
            case 'inverse':
                pass
            case _:
                raise base.ConfigValidationError('Invalid loss alpha function')

        if self.head_weights:
            for k, v in self.head_weights.items():
                self.require_type_range(v, f'{k} head_w', float, (0.0, None))


@dataclasses.dataclass
class EngineConfig(base.BaseConfigSection):
    engine_exec: EngineExec = field(default_factory=EngineExec)
    engine_optim: EngineOptim = field(default_factory=EngineOptim)
    engine_schedule: EngineSchedule = field(default_factory=EngineSchedule)
    engine_tasks: EngineTask = field(default_factory=EngineTask)

    def validate(self):
        self.engine_exec.validate()
        self.engine_optim.validate()
        self.engine_schedule.validate()
        self.engine_tasks.validate()
