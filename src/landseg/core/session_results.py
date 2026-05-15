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
Session step
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing

#
if typing.TYPE_CHECKING:
    import torch

# aliases
field = dataclasses.field

@dataclasses.dataclass(frozen=True)
class TrainingSessionStep: # pylint: disable=too-many-instance-attributes
    '''
    Immutable training progress snapshot exposed by runners.

    A TrainingStep represents the externally observable state of training
    after a completed epoch. It is the sole unit of progress emitted by
    runner generators and forms the public execution contract consumed
    by CLI pipelines, sweep logic, and monitoring tools.

    Invariants:
        - Exactly one TrainingStep is yielded per completed epoch.
        - Exactly one terminal TrainingStep is yielded per run
          (is_run_end == True).
        - metrics reflect the most recently completed epoch.
        - No TrainingStep represents partial or in-progress execution.
    '''
    # identity / location
    phase_name: str
    phase_index: int
    phase_max_epoch: int
    epoch_in_phase: int
    global_epoch: int
    # termination signals
    is_phase_end: bool
    is_run_end: bool
    stop_reason: str | None
    # metrics
    objective_name: str
    objective_value: float
    best_value_so_far: float
    best_epoch_so_far: int
    is_best_epoch: bool
    raw_metrics: EpochResults # raw

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return as a dictionary for serialization.'''
        return dataclasses.asdict(self)

@dataclasses.dataclass(frozen=True)
class EpochResults:
    '''
    Immutable container for metrics produced during a single epoch.

    Attributes:
        training: Results returned by the trainer for the epoch, or None
            if no training step was executed.
        validation: Results returned by the evaluator, or None if no
            evaluation step was executed.
    '''
    training: TrainerEpochResults | None = None
    validation: ValidationEpochResults | None = None
    inference: InferenceEpochResults | None = None

    @property
    def target_objective(self) -> str:
        '''Return the targert objective from the validation results.'''
        if self.validation:
            return '|'.join(self.validation.head_metrics.keys())
        return 'N/A'

    @property
    def target_metrics(self) -> float:
        '''
        Return the mean validation IoU from active heads.

        For each head if excluded classes are set, compute without them;
        otherwise compute from all present classes.
        '''
        if self.validation:
            return _get_mean_iou(self.validation.head_metrics)
        return -float('inf')

    @property
    def inference_metrics(self) -> float:
        '''
        Return the mean inference IoU from active heads.

        For each head if excluded classes are set, compute without them;
        otherwise compute from all present classes.
        '''
        if self.inference:
            return _get_mean_iou(self.inference.head_metrics)
        return -float('inf')

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return as a dictionary for serialization.'''
        return dataclasses.asdict(self)

@dataclasses.dataclass
class TrainerEpochResults:
    '''Results that update regularly during training flush at the end.'''
    epoch_step: int = 1
    global_step: int = 1
    metrics_updated: bool = False
    total_loss: float = 0.0
    current_lr: float | None = None
    head_losses: dict[str, float] = field(default_factory=dict)

    def clear(self) -> None:
        '''Reset all loss values to `0.0`.'''

        self.total_loss = 0.0
        for h in self.head_losses:
            self.head_losses[h] = 0.0

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return as a dictionary for serialization.'''
        return dataclasses.asdict(self)

@dataclasses.dataclass
class ValidationEpochResults:
    '''Evaluator aggregated epoch results.'''
    head_metrics: dict[str, AccumulatedMetrics] = field(default_factory=dict)

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return as a dictionary for serialization.'''
        return {k: v.as_dict for k, v in self.head_metrics.items()}

@dataclasses.dataclass
class InferenceEpochResults:
    '''Containe for inference results.'''
    head_metrics: dict[str, AccumulatedMetrics] = field(default_factory=dict)
    infer_labels: dict[str, torch.Tensor] = field(default_factory=dict)
    infer_preds: dict[str, torch.Tensor] = field(default_factory=dict)
    infer_errors: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return as a dictionary for serialization.'''
        return {k: v.as_dict for k, v in self.head_metrics.items()}

@dataclasses.dataclass
class AccumulatedMetrics:
    '''Container for IoU metrics, supports, and active-class views.'''
    cmatrix: list[list[int]] = field(default_factory=list)
    mean: float = 0.0
    ious: dict[str, float] = field(default_factory=dict)
    ac_mean: float = 0.0
    ac_ious: dict[str, float] = field(default_factory=dict)
    _locked: bool = field(default=False, init=False, repr=False)

    def __setattr__(self, key, value) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("Object is immutable after compute()")
        super().__setattr__(key, value)

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return as a dictionary for serialization.'''
        return dataclasses.asdict(self).pop('_locked')

    @property
    def as_str_list(self) -> list[str]:
        '''Human-readable summary for mean/class IoUs (all/active).'''
        str_list: list[str] = []
        # all classes
        m = f'{self.mean:.4f}'
        str_list.append('Mean IoU (all): ' + m)
        c = '|'.join(f'cls{k}={v:.4f}' for k, v in self.ious.items())
        str_list.append('Class IoU (all): ' + c)
        # subset of active classes (if not None)
        if bool(self.ac_mean):
            m = f'{self.ac_mean:.4f}'
            str_list.append('Mean IoU (active): ' + m)
            c = '|'.join(f'cls{k}={v:.4f}' for k, v in self.ac_ious.items())
            str_list.append('Class IoU (active): ' + c)
        # return strings
        return str_list

    def lock(self) -> None:
        '''Lock object via __setattr__ blocking.'''
        self._locked = True

# ------------------------------private  function------------------------------
def _get_mean_iou(head_metrics: dict[str, AccumulatedMetrics]) -> float:
    '''Mean IoU calculation helper.'''

    # retrieve iou metrics from monitor heads
    mean = 0.0
    mean_ac = 0.0

    # accumulate from all monitor heads
    for metrics in head_metrics.values():
        # accumulate moniter metrics for stae
        mean += metrics.mean
        mean_ac +=  metrics.ac_mean
        # collect per head metrics formatted strings

    # get average ious
    mean /= max(1, len(head_metrics))
    mean_ac /= max(1, len(head_metrics))

    # pick iou - prefer iou from active classes if present
    if not any([mean, mean_ac]):
        raise ValueError('No validation metrics found')
    return mean_ac if mean_ac else mean
