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

# aliases
field = dataclasses.field

@dataclasses.dataclass(frozen=True)
class TrainingSessionStep:
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
    phase_name: str # identity / location
    phase_index: int
    epoch: int = 1
    metrics: EpochResults | None = None # metrics
    is_phase_end: bool = False # termination signals
    is_run_end: bool = False
    early_stop_reason: str | None = None

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
    training: TrainerEpochResults | None
    validation: EvaluatorEpochResults | None

    def __str__(self) -> str:
        return '\n'.join([
            'Training Results:',
            str(self.training),
            'Validation Results:',
            str(self.validation),
        ])

@dataclasses.dataclass
class TrainerEpochResults:
    '''Trainer aggregated results.'''
    all_heads: list[str]
    current_bidx: int = 0
    total_loss: float = 0.0
    head_losses: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.head_losses = {h: 0.0 for h in self.all_heads}

    def __str__(self) -> str:
        return '\n'.join([
            f'All Heads: {self.all_heads}',
            f'Current Mean Total Loss: {self.mean_total_loss}',
            'Perhead Loss\n',
        ]) + '\n'.join([
            f'{h}: {l:4f}' for h, l in self.head_losses.items()
        ])

    @property
    def mean_total_loss(self) -> float:
        '''Return current moving average total loss.'''
        return self.total_loss / max(1, self.current_bidx)

    @property
    def mean_head_losses(self) -> dict[str, float]:
        '''Return current moving average per head losses.'''
        n = max(1, self.current_bidx)
        return {h: l / n for h, l in self.head_losses.items()}

    def clear(self) -> None:
        '''Reset all loss values to `0.0`.'''

        self.total_loss = 0.0
        for h in self.head_losses:
            self.head_losses[h] = 0.0

@dataclasses.dataclass
class EvaluatorEpochResults:
    '''Evaluator aggregated epoch results.'''
    all_heads: list[str]
    monitor_heads: list[str]
    head_metrics: dict[str, AccumulatedMetrics] = field(default_factory=dict)

    def __post_init__(self):
        self.head_metrics = {h: AccumulatedMetrics() for h in self.all_heads}

    def __str__(self) -> str:
        return '\n'.join([
            f'All Heads: {self.all_heads}',
            f'Current Monitor Heads: {self.monitor_heads}',
            'Perhead IoUs\n',
        ]) + '\t\n'.join([
            f'{h}: {m.as_str_list}'
            for h, m in self.head_metrics.items() if h in self.monitor_heads
        ])

    @property
    def target_metrics(self) -> float:
        '''
        Return the mean metrics from all monitor heads.

        For each head if active classes are set, prefer mean IoU from the
        active classes; otherwise count mean from all present classes.
        '''
        # retrieve iou metrics from monitor heads
        mean = 0.0
        mean_ac = 0.0

        # accumulate from all monitor heads
        for head in self.monitor_heads:
            assert head in self.head_metrics # sanity
            metrics = self.head_metrics[head]
            # accumulate moniter metrics for stae
            mean += metrics.mean
            mean_ac +=  metrics.ac_mean
            # collect per head metrics formatted strings

        # get average ious
        mean /= max(1, len(self.monitor_heads))
        mean_ac /= max(1, len(self.monitor_heads))

        # pick iou - prefer iou from active classes if present
        if not any([mean, mean_ac]):
            raise ValueError('No validation metrics found')
        return mean_ac if mean_ac else mean

@dataclasses.dataclass
class AccumulatedMetrics:
    '''Container for IoU metrics, supports, and active-class views.'''
    mean: float = 0.0
    ious: dict[str, float] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)
    ac_mean: float = 0.0
    ac_ious: dict[str, float] = field(default_factory=dict)
    ac_support: dict[str, int] = field(default_factory=dict)
    _locked: bool = field(default=False, init=False, repr=False)

    def __setattr__(self, key, value) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("Object is immutable after compute()")
        super().__setattr__(key, value)

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return as the metrics a nested dictionary.'''
        return dataclasses.asdict(self)

    @property
    def as_str_list(self) -> list[str]:
        '''Human-readable summary for mean/class IoUs (all/active).'''
        str_list: list[str] = []
        # all classes
        m = f'{self.mean:.4f}'
        str_list.append('Mean IoU (all): ' + m)
        c = '|'.join(f'cls{k}={v:.4f}' for k, v in self.ious.items())
        str_list.append('Class IoU (all): ' + c)
        # s = '|'.join(f'cls{k}={v}' for k, v in mm['support'].items())
        # text.append('Class support (all):\t' + s)
        # subset of active classes (if not None)
        if bool(self.ac_mean):
            m = f'{self.ac_mean:.4f}'
            str_list.append('Mean IoU (active): ' + m)
            c = '|'.join(f'cls{k}={v:.4f}' for k, v in self.ac_ious.items())
            str_list.append('Class IoU (active): ' + c)
            # s = '|'.join(f'cls{k}={v}' for k, v in mm['ac_support'].items())
            # text.append('Class support (active):\t' + s)
        # return text lines
        return str_list

    def lock(self) -> None:
        '''Lock object via __setattr__ blocking.'''

        self._locked = True

    def reset(self) -> None:
        '''Reset all attributes to default; will unlock.'''

        object.__setattr__(self, "_locked", False) # lock override
        self.mean = 0.0
        self.ious.clear()
        self.support.clear()
        self.ac_mean = 0.0
        self.ac_ious.clear()
        self.ac_support.clear()
