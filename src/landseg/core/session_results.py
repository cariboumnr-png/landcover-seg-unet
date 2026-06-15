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
class SessionStepSummary: # pylint: disable=too-many-instance-attributes
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
    raw_metrics: SessionStepResults # raw

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return as a dictionary for serialization.'''
        raw = self.raw_metrics
        return {
            'training': raw.training.as_dict if raw.training else None,
            'validation': raw.validation.as_dict if raw.validation else None,
            'inference': raw.inference.as_dict if raw.inference else None,
        }

@dataclasses.dataclass(frozen=True)
class SessionStepResults:
    '''
    Immutable container for metrics produced during a single epoch.

    Attributes:
        training: Results returned by the trainer for the epoch, or None
            if no training step was executed.
        validation: Results returned by the evaluator, or None if no
            evaluation step was executed.
    '''

    training: TrainStepResults | None = None
    validation: ValStepResults | None = None
    inference: InferStepResults | None = None
    _metric_name: str = 'iou'
    _track_heads: dict[str, float] | None = None

    @property
    def target_objective(self) -> str:
        '''Return the targert objective from the validation results.'''
        assert self.validation
        s = f'Tracking_metrics: {self._metric_name}'
        if s == 'iou':
            if self._track_heads is None:
                _heads = list(self.validation.head_metrics.keys())[0]
            else:
                _heads = ', '.join(self._track_heads)
            s = f'{s} | Active_heads: {_heads}'
        return s

    @property
    def target_metrics(self) -> float:
        '''
        Return the mean validation IoU from active heads.

        For each head if excluded classes are set, compute without them;
        otherwise compute from all present classes.
        '''
        if self.validation:
            return _track_metrics(
                self.validation.head_metrics,
                metric_name=self._metric_name,
                track_heads=self._track_heads,
                mtl_metrics=self.validation.mtl_metrics
            )
        return -float('inf')

    @property
    def inference_metrics(self) -> float:
        '''
        Return the mean inference IoU from active heads.

        For each head if excluded classes are set, compute without them;
        otherwise compute from all present classes.
        '''
        if self.inference:
            return _track_metrics(
                self.inference.head_metrics,
                metric_name=self._metric_name,
                track_heads=self._track_heads,
                mtl_metrics=self.inference.mtl_metrics
            )
        return -float('inf')

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return as a dictionary for serialization.'''
        return dataclasses.asdict(self)

    def track(
        self,
        metric_name: str,
        track_heads: dict[str, float] | None
    ):
        '''Tracking configuration'''

        # force IoU mode if there is only one active head (mtl metrics invalid)
        if self.validation and len(self.validation.head_metrics) == 1:
            metric_name = 'iou'
        object.__setattr__(self, '_metric_name', metric_name)
        object.__setattr__(self, '_track_heads', track_heads)

@dataclasses.dataclass
class TrainStepResults:
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
class ValStepResults:
    '''Evaluator aggregated epoch results.'''
    head_metrics: dict[str, AccumulatedMetrics] = field(default_factory=dict)
    mtl_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return as a dictionary for serialization.'''
        return {
            'head_metrics': {k: v.as_dict for k, v in self.head_metrics.items()},
            'mtl_metrics': dict(self.mtl_metrics) # as a shallow copy
        }

@dataclasses.dataclass
class InferStepResults:
    '''Containe for inference results.'''
    head_metrics: dict[str, AccumulatedMetrics] = field(default_factory=dict)
    mtl_metrics: dict[str, float] = field(default_factory=dict)
    infer_labels: dict[str, torch.Tensor] = field(default_factory=dict)
    infer_preds: dict[str, torch.Tensor] = field(default_factory=dict)
    infer_errors: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return as a dictionary for serialization.'''
        return {
            'head_metrics': {k: v.as_dict for k, v in self.head_metrics.items()},
            'mtl_metrics': dict(self.mtl_metrics) # as a shallow copy
        }

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
        _dict = dataclasses.asdict(self)
        _dict.pop('_locked') # exclude
        return _dict

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
def _track_metrics(
    head_metrics: dict[str, AccumulatedMetrics],
    *,
    metric_name: str,
    track_heads: dict[str, float] | None = None,
    mtl_metrics: dict[str, float] | None = None
) -> float:
    '''Track metrics as per configuration.'''

    match metric_name.lower():
        case 'gem':
            assert mtl_metrics and 'gem' in mtl_metrics # sanity
            return mtl_metrics['gem']

        case 'iou':
            if isinstance(track_heads, dict):
                m = {k: v for k, v in head_metrics.items() if k in track_heads}
                return _get_mean_iou(m, track_heads)

            if track_heads is None:
                m = next(iter(head_metrics.items())) # fall back to 1st head
                return _get_mean_iou({m[0]: m[1]}, None)

            raise ValueError(f'Invalid tracking head: {track_heads}')

        case _:
            raise ValueError(f'Invalid metric name: {metric_name}')

def _get_mean_iou(
    head_metrics: dict[str, AccumulatedMetrics],
    head_weights: dict[str, float] | None
) -> float:
    '''Mean IoU calculation helper.'''

    mean = 0.0
    # accumulate from all monitor heads
    for head, metrics in head_metrics.items():
        # get weight
        w = head_weights.get(head, 1.0) if head_weights else 1.0
        # pick iou - prefer iou from active classes if present
        mean += w * (metrics.ac_mean if metrics.ac_mean else metrics.mean)

    # get average ious
    mean /= max(1, len(head_metrics))

    return mean
