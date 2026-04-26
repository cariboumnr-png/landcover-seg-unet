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
Validation utilities: confusion matrix accumulation and IoU metrics.

Provides:
    - Dataclass container for accumlated metrics reporting.
    - ConfusionMatrix: incremental updates and IoU/mean IoU computation,
      with optional hierarchical gating and class exclusion.
'''

# standard imports
import dataclasses
import typing
# third-party imports
import torch

# aliases
field = dataclasses.field

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class ConfusionMatricConfig:
    '''Configuration for confusion-matrix computation.'''
    num_classes: int
    ignore_index: int
    parent_class_1b: int | None
    exclude_class_1b: tuple[int, ...] | None

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

    def __setattr__(self, key, value):
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
        str_list.append('Mean IoU (all):\t' + m)
        c = '|'.join(f'cls{k}={v:.4f}' for k, v in self.ious.items())
        str_list.append('Class IoU (all):\t' + c)
        # s = '|'.join(f'cls{k}={v}' for k, v in mm['support'].items())
        # text.append('Class support (all):\t' + s)
        # subset of active classes (if not None)
        if bool(self.ac_mean):
            m = f'{self.ac_mean:.4f}'
            str_list.append('Mean IoU (active):\t' + m)
            c = '|'.join(f'cls{k}={v:.4f}' for k, v in self.ac_ious.items())
            str_list.append('Class IoU (active):\t' + c)
            # s = '|'.join(f'cls{k}={v}' for k, v in mm['ac_support'].items())
            # text.append('Class support (active):\t' + s)
        # return text lines
        return str_list

    def lock(self):
        '''Lock object via __setattr__ blocking.'''
        self._locked = True

# --------------------------------Public  Class--------------------------------
class ConfusionMatrix:
    '''
    Incremental confusion matrix with IoU computation.

    Supports:
        - Hierarchical gating via a parent-class filter.
        - Excluding classes when reporting active-class IoUs.
    '''

    def __init__(self, config: ConfusionMatricConfig):
        '''
        Initialize the confusion matrix and configuration.

        Args:
            config:
                Dictionary with
                - 'num_classes': total number of classes (int)
                - 'ignore_index': label to ignore (int or None)
                - 'parent_class_1b': parent class (1-based) to gate on
                    when provided (int or None)
                - 'exclude_class_1b': classes (1-based) to exclude from
                    active-class metrics (list[int] or None)
        '''

        # assign attributes
        self.n_cls = config.num_classes
        self.ignore_index = config.ignore_index
        self.parent_class_1b = config.parent_class_1b
        self.exclude_class_1b = config.exclude_class_1b

        # set up running confusion matrix (start with zeros)
        h, w = self.n_cls, self.n_cls
        self.cm = torch.zeros((h, w), dtype=torch.int64)

        # init metrics data class
        self.metrics = AccumulatedMetrics() # will lock after compute()

    @torch.no_grad()
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> None:

        '''
        Update confusion matrix with a new batch.

        Args:
            preds: Model outputs of shape [B, C, H, W].
            targets: Child labels of shape [B, H, W], 1-based with
                ignore index.
            parent_raw_1b (kwarg): Optional parent labels (1-based). If
                provided and `parent_class_1b` is set, only those pixels
                are counted.
        '''

        # get valid pixels (mask off ignored)
        valid = targets != self.ignore_index

        # optional hierarchical gating using raw parent labels (1-based)
        parent_raw_1b = kwargs.get('parent_raw_1b')
        if parent_raw_1b is not None and self.parent_class_1b is not None:
            assert isinstance(parent_raw_1b, torch.Tensor)
            valid = valid & (parent_raw_1b == self.parent_class_1b)
        if valid.sum() == 0:
            return

        # get prediction for the batch
        preds_0b = torch.argmax(preds, dim=1) # [B, H, W] - along S (slice)

        # shift child target to 0-based for bincount indexing
        t0 = targets[valid].to(torch.int64) - 1 # because target_1b is 1..C
        p = preds_0b[valid].to(torch.int64) # preds should already be 0..C-1

        # safety: drop any accidental negatives, in case ignore slipped through
        # and clamp to the expected class range.
        t0 = t0.clamp(min=0, max=self.n_cls - 1)
        p = p.clamp(min=0, max=self.n_cls - 1)

        # flatten pair (true, pred) to unique index: t0 * C + p
        k = t0 * self.n_cls + p
        binc = torch.bincount(k, minlength=self.n_cls * self.n_cls)

        # accumulate in place
        self.cm += binc.view(self.n_cls, self.n_cls)

    def compute(self) -> None:
        '''
        Compute per-class IoU and aggregate means.

        Populates a `_Metrics` typed dict:
            - 'mean': Mean IoU over all classes (float)
            - 'ious': Dict of per-class IoU values {str: float}
            - 'support': Dict of per-class supports {str: int}
            - 'ac_mean': Mean IoU over active classes (float)
            - 'ac_ious': Dict of IoU per active class {str: float}
            - 'ac_support': Dict of support per active class {str: int}

        '''
        if self.cm.ndim != 2 or self.cm.shape[0] != self.cm.shape[1]:
            raise ValueError('Confusion matrix must be a square 2D tensor')

        # cm[i, j]: true class i predicted as j
        tp = torch.diag(self.cm).float()     # true positive
        fp = self.cm.sum(dim=0).float() - tp # false positive
        fn = self.cm.sum(dim=1).float() - tp # false negative
        # union per class as the denominator
        dn = tp + fp + fn

        # safe divide for iou
        eps = torch.finfo(tp.dtype).eps
        iou = torch.where(dn > 0, tp / dn.clamp_min(eps), torch.zeros_like(dn))
        iou_list = iou.tolist()
        # per-class supports (number of true pixels)
        support = self.cm.sum(dim=1).tolist()

        # parse from to-exclude classes if provided
        excld = self.exclude_class_1b # 1-based
        if excld is not None and len(excld) > 0:
            if not all((1 <= idx <= self.n_cls) for idx in excld):
                raise IndexError('Exclude classes out of index range')
            activ = set(range(len(iou))) - set(x - 1 for x in excld) # 0-based
        else:
            activ = ()

        # iterate ious and split into groups
        activ_sum = 0.0
        for idx in range(len(iou)):
            self.metrics.ious[f'{idx + 1}'] = iou_list[idx]
            self.metrics.support[f'{idx + 1}'] = support[idx]
            # if class is not excluded
            if idx in activ:
                # 1-based class label
                self.metrics.ac_ious[f'{idx + 1}'] = iou_list[idx]
                self.metrics.ac_support[f'{idx + 1}'] = support[idx]
                activ_sum += iou_list[idx]

        # set metrics outputs
        v = dn > 0 # mean IoU over classes with denom > 0
        self.metrics.mean = iou[v].mean().item() if v.any() else 0.0
        self.metrics.ac_mean = activ_sum / len(activ) if activ else 0.0

        # lock metrics
        self.metrics.lock()

    def reset(self, device: str) -> None:
        '''Zero the confusion matrix and move to specified device.'''

        self.cm = self.cm.zero_().to(device)
