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
Confusion matrix and IoU metrics for segmentation evaluation.

Provides incremental confusion matrix updates and derived metrics,
including per-class IoU and mean IoU, with support for ignore-index
handling, hierarchical gating, and class exclusion.

These components are used during validation and inference to
accumulate and summarize prediction performance.
'''

# standard imports
import dataclasses
# third-party imports
import torch
# local imports
import landseg.core as core

# aliases
field = dataclasses.field

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class ConfusionMatricConfig:
    '''Configuration for confusion-matrix construction and computation.'''
    num_classes: int
    ignore_index: int
    parent_class_1b: int | None
    exclude_class_1b: tuple[int, ...] | None

# --------------------------------Public  Class--------------------------------
class ConfusionMatrix:
    '''
    Incremental confusion matrix with IoU metric computation.

    Accumulates predictions over batches and computes per-class and
    mean IoU, with optional hierarchical filtering and class exclusion
    applied during metric reporting.
    '''

    def __init__(self, config: ConfusionMatricConfig):
        '''
        Initialize confusion matrix state and configuration.

        Args:
            config:
                Configuration specifying number of classes, ignore index,
                optional parent-class gating, and class exclusions for
                metric reporting.

        Notes:
            - Internal matrix is initialized to zeros and updated\
              incrementally during execution.
            - Metrics are stored in an accumulated container and\
              finalized after computation.
        '''

        # assign attributes
        self.n_cls = config.num_classes
        self.ignore_index = config.ignore_index
        self.parent_class_1b = config.parent_class_1b
        self.exclude_class_1b = config.exclude_class_1b

        # set up running confusion matrix (start with zeros)
        h, w = self.n_cls, self.n_cls
        self.cm = torch.zeros((h, w), dtype=torch.int64)

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

    def compute(self) -> core.AccumulatedMetrics:
        '''Compute IoUs and return a `core.AccumulatedMetrics` container.'''

        # init metrics data class
        metrics = core.AccumulatedMetrics()
        metrics.cmatrix = self.cm.tolist() # for serialization

        # sanity
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

        # parse from to-exclude classes if provided
        excld = self.exclude_class_1b # 1-based
        if excld is not None and len(excld) > 0:
            if not all((1 <= idx <= self.n_cls) for idx in excld):
                raise IndexError('Exclude classes out of index range')
            activ = set(range(len(iou))) - set(x - 1 for x in excld) # 0-based
        else:
            activ = ()

        # iterate IoUs (all class & active class)
        activ_sum = 0.0
        for idx in range(len(iou)):
            metrics.ious[f'{idx + 1}'] = iou_list[idx]
            # if class is not excluded
            if idx in activ:
                # 1-based class label
                metrics.ac_ious[f'{idx + 1}'] = iou_list[idx]
                activ_sum += iou_list[idx]
        # mean IoUs
        v = dn > 0 # mean IoU over classes with denom > 0
        metrics.mean = iou[v].mean().item() if v.any() else 0.0
        metrics.ac_mean = activ_sum / len(activ) if activ else 0.0

        # lock metrics and return
        metrics.lock()
        return metrics

    def reset(self, device: str) -> None:
        '''Zero the confusion matrix and move to specified device.'''

        self.cm = self.cm.zero_().to(device)
