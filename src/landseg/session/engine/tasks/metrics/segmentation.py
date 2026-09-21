# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Segmentation metrics and evaluation containers for prediction heads.

Provides incremental confusion matrix computation and derived IoU
metrics, alongside typed containers for multi-head validation.

Public APIs:
    - `ConfusionMatrix`: Incremental confusion matrix with IoU metrics.
    - `HeadMetrics`: Typed container mapping head names to confusion
      matrices.
    - `build_headmetrics`: Factory building per-head confusion matrices.
'''

# third-party imports
import torch
# local imports
import landseg.core as core
import landseg.session.engine.tasks.heads as heads


# ----- public classes
class ConfusionMatrix:
    '''
    Incremental confusion matrix with IoU metric computation.

    Accumulates predictions over batches and computes per-class and
    mean IoU, with optional hierarchical filtering and class exclusion
    applied during metric reporting.
    '''

    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        parent_class_1b: int | None,
        exclude_class_1b: tuple[int, ...] | None
    ):
        '''
        Initialize confusion matrix state and configuration.

        Args:
            num_classes:
                total number of classes for the prediction head.
            ignore_index:
                index to ignore in ground truth labels.
            parent_class_1b:
                optional parent class label (1-based) for gating.
            exclude_class_1b:
                optional tuple of class labels (1-based) to exclude.
        '''
        self.n_cls = num_classes
        self.ignore_index = ignore_index
        self.parent_class_1b = parent_class_1b
        self.exclude_class_1b = exclude_class_1b

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
            preds:
                model outputs of shape [B, C, H, W].
            targets:
                child labels of shape [B, H, W] (1-based).
            parent_raw_1b (kwarg):
                optional parent labels (1-based) for hierarchical
                gating.
        '''
        valid = targets != self.ignore_index

        parent_raw_1b = kwargs.get('parent_raw_1b')
        if parent_raw_1b is not None and self.parent_class_1b is not None:
            assert isinstance(parent_raw_1b, torch.Tensor)
            valid = valid & (parent_raw_1b == self.parent_class_1b)
        if valid.sum() == 0:
            return

        preds_0b = torch.argmax(preds, dim=1)
        t0 = targets[valid].to(torch.int64) - 1
        p = preds_0b[valid].to(torch.int64)

        t0 = t0.clamp(min=0, max=self.n_cls - 1)
        p = p.clamp(min=0, max=self.n_cls - 1)

        k = t0 * self.n_cls + p
        binc = torch.bincount(k, minlength=self.n_cls * self.n_cls)
        self.cm += binc.view(self.n_cls, self.n_cls)

    def compute(self) -> core.AccumulatedMetrics:
        '''Compute IoUs and return an accumulated metrics container.'''
        metrics = core.AccumulatedMetrics()
        metrics.cmatrix = self.cm.tolist()

        if self.cm.ndim != 2 or self.cm.shape[0] != self.cm.shape[1]:
            raise ValueError('Confusion matrix must be a square 2D tensor')

        tp = torch.diag(self.cm).float()
        fp = self.cm.sum(dim=0).float() - tp
        fn = self.cm.sum(dim=1).float() - tp
        dn = tp + fp + fn

        eps = torch.finfo(tp.dtype).eps
        iou = torch.where(dn > 0, tp / dn.clamp_min(eps), torch.zeros_like(dn))
        iou_list = iou.tolist()

        excld = self.exclude_class_1b
        if excld is not None and len(excld) > 0:
            if not all((1 <= idx <= self.n_cls) for idx in excld):
                raise IndexError('Exclude classes out of index range')
            activ = set(range(len(iou))) - set(x - 1 for x in excld)
        else:
            activ = ()

        activ_sum = 0.0
        for idx in range(len(iou)):
            metrics.ious[f'{idx + 1}'] = iou_list[idx]
            if idx in activ:
                metrics.ac_ious[f'{idx + 1}'] = iou_list[idx]
                activ_sum += iou_list[idx]

        v = dn > 0
        metrics.mean = iou[v].mean().item() if v.any() else 0.0
        metrics.ac_mean = activ_sum / len(activ) if activ else 0.0

        metrics.lock()
        return metrics

    def reset(self, device: str) -> None:
        '''Zero the confusion matrix and move to specified device.'''
        self.cm = self.cm.zero_().to(device)


class HeadMetrics:
    '''
    Typed wrapper around a mapping of heads to `ConfusionMatrix` objects.

    Provides key-based access to individual `ConfusionMatrix` instances
    and a typed container for passing head specs through the engine.
    '''

    def __init__(self, hmetrics: dict[str, ConfusionMatrix]):
        '''Initialize wrapper with head name mapping.'''
        self._hmetrics = hmetrics

    def __getitem__(self, key: str) -> ConfusionMatrix:
        return self._hmetrics[key]

    def __len__(self) -> int:
        return len(self._hmetrics)

    def as_dict(self) -> dict[str, ConfusionMatrix]:
        '''Return a shallow copy of the mapping as `dict[str, CM]`.'''
        return dict(self._hmetrics)


# ----- public functions
def build_headmetrics(
    headspecs: heads.HeadSpecs,
    *,
    ignore_index: int
) -> HeadMetrics:
    '''
    Construct `ConfusionMatrix` objects for each prediction head.

    Args:
        headspecs:
            structure describing each head's class count and gating.
        ignore_index:
            label index to ignore during metric updates.

    Returns:
        HeadMetrics:
            container mapping head names to initialized confusion
            matrices.
    '''
    out: dict[str, ConfusionMatrix] = {}
    for hname, hspec in headspecs.as_dict().items():
        out[hname] = ConfusionMatrix(
            num_classes=len(hspec.count),
            ignore_index=ignore_index,
            parent_class_1b=hspec.parent_cls,
            exclude_class_1b=hspec.exclude_cls,
        )

    return HeadMetrics(out)
