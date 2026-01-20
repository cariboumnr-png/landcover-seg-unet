'''Training metrics related functions.'''

from __future__ import annotations
# standard imports
import typing
# third-party imports
import torch
# local imports
import training.metrics

class ConfusionMatrix:
    '''
    Computes IoU metrics from predictions and targets.

    Supports hierarchical gating via a parent class filter and optional
    exclusion of classes when reporting 'active-class' metrics.
    '''

    def __init__(
            self,
            config: dict[str, int | None],
        ):
        '''doc'''

        # type guard on input config
        if not training.metrics.is_cm_config(config):
            raise ValueError(f'Invalid config: {config}')

        # assign attributes
        self.n_cls = config['num_classes']
        self.ignore_index = config['ignore_index']
        self.parent_class_1b = config['parent_class_1b']
        self.exclude_class_1b = config['exclude_class_1b']

        # set up running confusion matrix (start with zeros)
        h, w = self.n_cls, self.n_cls
        self.cm = torch.zeros((h, w), dtype=torch.int64)

        # store result as a class attribute
        self._metrics: _Metrics = {
            'mean': 0.0,
            'ious': {},
            'support': {},
            'ac_mean': 0.0,
            'ac_ious': {},
            'ac_support': {}
        }

    @torch.no_grad()
    def update(
            self,
            p0: torch.Tensor,
            t1: torch.Tensor,
            **kwargs
        ) -> None:

        '''
        In-place update confusion matrix w/ optional hierarchical gating.

        - `target_1b`: child head labels in 1-based + ignore_index.
        - `preds`: prediction labels in 0-based [0..C-1].
        - If `parent_raw_1b` (kwarg) are provided and
            `self.parent_class_1b`, is not None, only pixels where
            `parent==parent_class_1b` are counted.
        - `self.cm` is updated in place.

        Args:
            logits: [B, C, H, W] model outputs.
            target_1b: [B, H, W] 1-based class labels (+ ignore_index).
        '''

        # get valid pixels (mask off ignored)
        valid = t1 != self.ignore_index

        # optional hierarchical gating using raw parent labels (1-based)
        parent_raw_1b = kwargs.get('parent_raw_1b')
        if parent_raw_1b is not None and self.parent_class_1b is not None:
            assert isinstance(parent_raw_1b, torch.Tensor)
            valid = valid & (parent_raw_1b == self.parent_class_1b)
        if valid.sum() == 0:
            return

        # get prediction for the batch
        preds_0b = torch.argmax(p0, dim=1) # [B, H, W] - along S (slice)

        # shift child target to 0-based for bincount indexing
        t0 = t1[valid].to(torch.int64) - 1 # because target_1b is 1..C
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

        Returns:
            dict:
            - 'mean': Mean IoU over all classes (float)
            - 'ious': Dict of per-class IoU values {str: float}
            - 'support': Dict of per-class supports {str: int}
            - 'ac_mean': Mean IoU over active (non-excluded) classes (float)
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
            self._metrics['ious'][f'{idx + 1}'] = iou_list[idx]
            self._metrics['support'][f'{idx + 1}'] = support[idx]
            # if class is not excluded
            if idx in activ:
                # 1-based class label
                self._metrics['ac_ious'][f'{idx + 1}'] = iou_list[idx]
                self._metrics['ac_support'][f'{idx + 1}'] = support[idx]
                activ_sum += iou_list[idx]

        # set metrics outputs
        v = dn > 0 # mean IoU over classes with denom > 0
        self._metrics['mean'] = iou[v].mean().item() if v.any() else 0.0
        self._metrics['ac_mean'] = activ_sum / len(activ) if activ else 0.0

    def reset(self, device: str) -> None:
        '''Zero out the confusion matrix.'''

        self.cm = self.cm.zero_().to(device)

    @property
    def metrics_dict(self) -> dict[str, typing.Any]:
        '''Metrics as a plain dict.'''

        return dict(self._metrics)

    @property
    def metrics_text(self) -> list[str]:
        '''Printer friendly text presentation of the metrics.'''

        mm = self._metrics
        text = []
        # all classes
        m = f"{mm['mean']:.4f}"
        text.append('Mean IoU (all):\t' + m)
        c = '|'.join(f'cls{k}={v:.4f}' for k, v in mm['ious'].items())
        text.append('Class IoU (all):\t' + c)
        # s = '|'.join(f'cls{k}={v}' for k, v in mm['support'].items())
        # text.append('Class support (all):\t' + s)
        # subset of active classes (if not None)
        if self.exclude_class_1b:
            m = f"{mm['ac_mean']:.4f}"
            text.append('Mean IoU (active):\t' + m)
            c = '|'.join(f'cls{k}={v:.4f}' for k, v in mm['ac_ious'].items())
            text.append('Class IoU (active):\t' + c)
            # s = '|'.join(f'cls{k}={v}' for k, v in mm['ac_support'].items())
            # text.append('Class support (active):\t' + s)
        # return text lines
        return text

class _Metrics(typing.TypedDict):
    '''Typed dict for metrics.'''
    mean: float
    ious: dict[str, float]
    support: dict[str, int]
    ac_mean: float
    ac_ious: dict[str, float]
    ac_support: dict[str, int]
