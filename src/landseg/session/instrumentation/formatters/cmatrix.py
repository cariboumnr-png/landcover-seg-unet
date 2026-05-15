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
Utilities for generating confusion matrices and evaluation summaries
from model inference results.

This module provides:
- Efficient computation of dense confusion matrices using PyTorch.
- Automatic class range inference and validation.
- Optional handling of ignored labels (e.g., segmentation masks).
- Markdown-formatted reporting, including:
  - Confusion matrix table
  - Per-class IoU and mean IoU (mIoU)

Designed for lightweight evaluation pipelines where both numeric
outputs and human-readable summaries are required.
'''

# third-party imports
import torch


# -------------------------------Public Function-------------------------------
def get_cmatrix(
    targets: torch.Tensor,
    preds: torch.Tensor,
    *,
    class_range: tuple[int, int] | None = None,
    class_names: list[str] | None = None,
    exclude_cls: tuple[int, ...] | None = None,
    ignore_index: int | None = None,
) -> tuple[torch.Tensor, str]:
    '''
    Generates a confusion matrix tensor and a Markdown table.

    Args:
        targets: Tensor of integer ground truth labels.
        preds: Tensor of integer predictions.
        class_names: List of string names for each class. If None,
            defaults to: ['Class 0', 'Class 1', ...]
        class_range: Inclusive class index range. If None, inferred from
            preds and targets.
        ignore_index: Label value to ignore (e.g. 255 for segmentation).

    Returns:
        tuple:
            - A 2D PyTorch tensor representing the confusion matrix.
            - A Markdown-formatted string of the confusion matrix table.
    '''

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    # remove ignored labels before inferring class range
    if ignore_index is not None:
        valid_mask = targets_flat != ignore_index
        preds_flat = preds_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]

    # infer class range from data if not provided
    if class_range is None:
        if targets_flat.numel() == 0:
            raise ValueError('No valid pixels except ignore_index.')
        class_min = int(min(preds_flat.min().item(), targets_flat.min().item()))
        class_max = int(max(preds_flat.max().item(), targets_flat.max().item()))
        class_range = (class_min, class_max)
    else:
        class_min, class_max = class_range

    if class_max < class_min:
        raise ValueError(f'Invalid class_range: {class_range}')

    num_classes = class_max - class_min + 1
    # generate default class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(class_min, class_max + 1)]

    # ensure names align with class range
    if len(class_names) != num_classes:
        raise ValueError(
            f'Length of class_names ({len(class_names)}) must match '
            f'the number of classes implied by class_range '
            f'({num_classes}).'
        )

    # compute confusion matrix
    cm_tensor = _compute_cm_tensor(
        preds,
        targets,
        class_range=class_range,
        ignore_index=ignore_index,
    )
    # generate markdown report
    report = _format_report(
        cm_tensor,
        class_names=class_names,
        class_min=class_min,
        exclude_cls=exclude_cls,
    )
    return cm_tensor, report

# ------------------------------private function-------------------------------
def _compute_cm_tensor(
    preds: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_range: tuple[int, int],
    ignore_index: int | None = None,
) -> torch.Tensor:
    '''Compute a dense confusion matrix using flattened indices.'''

    preds_flat = preds.view(-1).long()
    labels_flat = labels.view(-1).long()

    # ensure equal number of elements
    if preds_flat.shape != labels_flat.shape:
        raise ValueError(
            f'Shape mismatch: preds {preds.shape} and labels {labels.shape} '
            f'must have the same number of elements.'
        )

    # filter out ignored labels
    if ignore_index is not None:
        valid_mask = labels_flat != ignore_index
        preds_flat = preds_flat[valid_mask]
        labels_flat = labels_flat[valid_mask]

    class_min, class_max = class_range
    num_classes = class_max - class_min + 1

    # shift labels to zero-based indexing for bincount
    preds_flat = preds_flat - class_min
    labels_flat = labels_flat - class_min

    # validate that values lie within expected range
    if labels_flat.min() < 0 or labels_flat.max() >= num_classes:
        raise ValueError(f'labels out of valid range {class_range}')
    if preds_flat.min() < 0 or preds_flat.max() >= num_classes:
        raise ValueError(f'preds out of valid range {class_range}')

    # encode 2D indices into 1D:
    # index = (true_label * num_classes) + predicted_label
    indices = (num_classes * labels_flat) + preds_flat

    # count occurrences and reshape into square matrix
    cm_tensor = torch.bincount(
        indices,
        minlength=num_classes ** 2,
    )
    return cm_tensor.reshape(num_classes, num_classes)

def _format_report(
    cm_tensor: torch.Tensor,
    *,
    class_names: list[str],
    class_min: int,
    exclude_cls: tuple[int, ...] | None,
) -> str:
    '''Combine confusion matrix and IoU tables into a markdown report.'''

    cm = cm_tensor
    cm_table = _format_cm_table(cm, class_names)
    iou_table = _format_iou_table(
        cm=cm,
        class_names=class_names,
        class_min=class_min,
        exclude_cls=set(exclude_cls or ()),
    )
    return cm_table + '\n\n' + iou_table

def _format_cm_table(
    cm: torch.Tensor,
    class_names: list[str],
) -> str:
    '''Format confusion matrix as a Markdown table.'''

    header = '| True \\ Pred | ' + ' | '.join(class_names) + ' |'
    sep = '|' + '|'.join(['---'] * (len(class_names) + 1)) + '|'
    rows = [header, sep]
    # each row corresponds to a ground-truth class
    for i, name in enumerate(class_names):
        row = [f'**{name}**']
        row += [str(cm[i, j].item())
                for j in range(len(class_names))]
        rows.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(rows)

def _format_iou_table(
    *,
    cm: torch.Tensor,
    class_names: list[str],
    class_min: int,
    exclude_cls: set[int],
) -> str:
    '''Compute per-class IoU and format as a Markdown table.'''

    # true positives, false positives, false negatives
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    denom = tp + fp + fn

    # table rows
    rows = [
        '| Class | IoU | Note |',
        '|---|---|---|',
    ]

    valid_ious = []
    for idx, name in enumerate(class_names):
        class_id = class_min + idx

        # skip excluded classes (e.g., background)
        if class_id in exclude_cls:
            rows.append(f'| {name} | - | excluded |')
            continue
        # no samples → undefined IoU
        if denom[idx] == 0:
            rows.append(f'| {name} | N/A | no samples |')
            continue
        
        # compute ious
        iou = tp[idx].float() / denom[idx].float()
        valid_ious.append(iou)
        # note on TP
        if tp[idx] == 0:
            rows.append(f'| {name} | {iou.item():.4f} | no TP |')
        else:
            rows.append(f'| {name} | {iou.item():.4f} | |')

    # mean IoU over valid classes
    miou = (torch.stack(valid_ious).mean().item() if valid_ious else 0.0)
    rows.append(f'| **mIoU** | **{miou:.4f}** | |')
    return '\n'.join(rows)
