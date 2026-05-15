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

'''Confusion matrix utilities for inference results'''

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

    # apply ignore mask before inferring class range
    if ignore_index is not None:
        valid_mask = targets_flat != ignore_index
        preds_flat = preds_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]

    # infer class range if not provided
    if class_range is None:
        if targets_flat.numel() == 0:
            raise ValueError('No valid pixels except ignore_index.')

        class_min = int(min(preds_flat.min().item(), targets_flat.min().item()))
        class_max = int(max( preds_flat.max().item(), targets_flat.max().item()))
        class_range = (class_min, class_max)
    else:
        class_min, class_max = class_range
    if class_max < class_min:
        raise ValueError(f'Invalid class_range: {class_range}')
    num_classes = class_max - class_min + 1

    # default class names
    if class_names is None:
        class_names = [f'Class {i}' for i in range(class_min, class_max + 1)]

    # validate class_names
    if len(class_names) != num_classes:
        raise ValueError(
            f'Length of class_names ({len(class_names)}) must match '
            f'the number of classes implied by class_range '
            f'({num_classes}).'
        )

    # compute confusion matrix tensor and markdown report
    cm_tensor = _compute_cm_tensor(
        preds,
        targets,
        class_range=class_range,
        ignore_index=ignore_index,
    )
    report = _format_report(
        cm_tensor=cm_tensor,
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
    '''Computes a dense confusion matrix tensor.'''

    preds_flat = preds.view(-1).long()
    labels_flat = labels.view(-1).long()
    if preds_flat.shape != labels_flat.shape:
        raise ValueError(
            f'Shape mismatch: preds {preds.shape} and labels {labels.shape} '
            f'must have the same number of elements.'
        )

    # remove ignored labels
    if ignore_index is not None:
        valid_mask = labels_flat != ignore_index
        preds_flat = preds_flat[valid_mask]
        labels_flat = labels_flat[valid_mask]

    class_min, class_max = class_range
    num_classes = class_max - class_min + 1

    # normalize to zero-based indexing
    preds_flat = preds_flat - class_min
    labels_flat = labels_flat - class_min

    # validate ranges
    if labels_flat.min() < 0 or labels_flat.max() >= num_classes:
        raise ValueError(f'labels out of valid range {class_range}')
    if preds_flat.min() < 0 or preds_flat.max() >= num_classes:
        raise ValueError(f'preds out of valid range {class_range}')

    # flat indexing: (True * num_classes) + Pred
    indices = (num_classes * labels_flat) + preds_flat
    cm_tensor = torch.bincount(indices, minlength=num_classes ** 2)
    return cm_tensor.reshape(num_classes, num_classes)

def _format_report(
    cm_tensor: torch.Tensor,
    class_names: list[str],
    class_min: int,
    exclude_cls: tuple[int, ...] | None,
) -> str:

    cm = cm_tensor

    cm_table = _format_cm_table(cm, class_names)

    iou_table = _format_iou_table(
        cm=cm,
        class_names=class_names,
        class_min=class_min,
        exclude_cls=set(exclude_cls or ()),
    )

    return cm_table + "\n\n" + iou_table

def _format_cm_table(cm: torch.Tensor, class_names: list[str]) -> str:
    header = "| True \\ Pred | " + " | ".join(class_names) + " |"
    sep = "|" + "|".join(["---"] * (len(class_names) + 1)) + "|"

    rows = [header, sep]

    for i, name in enumerate(class_names):
        row = [f"**{name}**"]
        row += [str(cm[i, j].item()) for j in range(len(class_names))]
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows)

def _format_iou_table(
    cm: torch.Tensor,
    class_names: list[str],
    class_min: int,
    exclude_cls: set[int],
) -> str:

    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    denom = tp + fp + fn

    rows = [
        "| Class | IoU | Note |",
        "|---|---|---|",
    ]

    valid_ious = []

    for idx, name in enumerate(class_names):
        class_id = class_min + idx

        if class_id in exclude_cls:
            rows.append(f"| {name} | - | excluded |")
            continue

        if denom[idx] == 0:
            rows.append(f"| {name} | N/A | no samples |")
            continue

        iou = tp[idx].float() / denom[idx].float()
        valid_ious.append(iou)

        note = "no TP" if tp[idx] == 0 else ""
        rows.append(f"| {name} | {iou.item():.4f} | {note} |")

    miou = (
        torch.stack(valid_ious).mean().item()
        if valid_ious else 0.0
    )

    rows.append(f"| **mIoU** | **{miou:.4f}** | |")

    return "\n".join(rows)
