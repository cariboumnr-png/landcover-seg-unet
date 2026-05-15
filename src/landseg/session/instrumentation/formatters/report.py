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
Utilities for generating Markdown evaluation reports from confusion
matrices.

This module provides:
- Markdown rendering of confusion matrices.
- Per-class intersection-over-union (IoU) computation.
- Mean IoU (mIoU) summaries.
- Optional exclusion of classes from IoU aggregation.
- Support for confusion matrices stored as PyTorch tensors or nested
  Python lists.

Confusion matrix convention:
- Rows correspond to ground-truth (true) classes.
- Columns correspond to predicted classes.

Designed for lightweight evaluation pipelines where both machine-friendly
numeric outputs and human-readable reports are required.
'''

# third-party imports
import torch

# -------------------------------Public Function-------------------------------
def report_iou(
    confusion_matrix: torch.Tensor | list[list[int]],
    *,
    class_names: list[str],
    exclude_cls: tuple[int, ...] | None,
    class_idx_base: int = 1,
) -> str:
    '''
    Generate a Markdown report containing:
    - a confusion matrix table,
    - per-class IoU values,
    - mean IoU (mIoU).

    Args:
        confusion_matrix:
            Dense confusion matrix with shape (C, C), where rows are
            ground-truth classes and columns are predicted classes.
            May be provided as either:
            - a PyTorch tensor, or
            - a nested Python list.

        class_names:
            Display names for classes in confusion matrix order.

        exclude_cls:
            Class IDs to exclude from IoU aggregation and reporting.
            Typically used for background or ignored classes.

        class_idx_base:
            Base value used to map confusion matrix row indices to
            external class IDs.

            Example:
                class_idx_base=1 maps:
                    row 0 -> class ID 1
                    row 1 -> class ID 2

    Returns:
        Markdown-formatted evaluation report.
    '''

    if isinstance(confusion_matrix, list):
        cm = torch.tensor(confusion_matrix)
    else:
        cm = confusion_matrix

    cm_table = _format_cm_table(cm, class_names)
    iou_table = _format_iou_table(
        cm,
        class_names,
        class_idx_base=class_idx_base,
        exclude_cls=set(exclude_cls or ()),
    )
    return cm_table + '\n\n' + iou_table

def _format_cm_table(
    cm: torch.Tensor,
    class_names: list[str],
) -> str:
    '''Format confusion matrix as a Markdown table.'''

    # rows -> ground-truth
    # Columns -> predicted
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
    cm: torch.Tensor,
    class_names: list[str],
    *,
    exclude_cls: set[int],
    class_idx_base: int = 1,
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
        class_id = class_idx_base + idx

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
