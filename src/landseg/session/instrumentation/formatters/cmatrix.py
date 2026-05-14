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
def get_confusion_matrix_and_table(
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str] | None = None,
    num_classes: int | None = None
) -> tuple[torch.Tensor, str]:
    '''
    Generates a confusion matrix tensor and a Markdown table.

    Args:
        preds: Tensor of integer predictions.
        labels: Tensor of integer ground truth labels.
        class_names: List of string names for each class. If None,
            defaults to ['Class 0', 'Class 1', ...].
        num_classes: Total number of classes. If None, inferred from the
            maximum value in preds and labels.

    Returns:
        tuple:
            - A 2D PyTorch tensor representing the confusion matrix.
            - A Markdown-formatted string of the confusion matrix table.
    '''
    # infer num of classes if not provided
    if num_classes is None:
        if class_names is not None:
            num_classes = len(class_names)
        else:
            num_classes = int(max(preds.max().item(), labels.max().item())) + 1

    # get default class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    # validate class_names length matches num_classes
    if len(class_names) != num_classes:
        raise ValueError(
            f'Length of class_names ({len(class_names)}) must match'
            f'num_classes ({num_classes}).'
        )

    # 4compute outputs
    cm_tensor = _compute_cm_tensor(preds, labels, num_classes)
    cm_markdown = _format_cm_markdown(cm_tensor, class_names)

    return cm_tensor, cm_markdown

# ------------------------------private  function------------------------------
def _compute_cm_tensor(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    '''Computes a dense confusion matrix tensor.'''

    # flatten tensors
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)
    # sanity
    if preds_flat.shape != labels_flat.shape:
        raise ValueError(
            f'Shape mismatch: preds {preds.shape} and labels {labels.shape} '
            f'must have the same number of elements.'
        )

    # flat indexing: (True * num_classes) + Pred
    indices = (num_classes * labels_flat) + preds_flat

    # calculate bincount and reshape to 2D
    cm_tensor = torch.bincount(indices, minlength=num_classes**2)
    return cm_tensor.reshape(num_classes, num_classes)

def _format_cm_markdown(
    cm_tensor: torch.Tensor,
    class_names: list[str]
) -> str:
    '''Formats a 2D confusion matrix into a Markdown table.'''

    # headers
    num_classes = len(class_names)
    header = '| True \\ Pred | ' + ' | '.join(class_names) + ' |'
    separator = '|' + '|'.join(['---'] * (num_classes + 1)) + '|'

    # table rows
    rows = [header, separator]
    for i in range(num_classes):
        row_data = [f'**{class_names[i]}**']
        for j in range(num_classes):
            row_data.append(str(cm_tensor[i, j].item()))
        rows.append('| ' + ' | '.join(row_data) + ' |')
    return '\n'.join(rows)
