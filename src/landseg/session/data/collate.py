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
Batch collation for segmentation dataset items.

Provides a custom collate function used as `collate_fn` in
`torch.utils.data.DataLoader`. It handles the labeled/unlabeled
duality of `DatasetItem` batches and stacks domain tensors across
batch items.

Public APIs:
    - `batch`: collate a sequence of `DatasetItem` into a
      single batched `DatasetItem`.
'''


# standard imports
from __future__ import annotations
import typing
# third-party imports
import torch
# local imports
import landseg.session.contracts as contracts


# ----- public function
def batch(
    input_batch: typing.Sequence[contracts.DatasetItem]
) -> contracts.DatasetItem:
    '''
    Customized collate function to properly stack a batch.

    Contract per split:
      - Labeled: every y is [ps, ps] (long) -> stacked to [B, ps, ps]
      - Unlabeled: every y is empty tensor -> stacked to [B, 0] (long)
      - Domain: all items share the same keys; each stacks to [B, ...]
    '''
    # unpack batch items into separate lists
    xs, ys, ds = zip(*input_batch)

    # x is always stackable
    xs_out = torch.stack(xs, dim=0) # x -> [B, C, H, W]

    # determine if labeled or unlabeled batch from first item
    y0 = ys[0]
    labeled_batch = y0.numel() > 0

    if labeled_batch:
        # ensure all y match shape of first y
        exp_shape = y0.shape
        for i, y in enumerate(ys):
            if y.shape != exp_shape:
                raise ValueError(
                    f'inconsistent y shapes in batch at index {i}: '
                    f'expected {tuple(exp_shape)} but got {tuple(y.shape)}'
                )
        ys_out = torch.stack(ys, dim=0).long()
    else:
        # unlabeled/inference: all y must be empty tensors
        for i, y in enumerate(ys):
            if y.numel() != 0:
                raise ValueError(
                    f'mixed labeled/unlabeled batch: item {i} has non-empty y'
                )
        ys_out = torch.stack(ys, dim=0).long()

    # domain assumes consistent keys across batch
    dom_out = {} # -> dict[str, [B, V]] or dict[str, [B]]
    first_dom = ds[0]
    for key in first_dom.keys():
        dom_out[key] = torch.stack([d[key] for d in ds], dim=0)

    return xs_out, ys_out, dom_out
