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
Data contracts for batch processing and multi-head tensors.
Defines core type aliases for data pipeline exchange objects between
data loaders, batch engines, and task regularizers.
Public APIs:
    - `DatasetItem`: sample or batch tuple (x, y, domain).
    - `TensorDict`: mapping of string names to PyTorch tensors.
'''

# standard imports
from __future__ import annotations
import typing
# third-party imports
import torch


# ----- typing aliases
TensorDict: typing.TypeAlias = dict[str, torch.Tensor]

DatasetItem: typing.TypeAlias = tuple[torch.Tensor, torch.Tensor, TensorDict]
'''
A tuple from one sample of the dataset: x (always present), y (can be
a placeholder during inference, e.g., `torch.Tensor([1])`) and domain
(always present but can be empty).
'''
