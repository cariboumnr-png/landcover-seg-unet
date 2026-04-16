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

'''Project-wide type aliases and lazy imports for type checking.'''

# standard imports
import typing
# third-party imports
import torch

# batch context
Tensor: typing.TypeAlias = torch.Tensor

TorchDict: typing.TypeAlias = dict[str, Tensor]
'''
A tyical string-indexed torch Tensor dictionary.
'''

DatasetItem: typing.TypeAlias = tuple[Tensor, Tensor, TorchDict]
'''
A tuple from one sample of the dataset: x (always present), y (can be
a placeholder during inference, e.g., `torch.Tensor([1])`) and domain
(always present but can be empty).
'''

DatasetBatch: typing.TypeAlias = typing.Sequence[DatasetItem]
'''
A collection of `DatasetItem` objects.
'''
