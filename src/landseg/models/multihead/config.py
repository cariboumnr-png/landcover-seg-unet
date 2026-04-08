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

'''Multihead model configuration dataclasses.'''

from __future__ import annotations
# standard imports
import dataclasses
import typing

#
@dataclasses.dataclass
class BackboneConfig:
    '''General configuration'''
    body: str
    base_ch: int
    conv_params: dict[str, typing.Any]

# -------------------------model general configuration-------------------------
@dataclasses.dataclass
class ModelConfig:
    '''General configuration'''
    in_ch: int
    logit_adjust: dict[str, list[float]]
    heads_w_counts: dict[str, list[int]]

# ----------------------model conditioning configuration----------------------
@dataclasses.dataclass
class ConditioningConfig:
    '''Conditioning configuration.'''

    mode: str | None        # mode ['concat', 'film', 'hybrid']
    domain_ids_num: int     # id categories
    domain_vec_dim: int     # vector dims
    concat: ConcatConfig    # Concat
    film: FilmConfig        # FiLM

@dataclasses.dataclass
class ConcatConfig:
    '''Concatenation adapter configuration.'''
    out_dim: int
    use_ids: bool
    use_vec: bool
    use_mlp: bool

@dataclasses.dataclass
class FilmConfig:
    '''FiLM conditioner configuration'''
    embed_dim: int
    use_ids: bool
    use_vec: bool
    hidden: int
