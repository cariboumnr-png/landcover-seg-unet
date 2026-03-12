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
Training phase
'''

# standard imports
from __future__ import annotations
import dataclasses

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class Phase:
    '''doc'''
    name: str
    num_epochs: int
    heads: HeadsConifg
    la_scheme: LogitAdjustScheme
    lr_scale: float = 1.0
    finished: bool = False

    def __str__(self) -> str:
        return '\n'.join([
            f'- Phase Name:\t{self.name}',
            f'- Max Epochs:\t{self.num_epochs}',
            str(self.heads),
            str(self.la_scheme),
            f'- LR Scale:\t{self.lr_scale}'
        ])

@dataclasses.dataclass
class HeadsConifg:
    '''Phase level heads-related config.'''
    active_heads: list[str]
    frozen_heads: list[str] | None
    excluded_cls: dict[str, list[int]] | None

    def __str__(self) -> str:
        indent: int=2
        s = ' ' * indent
        return f'\n{s}'.join([
            '- Heads Specs',
            f'- Active Heads:\t{self.active_heads}',
            f'- Frozen Heads:\t{self.frozen_heads}',
            f'- Excld. Class:\t{self.excluded_cls}',
        ])

@dataclasses.dataclass
class LogitAdjustScheme:
    '''Logit adjustment scheme.'''
    logit_adjust_alpha: float
    enable_train_logit_adjustment: bool
    enable_val_logit_adjustment: bool
    enable_test_logit_adjustment: bool

    def __str__(self) -> str:
        indent: int=2
        s = ' ' * indent
        return f'\n{s}'.join([
            '- Logit Adjustment',
            f'- Global Alpha:\t{self.logit_adjust_alpha:.2f}',
            f'- Training Stage:\t{self.enable_train_logit_adjustment}',
            f'- Validation Stage:\t{self.enable_val_logit_adjustment}',
            f'- Inference Stage:\t{self.enable_test_logit_adjustment}',
        ])
