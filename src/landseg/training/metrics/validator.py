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

'''Metrics config validation.'''

# standard imports
import typing

class ConfusionMatrixConfig(typing.TypedDict):
    '''Config dict for confusion matrix compute.'''
    num_classes: int
    ignore_index: int
    parent_class_1b: int | None
    exclude_class_1b: tuple[int, ...] | None

def is_cm_config(
        cfg: dict[str, int | None]
    ) -> typing.TypeGuard[ConfusionMatrixConfig]:
    '''Validate confusion matrix calc config dict.'''

    def _is_parent_class(v):
        return isinstance(v, int) or v is None

    def _is_exlcude_class(v):
        return (
            isinstance(v, tuple) and len(v) == 0 or
            (
                isinstance(v, tuple) and len(v) > 0 and
                all(isinstance(x, int) for x in v)
            )
        )

    return(
        isinstance(cfg, dict) and
        isinstance(cfg.get('num_classes'), int) and
        isinstance(cfg.get('ignore_index'), int) and
        _is_parent_class(cfg.get('parent_class_1b')) and
        _is_exlcude_class(cfg.get('exclude_class_1b'))
    )
