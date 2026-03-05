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

'''Validation metrics calculation functions for trainer.'''

# local imports
import landseg.training.common as common
import landseg.training.metrics as metrics

class HeadMetrics:
    '''
    Typed wrapper around a mapping of heads to `ConfusionMatrix` objects.

    This class provides:
    - key-based access to individual `ConfusionMatrix` instances.
    - a stable, strongly-typed container for passing head specs through
    the codebase.

    It is *not* a full `dict` replacement. To work with the underlying
    mapping directly, use method: `as_dict()`.
    '''

    def __init__(self, specs: dict[str, metrics.ConfusionMatrix]):
        self._specs = specs

    def __getitem__(self, key: str) -> common.MetricLike:
        return self._specs[key]

    def __len__(self) -> int:
        return len(self._specs)

    def as_dict(self) -> dict[str, common.MetricLike]:
        '''Return a shallow copy of the mapping as `dict[str, Spec]`.'''
        return dict(self._specs)

def build_headmetrics(
        headspecs: common.HeadSpecsLike,
        ignore_index: int
    ) -> HeadMetrics:
    '''Generate concreate head metric classes indexed by head name.'''

    out: dict[str, metrics.ConfusionMatrix] = {}
    # iterate through configs
    for hname, hspec in headspecs.as_dict().items():
        # attach to output list
        config = {
            'num_classes': hspec.num_classes,
            'ignore_index': ignore_index,
            'parent_class_1b': hspec.parent_cls,
            'exclude_class_1b': hspec.exclude_cls
        }
        out[hname] = metrics.ConfusionMatrix(config)
    return HeadMetrics(out)
