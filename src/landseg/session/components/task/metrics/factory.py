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
Factory utilities for constructing per-head validation metrics.

Provides:
    - HeadMetrics: typed container mapping head names to ConfusionMatrix
      instances.
    - build_headmetrics: factory that builds per-head ConfusionMatrix
      objects from head specifications.

Used by the trainer to compute IoU-based metrics for each prediction head.
'''

# local imports
import landseg.session.components.task as task
import landseg.session.components.task.metrics as metrics

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

    def __init__(self, hmetrics: dict[str, metrics.ConfusionMatrix]):
        self._hmetrics = hmetrics

    def __getitem__(self, key: str) -> metrics.ConfusionMatrix:
        return self._hmetrics[key]

    def __len__(self) -> int:
        return len(self._hmetrics)

    def as_dict(self) -> dict[str, metrics.ConfusionMatrix]:
        '''Return a shallow copy of the mapping as `dict[str, CM]`.'''
        return dict(self._hmetrics)

def build_headmetrics(
    headspecs: task.HeadSpecs,
    *,
    ignore_index: int
) -> HeadMetrics:
    '''
    Construct ConfusionMatrix objects for each prediction head.

    Args:
        headspecs: Structure describing each head's number of classes,
            optional parent-class gating, and excluded classes.
        ignore_index: Label index to ignore during metric updates.

    Returns:
        A HeadMetrics container, mapping head names to initialized
        ConfusionMatrix instances.
    '''

    out: dict[str, metrics.ConfusionMatrix] = {}
    # iterate through configs
    for hname, hspec in headspecs.as_dict().items():
        # attach to output list
        config = metrics.ConfusionMatricConfig(
            num_classes=hspec.num_classes,
            ignore_index=ignore_index,
            parent_class_1b=hspec.parent_cls,
            exclude_class_1b=hspec.exclude_cls
        )
        out[hname] = metrics.ConfusionMatrix(config)
    return HeadMetrics(out)
