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
Utilities for constructing per-head composite loss functions.

This module builds loss components for each prediction head defined in a
multi-task model. It provides:

    - HeadLosses: a typed wrapper around a mapping of head names to
      CompositeLoss instances.
    - build_headlosses: factory that instantiates and configures
      CompositeLoss objects per head, including per-head a parameters
      for focal loss.

Used by the trainer to supply consistent, per-head loss computation
objects.
'''

# local imports
import landseg.session.components.task as task
import landseg.session.components.task.loss as loss

# --------------------------------Public  Class--------------------------------
class HeadLosses:
    '''
    Typed wrapper around a mapping of heads to `CompositeLoss` objects.

    This class provides:
    - key-based access to individual `CompositeLoss` instances.
    - a stable, strongly-typed container for passing head specs through
    the codebase.

    It is *not* a full `dict` replacement. To work with the underlying
    mapping directly, use method: `as_dict()`.
    '''

    def __init__(self, hloss: dict[str, loss.CompositeLoss]):
        self._hloss = hloss

    def __getitem__(self, key: str) -> loss.CompositeLoss:
        return self._hloss[key]

    def __len__(self) -> int:
        return len(self._hloss)

    def as_dict(self) -> dict[str, loss.CompositeLoss]:
        '''Return a shallow copy of the mapping as `dict[str, Loss]`.'''
        return dict(self._hloss)

# -------------------------------Public Function-------------------------------
def build_headlosses(
    headspecs: task.HeadSpecs,
    *,
    config: task.TaskConfig,
    ignore_index: int,
    spectral_band_indices: list[int] | None = None
) -> HeadLosses:
    '''
    Construct a mapping of head names to configured `CompositeLoss`
    instances.

    Args:
        headspecs A structure describing the model's prediction heads,
            including each head's name and its loss-specific parameters
            (e.g., per-head a values for focal loss).
        config: Base loss configuration shared across heads. Each enabled
            loss type must have its configuration block under a name
            matching those used by `CompositeLoss.registry`. This
            dictionary is modified per head if focal a values are
            provided.
        ignore_index: Label index to exclude from all loss computations.

    Returns:
        A `HeadLosses` container, providing typed access to the concrete
        `CompositeLoss` objects keyed by head name.

    Notes:
        - If focal loss is enabled, its a parameter is replaced for each
          head using values from `headspecs`.
        - The function expects alignment between head names and the
          configuration dictionary.
    '''

    loss_dict: dict[str, loss.CompositeLoss] = {}
    # iterate through names
    per_head_alphas = {
        h.name: h.loss_alpha for h in headspecs.as_dict().values()
    }
    for name in per_head_alphas.keys():
        # update loss alpha dein head
        # init loss compute module for each head
        loss_cls = loss.CompositeLoss(
            config,
            ignore_index=ignore_index,
            focal_alpha=per_head_alphas[name],
            spectral_band_indices=spectral_band_indices
        )
        loss_dict[name] = loss_cls
    return HeadLosses(loss_dict)
