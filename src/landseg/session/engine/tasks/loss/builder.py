# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
multi-task model, configuring per-head alpha parameters for focal loss.

Used by the trainer to supply consistent, per-head loss computation
objects.

Public APIs:
    - `HeadLosses`: Typed mapping container for per-head loss modules.
    - `build_headlosses`: Construct per-head `CompositeLoss` modules.
'''

# third-party imports
import torch
# local imports
import landseg.session.engine.tasks.heads as heads
import landseg.session.engine.tasks.loss.composite as composite


# ----- public classes
class HeadLosses:
    '''
    Typed wrapper around a mapping of heads to `CompositeLoss` objects.

    This class provides:
        - Key-based access to individual `CompositeLoss` instances.
        - A stable, strongly-typed container for passing head loss
          modules through the codebase.

    It is *not* a full `dict` replacement. To work with the underlying
    mapping directly, use `as_dict()`.
    '''

    def __init__(self, hloss: dict[str, composite.CompositeLoss]):
        '''
        Initialize the container from a per-head loss mapping.

        Args:
            hloss:
                Mapping of head name to `CompositeLoss` instance.
        '''
        self._hloss = hloss

    def __getitem__(self, key: str) -> composite.CompositeLoss:
        '''
        Retrieve the composite loss module for a given head name.

        Args:
            key:
                Head name key.

        Returns:
            composite.CompositeLoss:
                Loss module registered for `key`.
        '''
        return self._hloss[key]

    def __len__(self) -> int:
        return len(self._hloss)

    def as_dict(self) -> dict[str, composite.CompositeLoss]:
        '''
        Return a shallow copy of the per-head loss mapping.

        Returns:
            dict[str, composite.CompositeLoss]:
                Shallow copy mapping head names to `CompositeLoss`
                modules.
        '''
        return dict(self._hloss)


# ----- public functions
def build_headlosses(
    headspecs: heads.HeadSpecs,
    *,
    config: composite.CompositeLossConfig,
    ignore_index: int,
    spectral_band_indices: list[int] | None = None,
    ecological_similarity_matrix: torch.Tensor | None = None
) -> HeadLosses:
    '''
    Build a mapping of head names to `CompositeLoss` instances.

    Args:
        headspecs:
            Structure describing the model prediction heads.
        config:
            Base loss configuration shared across heads.
        ignore_index:
            Label index to exclude from all loss computations.
        spectral_band_indices:
            Optional list of spectral band indices.
        ecological_similarity_matrix:
            Optional explicit similarity tensor override.

    Returns:
        HeadLosses:
            Typed container mapping heads to `CompositeLoss` modules.
    '''
    loss_dict: dict[str, composite.CompositeLoss] = {}
    for name, headspec in headspecs.as_dict().items():
        sim_mat = (
            ecological_similarity_matrix
            if ecological_similarity_matrix is not None
            else headspec.similarity_matrix
        )
        loss_cls = composite.CompositeLoss(
            config,
            ignore_index=ignore_index,
            focal_alpha=headspec.loss_alpha,
            spectral_band_indices=spectral_band_indices,
            ecological_similarity_matrix=sim_mat
        )
        loss_dict[name] = loss_cls
    return HeadLosses(loss_dict)
