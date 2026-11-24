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
Top-level namespace for `landseg.knowledge`.

Exposes ecological domain taxonomy validation, species embedding
generation, and profile resolution APIs via lazy resolution to keep
import order simple and circular-free.

Public APIs:
    - `SpeciesEmbeddingsMetadata`: TypedDict for embeddings profile metadata.
    - `SpeciesEntry`: TypedDict for per-species embedding entry metadata.
    - `TaxonomySpec`: TypedDict for domain taxonomy specification.
    - `generate_embeddings_and_matrix`: Build species embeddings and matrix.
    - `resolve_profile_metadata`: Load embeddings metadata for a profile.
    - `resolve_similarity_matrix`: Load precomputed species similarity matrix.
    - `validate_specs`: Validate taxonomy specs against knowledge base.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # functions
    'generate_embeddings_and_matrix',
    'resolve_profile_metadata',
    'resolve_similarity_matrix',
    'validate_specs',
    # typing
    'SpeciesEmbeddingsMetadata',
    'SpeciesEntry',
    'TaxonomySpec',
]


# for static check
if typing.TYPE_CHECKING:
    from .embeddings import (
        SpeciesEmbeddingsMetadata,
        SpeciesEntry,
        generate_embeddings_and_matrix,
    )
    from .resolver import (
        resolve_profile_metadata,
        resolve_similarity_matrix,
    )
    from .taxonomy import (
        TaxonomySpec,
        validate_specs,
    )


def __getattr__(name: str):
    if name in {
        'SpeciesEmbeddingsMetadata',
        'SpeciesEntry',
        'generate_embeddings_and_matrix',
    }:
        obj = importlib.import_module('.embeddings', __package__)
        return getattr(obj, name)

    if name in {
        'resolve_profile_metadata',
        'resolve_similarity_matrix',
    }:
        obj = importlib.import_module('.resolver', __package__)
        return getattr(obj, name)

    if name in {
        'TaxonomySpec',
        'validate_specs',
    }:
        obj = importlib.import_module('.taxonomy', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
