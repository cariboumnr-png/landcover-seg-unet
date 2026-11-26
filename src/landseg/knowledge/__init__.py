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
    - `SpeciesEmbeddingsMetadata`: Species profile metadata TypedDict.
    - `SpeciesEntry`: Per-species embedding entry TypedDict.
    - `TaxonomySpec`: Domain taxonomy specification TypedDict.
    - `generate_embeddings_and_matrix`: Build embeddings and matrix.
    - `resolve_head_knowledge`: Resolve head taxonomies and matrices.
    - `resolve_profile_metadata`: Load profile embeddings metadata.
    - `resolve_similarity_matrix`: Load species similarity matrix.
    - `validate_specs`: Validate taxonomy specs against knowledge base.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'SpeciesEmbeddingsMetadata',
    'SpeciesEntry',
    'TaxonomySpec',
    'generate_embeddings_and_matrix',
    'resolve_head_knowledge',
    'resolve_profile_metadata',
    'resolve_similarity_matrix',
    'validate_specs',
]


# for static check
if typing.TYPE_CHECKING:
    from .embeddings import (
        SpeciesEmbeddingsMetadata,
        SpeciesEntry,
        generate_embeddings_and_matrix,
    )
    from .resolver import (
        resolve_head_knowledge,
        resolve_profile_metadata,
        resolve_similarity_matrix,
    )
    from .taxonomy import (
        TaxonomySpec,
        validate_specs,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.embeddings': [
        'SpeciesEmbeddingsMetadata',
        'SpeciesEntry',
        'generate_embeddings_and_matrix',
    ],
    '.resolver': [
        'resolve_head_knowledge',
        'resolve_profile_metadata',
        'resolve_similarity_matrix',
    ],
    '.taxonomy': [
        'TaxonomySpec',
        'validate_specs',
    ],
})
