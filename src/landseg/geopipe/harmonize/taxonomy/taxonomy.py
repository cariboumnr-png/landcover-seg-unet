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
Taxonomy resolver and validation for ecological domain metadata.

This module validates user-declared species taxonomy profiles and code
mappings against canonical knowledge base profiles and resolves target
integer classes to deterministic embedding matrix indices.

Public APIs:
    - `get_available_profiles`: Return registered taxonomy profile names.
    - `validate_specs`: Validate taxonomy specs against knowledge base.
'''

# standard imports
from __future__ import annotations
import json
import os
# local imports
import landseg.geopipe.core as geo_core
import landseg.knowledge as knowledge


# ----- public functions
def get_available_profiles(
    knowledge_root: str = 'knowledge',
) -> list[str]:
    '''
    Return list of registered taxonomy profile names in knowledge base.

    Args:
        knowledge_root:
            Root directory of the knowledge base.

    Returns:
        list[str]:
            List of profile directory names containing species metadata.
    '''
    emb_dir = os.path.join(knowledge_root, 'embeddings')
    if not os.path.isdir(emb_dir):
        return []
    profiles: list[str] = []
    for item in os.listdir(emb_dir):
        sub = os.path.join(emb_dir, item)
        if os.path.isdir(sub):
            meta_path = os.path.join(sub, 'species_metadata.json')
            if os.path.isfile(meta_path):
                profiles.append(item)
    return sorted(profiles)


def validate_specs(
    profile: str,
    species_mapping: dict[str, str],
    num_cls: int,
    knowledge_root: str = './knowledge',
) -> geo_core.TaxonomySpecs:
    '''
    Validate a label layer taxonomy specification against knowledge base.

    Args:
        profile:
            Canonical taxonomy profile name.
        species_mapping:
            Mapping of class index strings to species codes.
        num_cls:
            Number of active classes for the label layer (1..N).
        knowledge_root:
            Root directory of the knowledge base.

    Returns:
        geo_core.TaxonomySpecs:
            Resolved taxonomy specification dictionary.
    '''
    if len(species_mapping) != num_cls:
        raise ValueError(
            f'Taxonomy declares {len(species_mapping)} classes, but the label '
            f'layer declares {num_cls} active classes.'
        )
    code_lookup = _resolve_taxonomy_metadata(profile, knowledge_root)

    # resolve entries
    canonical_indices: dict[str, int] = {}

    for class_idx in range(1, num_cls + 1):
        class_idx = str(class_idx)
        code = species_mapping[class_idx]

        entry = code_lookup.get(code)
        if entry is None:
            raise ValueError(
                f'Unknown taxonomy class code "{code}" for class index '
                f'{class_idx} under profile "{profile}". '
                f'Class names must exactly match the canonical codes defined '
                f'by the profile.'
            )

        canonical_indices[class_idx] = entry['index']

    _specs: geo_core.TaxonomySpecs = {
        'profile': profile,
        'canonical_indices': canonical_indices
    }
    return _specs


# ----- private helpers
def _resolve_taxonomy_metadata(
    profile: str,
    root: str = 'knowledge',
) -> dict[str, knowledge.SpeciesEntry]:
    '''Load canonical metadata JSON for the given taxonomy profile.'''
    candidate_paths = [
        os.path.join(root, 'embeddings', profile, 'species_metadata.json'),
        os.path.join(profile, 'species_metadata.json'),
        profile,
    ]
    meta_path: str | None = None
    for p in candidate_paths:
        if os.path.isfile(p):
            meta_path = os.path.abspath(p)
            break

    if meta_path is None:
        available = get_available_profiles(root)
        avail_str = (
            ', '.join(f"'{a}'" for a in available) if available else 'none'
        )
        raise ValueError(
            f'Taxonomy profile "{profile}" not found in "{root}". '
            f'Available profiles: [{avail_str}].'
        )

    meta: knowledge.SpeciesEmbeddingsMetadata
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    classes_meta = meta.get('classes', [])
    if not classes_meta:
        raise ValueError(
            f'Taxonomy metadata for "{profile}" contains no class entries.'
        )

    code_lookup = {entry['code']: entry for entry in classes_meta}

    return code_lookup
