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
    - `validate_specs`: Validate taxonomy specs against knowledge base.
'''

# standard imports
import pathlib
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.knowledge.embeddings as embeddings


# ----- public functions
def validate_specs(
    profile: str,
    species_mapping: dict[str, str],
    num_cls: int,
    knowledge_root: pathlib.Path | str | None = None,
) -> geo_core.TaxonomySpec:
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
        geo_core.TaxonomySpec:
            Resolved taxonomy specification dictionary.
    '''
    if knowledge_root is None:
        raise ValueError(
            f'Dataset manifest requests taxonomy validation for '
            f'"{profile}", but execution.knowledge_dir is not configured.'
        )
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

    _specs: geo_core.TaxonomySpec = {
        'profile': profile,
        'canonical_indices': canonical_indices
    }
    return _specs


# ----- private helpers
def _get_profiles(knowledge_root: pathlib.Path | str | None) -> list[str]:
    '''Return list of registered taxonomy profile names.'''
    if knowledge_root is None:
        return []
    emb_dir = pathlib.Path(knowledge_root) / 'embeddings'
    if not emb_dir.is_dir():
        return []
    profiles: list[str] = []
    for sub in emb_dir.iterdir():
        if sub.is_dir():
            meta_path = sub / 'species_metadata.json'
            if meta_path.is_file():
                profiles.append(sub.name)
    return sorted(profiles)


def _resolve_taxonomy_metadata(
    profile: str,
    root: pathlib.Path | str | None = None,
) -> dict[str, embeddings.SpeciesEntry]:
    '''Load canonical metadata JSON for the given taxonomy profile.'''
    candidate_paths: list[pathlib.Path] = []
    if root is not None:
        candidate_paths.append(
            pathlib.Path(root) / 'embeddings' / profile / 'species_metadata.json'
        )
    candidate_paths.extend([
        pathlib.Path(profile) / 'species_metadata.json',
        pathlib.Path(profile),
    ])
    meta_path: pathlib.Path | None = None
    for p in candidate_paths:
        if p.is_file():
            meta_path = p.resolve()
            break

    if meta_path is None:
        available = _get_profiles(root)
        avail_str = (
            ', '.join(f"'{a}'" for a in available) if available else 'none'
        )
        raise ValueError(
            f'Taxonomy profile "{profile}" not found in "{root}". '
            f'Available profiles: [{avail_str}].'
        )

    ctrl = artifacts.Controller[embeddings.SpeciesEmbeddingsMetadata](meta_path)
    meta = ctrl.fetch(strict=True)

    classes_meta = meta.get('classes', [])
    if not classes_meta:
        raise ValueError(
            f'Taxonomy metadata for "{profile}" contains no class entries.'
        )

    code_lookup = {entry['code']: entry for entry in classes_meta}

    return code_lookup
