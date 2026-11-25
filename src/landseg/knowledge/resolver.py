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
Resolution helper for knowledge base assets and profiles.
'''

# standard imports
import pathlib
import typing
# third-party imports
import torch
# local imports
import landseg.artifacts as artifacts
import landseg.knowledge as knowledge


def resolve_similarity_matrix(
    profile: str,
    knowledge_root: pathlib.Path | str | None = None,
) -> torch.Tensor:
    '''
    Resolve and load precomputed species similarity matrix for a profile.

    Args:
        profile: Canonical species profile name.
        knowledge_root: Root directory path for knowledge base artifacts.

    Returns:
        PyTorch tensor [N, N] containing pairwise cosine similarities.

    Raises:
        FileNotFoundError: If knowledge_root is None or artifact missing.
        artifacts.ArtifactError: If hash validation fails or the file
            is corrupted.
    '''
    if knowledge_root is None:
        raise FileNotFoundError(
            f'Knowledge root not configured; cannot resolve matrix for '
            f'"{profile}"'
        )
    kp = artifacts.KnowledgePaths(root=pathlib.Path(knowledge_root))
    matrix_path = kp.similarity_matrix_fpath(profile)
    ctrl = artifacts.Controller[torch.Tensor](matrix_path)
    return ctrl.fetch()


def resolve_profile_metadata(
    profile: str,
    knowledge_root: pathlib.Path | str | None = None,
) -> knowledge.SpeciesEmbeddingsMetadata:
    '''
    Resolve and load species embeddings metadata for a profile.

    Args:
        profile: Canonical species profile name.
        knowledge_root: Root directory path for knowledge base artifacts.

    Returns:
        Metadata dictionary including class codes, names, and dimensions.

    Raises:
        FileNotFoundError: If knowledge_root is None or artifact missing.
        artifacts.ArtifactError: If hash validation fails or the file
            is corrupted.
    '''
    if knowledge_root is None:
        raise FileNotFoundError(
            f'Knowledge root not configured; cannot resolve metadata for '
            f'"{profile}"'
        )
    kp = artifacts.KnowledgePaths(root=pathlib.Path(knowledge_root))
    meta_path = kp.metadata_fpath(profile)
    ctrl = artifacts.Controller[knowledge.SpeciesEmbeddingsMetadata](meta_path)
    return ctrl.fetch()


def resolve_head_knowledge(
    counts: typing.Mapping[str, typing.Sequence[int]],
    raw_taxonomy: typing.Mapping[str, typing.Mapping[str, typing.Any]],
    class_names: typing.Mapping[str, typing.Sequence[str]],
    knowledge_root: pathlib.Path | str | None = None,
) -> tuple[dict[str, dict[str, typing.Any]], dict[str, torch.Tensor]]:
    '''Resolve validated head taxonomies and similarity matrices.'''
    taxonomy: dict[str, dict[str, typing.Any]] = {
        k: dict(v) for k, v in raw_taxonomy.items()
    }
    sim_matrices: dict[str, torch.Tensor] = {}
    k_paths = (
        artifacts.KnowledgePaths(pathlib.Path(knowledge_root))
        if knowledge_root is not None
        else None
    )
    if k_paths is None or not k_paths.root.exists():
        return taxonomy, sim_matrices

    for hname, tax_dict in taxonomy.items():
        if not isinstance(tax_dict, dict) or 'profile' not in tax_dict:
            continue
        profile = tax_dict['profile']
        cls_list = class_names.get(hname)
        if cls_list and hname in counts and len(cls_list) == len(counts[hname]):
            mapping = {str(i): c for i, c in enumerate(cls_list, 1)}
            try:
                resolved = knowledge.validate_specs(
                    profile,
                    mapping,
                    len(counts[hname]),
                    knowledge_root=k_paths.root,
                )
                taxonomy[hname] = dict(resolved)
            except (ValueError, FileNotFoundError):
                pass
        try:
            sim_matrices[hname] = resolve_similarity_matrix(
                profile,
                knowledge_root=k_paths.root,
            )
        except (FileNotFoundError, artifacts.ArtifactError):
            pass

    return taxonomy, sim_matrices
