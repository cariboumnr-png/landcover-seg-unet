# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
    return ctrl.fetch(strict=True)


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
    return ctrl.fetch(strict=True)
