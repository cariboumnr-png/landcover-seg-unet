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
from __future__ import annotations
import pathlib
# third-party imports
import torch
# local imports
import landseg.artifacts as artifacts
import landseg.knowledge as knowledge

MetaController = artifacts.Controller[knowledge.SpeciesEmbeddingsMetadata]


def resolve_similarity_matrix(
    profile: str,
    knowledge_root: str | pathlib.Path = 'knowledge',
) -> torch.Tensor:
    '''
    Resolve and load precomputed species similarity matrix for a profile.

    Args:
        profile: Canonical species profile name.
        knowledge_root: Root directory path for knowledge base artifacts.

    Returns:
        PyTorch tensor [N, N] containing pairwise cosine similarities.

    Raises:
        FileNotFoundError: If the similarity matrix artifact is missing.
        artifacts.ArtifactError: If hash validation fails or the file
            is corrupted.
    '''
    kp = artifacts.KnowledgePaths(root=knowledge_root)
    matrix_path = kp.similarity_matrix_fpath(profile)
    ctrl = artifacts.Controller[torch.Tensor].load_pt_or_fail(matrix_path)
    tensor = ctrl.fetch()
    return tensor


def resolve_profile_metadata(
    profile: str,
    knowledge_root: str | pathlib.Path = 'knowledge',
) -> knowledge.SpeciesEmbeddingsMetadata:
    '''
    Resolve and load species embeddings metadata for a profile.

    Args:
        profile: Canonical species profile name.
        knowledge_root: Root directory path for knowledge base artifacts.

    Returns:
        Metadata dictionary including class codes, names, and dimensions.

    Raises:
        FileNotFoundError: If the metadata artifact is missing.
        artifacts.ArtifactError: If hash validation fails or the file
            is corrupted.
    '''
    kp = artifacts.KnowledgePaths(root=knowledge_root)
    meta_path = kp.metadata_fpath(profile)
    ctrl = MetaController.load_json_or_fail(meta_path)
    meta = ctrl.fetch()
    return meta
