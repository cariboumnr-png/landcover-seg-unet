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
Unit tests for `landseg.knowledge.resolver`.
'''

# third-party imports
import pytest
import torch
# local imports
import landseg.artifacts as artifacts
import landseg.knowledge as knowledge


# ----- `resolve_similarity_matrix` tests
def test_resolve_similarity_matrix(tmp_path):
    '''
    Given: A knowledge base containing a valid similarity matrix.
    When: `resolve_similarity_matrix` is called with profile name.
    Then: Return loaded torch tensor with verified integrity.
    '''
    kp = artifacts.KnowledgePaths(root=tmp_path / 'knowledge')
    profile = 'test_profile'
    mat_path = kp.similarity_matrix_fpath(profile)

    tensor_in = torch.eye(3, dtype=torch.float32)
    artifacts.Controller[torch.Tensor](mat_path).persist(tensor_in)

    loaded = knowledge.resolve_similarity_matrix(
        profile=profile,
        knowledge_root=kp.root,
    )
    assert torch.allclose(loaded, tensor_in)


def test_resolve_similarity_matrix_missing(tmp_path):
    '''
    Given: A non-existent profile name.
    When: `resolve_similarity_matrix` is called.
    Then: Raise `ArtifactError`.
    '''
    with pytest.raises(artifacts.ArtifactError):
        knowledge.resolve_similarity_matrix(
            profile='non_existent_profile',
            knowledge_root=tmp_path / 'knowledge',
        )


# ----- `resolve_profile_metadata` tests
def test_resolve_profile_metadata(tmp_path):
    '''
    Given: A knowledge base containing valid profile metadata.
    When: `resolve_profile_metadata` is called with profile name.
    Then: Return loaded metadata dictionary.
    '''
    kp = artifacts.KnowledgePaths(root=tmp_path / 'knowledge')
    profile = 'test_profile'
    meta_path = kp.metadata_fpath(profile)

    meta_in = {'num_classes': 5, 'model_name': 'test_model'}
    artifacts.Controller[dict](meta_path).persist(meta_in)

    loaded = knowledge.resolve_profile_metadata(
        profile=profile,
        knowledge_root=kp.root,
    )
    assert loaded['num_classes'] == 5
    assert loaded['model_name'] == 'test_model'
