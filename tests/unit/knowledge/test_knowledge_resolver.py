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

'''Unit tests for `landseg.knowledge.resolver`.'''

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


def test_resolve_similarity_matrix_none_root():
    '''
    Given: No knowledge_root provided.
    When: `resolve_similarity_matrix` is called.
    Then: Raise `FileNotFoundError`.
    '''
    with pytest.raises(
        FileNotFoundError, match='Knowledge root not configured'
    ):
        knowledge.resolve_similarity_matrix(profile='any')


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


def test_resolve_profile_metadata_none_root():
    '''
    Given: No knowledge_root provided.
    When: `resolve_profile_metadata` is called.
    Then: Raise `FileNotFoundError`.
    '''
    with pytest.raises(
        FileNotFoundError, match='Knowledge root not configured'
    ):
        knowledge.resolve_profile_metadata(profile='any')


# ----- `resolve_head_knowledge` tests
def test_resolve_head_knowledge(tmp_path, monkeypatch):
    '''
    Given: Per-head counts, taxonomy specs, class names, and root.
    When: Calling `resolve_head_knowledge` with missing or valid root.
    Then: Bypass missing roots and resolve validated taxonomy + tensors.
    '''
    counts = {'leadspc': [10, 20]}
    raw_taxonomy = {'leadspc': {'profile': 'ontario_tree_species'}}
    class_names = {'leadspc': ['SB', 'PJ']}

    # missing knowledge root preserves raw taxonomy cleanly
    tax_empty, sims_empty = knowledge.resolve_head_knowledge(
        counts,
        raw_taxonomy,
        class_names,
        tmp_path / 'missing',
    )
    assert tax_empty == raw_taxonomy
    assert not sims_empty

    # valid knowledge root resolves validated taxonomy & matrix
    kroot = tmp_path / 'knowledge'
    kroot.mkdir()
    kp = artifacts.KnowledgePaths(root=kroot)
    mat = torch.eye(2, dtype=torch.float32)
    artifacts.Controller[torch.Tensor](
        kp.similarity_matrix_fpath('ontario_tree_species')
    ).persist(mat)
    monkeypatch.setattr(
        'landseg.knowledge.validate_specs',
        lambda profile, mapping, num_cls, knowledge_root: {
            'profile': profile,
            'verified': num_cls,
        },
    )

    tax_ok, sims_ok = knowledge.resolve_head_knowledge(
        counts,
        raw_taxonomy,
        class_names,
        kroot,
    )
    assert tax_ok == {
        'leadspc': {'profile': 'ontario_tree_species', 'verified': 2}
    }
    assert torch.allclose(sims_ok['leadspc'], mat)
