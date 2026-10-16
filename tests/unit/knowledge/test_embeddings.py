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
Unit tests for `landseg.knowledge.embeddings`.
'''

# standard imports
import unittest.mock
# third-party imports
import numpy
import torch
# local imports
import landseg.artifacts as artifacts
import landseg.knowledge.embeddings as embeddings_mod


# ----- `generate_embeddings_and_matrix` tests
def test_generate_embeddings_with_knowledge_root_str(dummy_csv_path, tmp_path):
    '''
    Given: A dummy species profiles CSV and knowledge root string path.
    When: `generate_embeddings_and_matrix()` is called.
    Then: Artifacts are generated in the canonical paths.
    '''
    kroot = tmp_path / 'knowledge'
    kp = artifacts.KnowledgePaths(root=kroot)

    mock_model = unittest.mock.MagicMock()
    mock_model.encode.return_value = numpy.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=numpy.float32
    )

    with unittest.mock.patch(
        'sentence_transformers.SentenceTransformer', return_value=mock_model
    ):
        embeddings_mod.generate_embeddings_and_matrix(
            csv_path=dummy_csv_path,
            knowledge_root=kroot,
            normalize=True,
        )

    profile = 'dummy_species_profiles'
    emb_path = kp.embeddings_fpath(profile)
    sim_path = kp.similarity_matrix_fpath(profile)
    meta_path = kp.metadata_fpath(profile)
    sim_csv_path = kp.similarity_csv_fpath(profile)

    assert emb_path.is_file()
    assert sim_path.is_file()
    assert meta_path.is_file()
    assert sim_csv_path.is_file()

    sim_tensor = torch.load(sim_path, weights_only=True)
    assert sim_tensor.shape == (2, 2)
    assert torch.allclose(sim_tensor, torch.eye(2))

    ctrl = artifacts.Controller[
        embeddings_mod.SpeciesEmbeddingsMetadata
    ].load_json_or_fail(meta_path)
    meta = ctrl.fetch()
    assert meta is not None
    assert meta['num_classes'] == 2
    assert meta['key_column'] == 'group_code'


def test_generate_embeddings_with_knowledge_root_path(dummy_csv_path, tmp_path):
    '''
    Given: A dummy species profiles CSV and `pathlib.Path` knowledge root.
    When: `generate_embeddings_and_matrix()` is called.
    Then: Target artifacts are built in the expected directory.
    '''
    kroot_path = tmp_path / 'path_knowledge'
    kp = artifacts.KnowledgePaths(root=kroot_path)

    mock_model = unittest.mock.MagicMock()
    mock_model.encode.return_value = numpy.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=numpy.float32
    )

    with unittest.mock.patch(
        'sentence_transformers.SentenceTransformer', return_value=mock_model
    ):
        embeddings_mod.generate_embeddings_and_matrix(
            csv_path=dummy_csv_path,
            knowledge_root=kroot_path,
            normalize=True,
        )

    profile = 'dummy_species_profiles'
    assert kp.similarity_matrix_fpath(profile).is_file()
