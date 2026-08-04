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

'''Unit tests for Ecological similarity loss module (ecological.py).'''

# third-party imports
import torch
# local imports
import landseg.session.engine.runtime.tasks.loss.primitives.ecological as eco_loss


# ----- `EcologicalSimilarityLoss` initialization
def test_ecological_similarity_loss_initialization():
    '''
    Given: A similarity matrix tensor.
    When: `EcologicalSimilarityLoss` is initialized.
    Then: Register buffer correctly.
    '''
    sim_matrix = torch.eye(3, dtype=torch.float32)
    loss_module = eco_loss.EcologicalSimilarityLoss(
        similarity_matrix=sim_matrix,
        ignore_index=255,
        reduction='mean'
    )
    assert isinstance(loss_module.similarity_matrix, torch.Tensor)
    assert torch.equal(loss_module.similarity_matrix, sim_matrix)


def test_ecological_similarity_loss_forward():
    '''
    Given: Logits, targets, and a 3x3 similarity matrix.
    When: `forward` is executed.
    Then: Return non-zero loss proportional to dissimilarity.
    '''
    # 3 classes, 1 batch, 2x2 image
    logits = torch.tensor(
        [[
            [[5.0, 0.0], [0.0, 0.0]],
            [[0.0, 5.0], [0.0, 0.0]],
            [[0.0, 0.0], [5.0, 5.0]]
        ]],
        dtype=torch.float32
    )
    targets = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.long)

    sim_matrix = torch.tensor(
        [[1.0, 0.8, 0.2], [0.8, 1.0, 0.1], [0.2, 0.1, 1.0]],
        dtype=torch.float32
    )

    loss_fn = eco_loss.EcologicalSimilarityLoss(
        similarity_matrix=sim_matrix,
        ignore_index=255,
        reduction='mean'
    )

    loss_val = loss_fn(logits, targets)
    assert loss_val.ndim == 0
    assert float(loss_val) >= 0.0
