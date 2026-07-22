# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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

'''Unit tests for MultiHeadUNet frame (unet.py).'''

# third-party imports
import pytest
import torch
# local imports
import landseg.models.frames.unet as frames_unet


# ----- `MultiHeadUNet` tests
def test_multihead_unet_initialization(
    dataspecs,
    mock_backbone_config_factory,
    mock_domain_config_factory,
):
    '''
    Given: A DataSpec and configuration objects.
    When: Initializing a MultiHeadUNet frame.
    Then: Correctly construct model heads, conditioners, divisor, and safety.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    conditioning_cfg = {
        'concat': mock_domain_config_factory(use_ids=True, ids_embd_dims=4),
        'film': mock_domain_config_factory(use_vec=True, vec_proj_dims=4),
    }

    model = frames_unet.MultiHeadUNet(
        input_patch_size=256,
        dataspecs=dataspecs,
        backbone_config=backbone_cfg,
        conditioning_config=conditioning_cfg,
    )

    assert isinstance(model.heads, torch.nn.Module)
    assert model.concat is not None
    assert model.film is not None
    assert model.spatial_divisor == 16
    assert model.num_safety.enable_clamp is True


def test_multihead_unet_initialization_no_domain(
    dataspecs,
    mock_backbone_config_factory,
):
    '''
    Given: A DataSpec without any domain dimensions.
    When: Initializing a MultiHeadUNet frame.
    Then: Construct the model without any concat/film conditioning blocks.
    '''
    dataspecs.domains = dataspecs.domains.__class__(
        train=None, val=None, test=None, ids_num=0, vec_dim=0
    )

    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)

    model = frames_unet.MultiHeadUNet(
        input_patch_size=256,
        dataspecs=dataspecs,
        backbone_config=backbone_cfg,
        conditioning_config={},
    )

    assert model.concat is None
    assert model.film is None


def test_multihead_unet_incompatible_spatial_size(
    dataspecs,
    mock_backbone_config_factory,
):
    '''
    Given: A patch size that is indivisible by the model divisor.
    When: Initializing a MultiHeadUNet frame.
    Then: Raise a RuntimeError alerting spatial size incompatibility.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    dataspecs.meta.image_specs = dataspecs.meta.image_specs.__class__(
        num_channels=4,
        height_width=15,
        array_key='image',
        band_map={'red': 0, 'green': 1, 'blue': 2, 'dem': 3}
    )

    with pytest.raises(RuntimeError, match='spatial size is incompatible'):
        frames_unet.MultiHeadUNet(
            input_patch_size=256,
            dataspecs=dataspecs,
            backbone_config=backbone_cfg,
            conditioning_config={},
        )


def test_multihead_unet_build_dummy_batch(
    dataspecs,
    mock_backbone_config_factory,
):
    '''
    Given: A constructed MultiHeadUNet model.
    When: Invoking build_dummy_batch.
    Then: Generate a batch dictionary containing x, ids, and vec tensors.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = frames_unet.MultiHeadUNet(
        input_patch_size=256,
        dataspecs=dataspecs,
        backbone_config=backbone_cfg,
        conditioning_config={},
    )

    batch = model.build_dummy_batch(batch_size=3, device='cpu')

    assert 'x' in batch
    assert 'ids_domain' in batch
    assert 'vec_domain' in batch

    assert batch['x'].shape == (3, 4, 256, 256)
    assert batch['ids_domain'].shape == (3,)
    assert batch['vec_domain'].shape == (3, 2)


def test_multihead_unet_head_management(
    dataspecs,
    mock_backbone_config_factory,
):
    '''
    Given: A constructed MultiHeadUNet model.
    When: Setting active/frozen heads or resetting them.
    Then: Correctly apply active, frozen state and reset to None.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = frames_unet.MultiHeadUNet(
        input_patch_size=256,
        dataspecs=dataspecs,
        backbone_config=backbone_cfg,
        conditioning_config={},
    )

    model.set_active_heads(['head_1'])
    assert model.heads.active == ['head_1']

    model.set_frozen_heads(['head_1'])
    assert model.heads.frozen == ['head_1']

    model.reset_heads()
    assert model.heads.active is None
    assert model.heads.frozen is None


def test_multihead_unet_logit_adjust_properties(
    dataspecs,
    mock_backbone_config_factory,
):
    '''
    Given: A DataSpec containing logits adjustments.
    When: Initializing a MultiHeadUNet frame.
    Then: Correctly load logit adjustments tensors and scale properties.
    '''
    dataspecs.heads = dataspecs.heads.__class__(
        class_counts={'head_1': [100, 200]},
        logits_adjust={'head_1': [0.2, 0.1]},
        head_parent={'head_1': None},
        head_parent_cls={'head_1': None},
    )

    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = frames_unet.MultiHeadUNet(
        input_patch_size=256,
        dataspecs=dataspecs,
        backbone_config=backbone_cfg,
        conditioning_config={},
    )

    assert 'head_1' in model.logit_adjust
    assert model.logit_adjust['head_1'].shape == (1, 2, 1, 1)
    assert model.logit_adjust_alpha == 1.0

    model.set_logit_adjust_alpha(0.5)
    assert model.logit_adjust_alpha == 0.5


def test_multihead_unet_forward_pass(
    dataspecs,
    mock_backbone_config_factory,
    mock_domain_config_factory,
):
    '''
    Given: A MultiHeadUNet model with concats and film conditioners.
    When: Running forward pass with dummy batch inputs.
    Then: Return a dictionary containing the predicted head tensors.
    '''
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    conditioning_cfg = {
        'concat': mock_domain_config_factory(use_ids=True, ids_embd_dims=4),
        'film': mock_domain_config_factory(use_vec=True, vec_proj_dims=4),
    }

    model = frames_unet.MultiHeadUNet(
        input_patch_size=256,
        dataspecs=dataspecs,
        backbone_config=backbone_cfg,
        conditioning_config=conditioning_cfg,
    )

    batch = model.build_dummy_batch(batch_size=2)

    outputs = model(
        batch['x'],
        ids_domain=batch['ids_domain'],
        vec_domain=batch['vec_domain']
    )

    assert isinstance(outputs, dict)
    assert 'head_1' in outputs
    assert 'head_2' in outputs
    assert outputs['head_1'].shape == (2, 2, 256, 256)
    assert outputs['head_2'].shape == (2, 3, 256, 256)
