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

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access

'''Unit tests for MultiHeadUNet frame (unet.py).'''

# third-party imports
import torch
import pytest
# local imports
import landseg.models.frames.unet as frames_unet


# ----- MultiHeadUNet tests
def test_multihead_unet_initialization(
    dataspecs,
    mock_backbone_config_factory,
    mock_domain_config_factory,
):
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
    # modify dataspecs to have no domains
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
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    # spatial_divisor is 16.
    # input height_width in dataspecs (from tests/unit/conftest.py) is 256.
    # If we force the heights to be indivisible, it should raise RuntimeError.
    dataspecs.meta.image_specs = dataspecs.meta.image_specs.__class__(
        num_channels=4,
        height_width=15, # indivisible by 16
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
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = frames_unet.MultiHeadUNet(
        input_patch_size=256,
        dataspecs=dataspecs,
        backbone_config=backbone_cfg,
        conditioning_config={},
    )

    batch = model.build_dummy_batch(batch_size=3, device='cpu')

    assert 'x' in batch
    # dataspecs has ids_num=3 and vec_dim=2, so they should be present
    assert 'ids_domain' in batch
    assert 'vec_domain' in batch

    assert batch['x'].shape == (3, 4, 256, 256)
    assert batch['ids_domain'].shape == (3,)
    assert batch['vec_domain'].shape == (3, 2)


def test_multihead_unet_head_management(
    dataspecs,
    mock_backbone_config_factory,
):
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = frames_unet.MultiHeadUNet(
        input_patch_size=256,
        dataspecs=dataspecs,
        backbone_config=backbone_cfg,
        conditioning_config={},
    )

    # test active heads
    model.set_active_heads(['head1'])
    assert model.heads.active == ['head1']

    # test frozen heads
    model.set_frozen_heads(['head1'])
    assert model.heads.frozen == ['head1']

    # test reset
    model.reset_heads()
    assert model.heads.active is None
    assert model.heads.frozen is None


def test_multihead_unet_logit_adjust_properties(
    dataspecs,
    mock_backbone_config_factory,
):
    # register logits adjustments in dataspecs
    dataspecs.heads = dataspecs.heads.__class__(
        class_counts={'head1': [100, 200]},
        logits_adjust={'head1': [0.2, 0.1]},
        head_parent={'head1': None},
        head_parent_cls={'head1': None},
    )

    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    model = frames_unet.MultiHeadUNet(
        input_patch_size=256,
        dataspecs=dataspecs,
        backbone_config=backbone_cfg,
        conditioning_config={},
    )

    assert 'head1' in model.logit_adjust
    assert model.logit_adjust['head1'].shape == (1, 2, 1, 1)
    assert model.logit_adjust_alpha == 1.0

    model.set_logit_adjust_alpha(0.5)
    assert model.logit_adjust_alpha == 0.5


def test_multihead_unet_forward_pass(
    dataspecs,
    mock_backbone_config_factory,
    mock_domain_config_factory,
):
    backbone_cfg = mock_backbone_config_factory(body_type='unet', base_ch=16)
    # enable concat and film conditioning
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

    # execute forward
    outputs = model(
        batch['x'],
        ids_domain=batch['ids_domain'],
        vec_domain=batch['vec_domain']
    )

    assert isinstance(outputs, dict)
    assert 'head1' in outputs
    # output channel counts for 'head1' is 2 (since class_counts is [100, 200], which is length 2)
    assert outputs['head1'].shape == (2, 2, 256, 256)
