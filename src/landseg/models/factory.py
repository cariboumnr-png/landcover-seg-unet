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

'''
Factory utilities for constructing and validating multi-head UNet models.

This module provides a single entry point for building a fully configured
`MultiHeadUNet` from dataset specifications and explicit architectural
configuration objects.

Key responsibilities:
    - Translate dataset metadata into model-ready configuration
      (input channels, head topology, class structure).
    - Instantiate a MultiHeadUNet backbone with optional domain conditioning.
    - Apply optional runtime overrides (e.g., logit adjustment, clamping).
    - Perform strict build-time validation via a synthetic forward pass.

The design is intentionally framework-agnostic with respect to experiment
management systems (e.g., Hydra). All configuration inputs are treated as
explicit structural contracts rather than global state.
'''

# standard imports
import typing
# third-party imports
import torch
# local imports
import landseg.core as core
import landseg.models.backbones as backbones
import landseg.models.core as model_core
import landseg.models.frames as frames

# -------------------------------Public Function-------------------------------
def build_multihead_unet(
    *,
    dataspecs: core.DataSpecs,
    body_config: backbones.UNetBodyConfig,
    conditioning_config: typing.Mapping[str, model_core.DomainTargetConfig],
    **kwargs
) -> frames.MultiHeadBaseModel:
    '''
    Construct and validate a `MultiHeadUNet` model.

    This factory assembles a complete multi-head segmentation model by:
        - Deriving structural constraints from dataset specifications.
        - Instantiating a UNet/UNet++ backbone via `body_config`.
        - Applying optional domain-conditioning (concat and/or FiLM).
        - Injecting runtime safety features (e.g., logit adjustment,
            clamping).
        - Running a strict forward-pass validation to verify consistency.

    Parameters are keyword-only to enforce explicit configuration
    boundaries and avoid ambiguity in model construction.

    Args:
        dataspecs: Dataset-derived specifications
        body_config: Backbone configuration describing architecture
            selection and convolutional structure parameters.
        conditioning_config: Mapping of domain names to conditioning
            strategies, defining how domain information is injected
            (concatenation and/or FiLM).
        **kwargs:
            Optional runtime overrides forwarded to the model constructor,
            including:
            - enable_clamp
            - clamp_range

    Returns:
        frames.MultiHeadBaseModel: Fully initialized and validated
            MultiHeadUNet instance.

    Raises:
        ValueError: If the backbone configuration is invalid or
            unsupported.
        RuntimeError: If the model fails structural or forward-pass
            validation.
    '''

    model = frames.MultiHeadUNet(
        dataspecs=dataspecs,
        backbone_config=body_config,
        conditioning_config=dict(conditioning_config),
        **kwargs
    )
    _validate_model_build(model, dataspecs)
    return model

def _validate_model_build(
    model: frames.MultiHeadBaseModel,
    dataspecs: core.DataSpecs,
    *,
    batch_size: int = 2,
) -> None:
    '''
    Build-time structural and numerical validations.

    The function constructs a synthetic batch using `build_dummy_batch()`
    from the model and executes a forward pass in eval/no_grad mode.

    Structural constraints:
        - Model accepts generated input tensors (from the model).
        - Output is a dict[str, Tensor].
        - Output heads exactly match dataspecs.heads.class_counts keys.
        - Each head output is a 4D tensor (B, C, H, W).
        - Batch dimension matches requested batch_size.
        - Channel dimension matches number of classes per head.
        - Spatial dimensions are not validated against input resolution.

    Numerical constraints:
        - All output tensors contain only finite values (no NaN/Inf).
    '''

    model.eval()

    # ----- forward pass
    b = model.build_dummy_batch(batch_size=batch_size)
    with torch.no_grad():
        try:
            outputs = model(
                b['x'],
                ids_domain = b.get('ids_domain'),
                vec_domain = b.get('vec_domain')
            )
        except Exception as e:
            raise RuntimeError('Model forward validation failed.') from e

    # ----- output container contract
    if not isinstance(outputs, dict):
        raise RuntimeError(f'Expected dict[str, Tensor], got {type(outputs)}')

    expected_heads = set(dataspecs.heads.class_counts.keys())
    actual_heads = set(outputs.keys())
    if expected_heads != actual_heads:
        raise RuntimeError(
            'Head mismatch between model and dataspecs.\n'
            f'Expected: {expected_heads}\n'
            f'Actual: {actual_heads}'
        )

    # ----- per-head validation
    for head, class_counts in dataspecs.heads.class_counts.items():

        y = outputs[head]

        # head type and shape
        if not torch.is_tensor(y):
            raise RuntimeError(f'Head {head} is not a tensor')
        if y.ndim != 4:
            raise RuntimeError(f'Head {head} must be BCHW, got {y.shape}')

        bsz, ch, _, _ = y.shape
        # batch consistency only (no spatial assumptions)
        if bsz != batch_size:
            raise RuntimeError(
                f'Batch mismatch in head {head}: '
                f'{bsz} != {batch_size}'
            )

        # channel topology
        if ch != len(class_counts):
            raise RuntimeError(
                f'Channel mismatch in head {head}: '
                f'{ch} != {len(class_counts)}'
            )

        # numeric stability
        if not torch.isfinite(y).all():
            raise RuntimeError(f'Non-finite values in head {head}')
