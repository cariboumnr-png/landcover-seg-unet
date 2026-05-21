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

'''
Multihead UNet architecture with domain conditioning and safety utils.

**Overview**\n
This module composes a UNet backbone with a lightweight multihead output
manager to support multi-prediction tasks (e.g., segmentation, detection
heatmaps, or auxiliary heads) from shared features. Domain conditioning
is supported through two complementary mechanisms:

- **Concatenation** (input-level): domain channels are appended to the
  input tensor when enabled. This is useful for simple categorical
  indicators or low-dimensional continuous descriptors that should
  influence all stages.

- **FiLM** (bottleneck-level): Feature-wise Linear Modulation is applied
  at the U-Net bottleneck using learned affine parameters derived from
  domain embeddings. This focuses adaptation where global context is
  strongest.

The design cleanly separates concerns:
- `MultiHeadUNet`: orchestrates backbone, conditioning, and head routing.
- `_HeadManager`: manages a set of 1x1 conv heads, activate/freeze state,
  and optional per-head logit adjustment.
- `_DomainRouter`: decides what domain info feeds concatenation vs FiLM,
  with optional projections from raw vectors to required dimensions.
- `_NumericSafety`: centralizes autocast context and value clamping to
  improve numerical stability during training/inference.

**Expected Shapes**
- Input  `x`: (N, C_in, H, W)
- U-Net body output for heads: (N, base_ch, H, W)
- Per-head logits: (N, C_head, H, W)

**Extension Points**
- Add or replace heads in `_HeadManager`.
- Swap input concatenation or FiLM policies in `_DomainRouter`.
- Adjust clamping ranges and autocast dtype in `_NumericSafety`.
- Replace the backbone with a drop-in module exposing `.encode/.decode`.

**Notes**
- Perhead logit adjustment supports label-frequency priors or calibrated
  offsets via a broadcastable `(1, C_head, 1, 1)` tensor stored in a
  non-trainable `Parameter`.
'''

# third-party imports
import torch
# local imports
import landseg._constants as c
import landseg.core as core
import landseg.models.backbones as backbones
import landseg.models.backbones.unet as unet_backbones
import landseg.models.core as model_core
import landseg.models.frames as frames
import landseg.models.frames.unet as unet

class MultiHeadUNet(frames.MultiHeadBaseModel):
    '''
    UNet with multihead outputs and optional domain conditioning.

    Supports:
        - Input-level domain concatenation (ConcatAdapter).
        - Bottleneck-level conditioning via FiLM (FilmConditioner).
        - Per-head logit adjustments.
        - Autocast and clamping utilities for numerical stability.

    The model orchestrates:
        * UNet backbone (.body)
        * Multihead output manager (.heads)
        * Domain routing to concat / FiLM branches (.domain_router)
        * Safety utilities controlling mixed precision (.safety)
    '''

    def __init__(
        self,
        *,
        dataspecs: core.DataSpecs,
        backbone_config: backbones.UNetBodyConfig,
        conditioning_config: dict[str, model_core.DomainTargetConfig],
        **kwargs
      ):
        '''
        Initialize a multihead UNet-based model.

        This initializer constructs a complete multi-head UNet model from
        pre-validated configuration objects supplied by the application
        layer (e.g., CLI / experiment runner). The model itself does not
        depend on any global or external configuration system.

        The configuration inputs are treated as *structural contracts*
        (typically via Protocols) and may originate from Hydra-backed
        dataclasses, plain dataclasses, or other compatible objects.

        Initializes:
        - UNet backbone specified by `backbone_config`.
        - Per-head Conv2d blocks and head routing via `_HeadManager`.
        - Optional input-level domain concatenation (`ConcatAdapter`).
        - Optional bottleneck-level FiLM conditioning (`FilmConditioner`).
        - Domain routing logic for IDs and vectors (`_DomainRouter`).
        - Numeric safety utilities (autocast and value clamping).
        - Per-head logit adjustment buffers (non-trainable).

        Args:
            backbone_config:
                Backbone-level configuration describing:
                - backbone variant (e.g., 'unet', 'unetpp'),
                - base channel width,
                - convolutional block parameters forwarded to backbone.
            dataspecs_config:
                Model-level configuration defining:
                - input channel count,
                - head definitions and class counts,
                - per-head logit adjustment priors (optional).
            conditioning:
                Domain conditioning configuration specifying how
                categorical IDs and/or continuous vectors are routed to:
                - input concatenation,
                - bottleneck FiLM modulation.
            **kwargs:
                Optional runtime flags and overrides, including:
                - `enable_logit_adjust` (bool): runtime toggle for logit
                  adjustment (default: True).
                - `enable_clamp` (bool): enable numeric clamping
                  (default: True).
                - `clamp_range` (tuple[float, float]): numeric clamp
                  bounds (default: (1e-4, 1e4)).

        Notes:
            - All parameters are keyword-only by design to make configuration
              boundaries explicit and order-independent.
            - Configuration ownership resides outside the model module;
              this class assumes inputs are already validated.
            - The model body must expose a `.encode()` / `.decode()` interface
              compatible with UNet-style backbones.
        '''

        super().__init__()

        # base channels
        base_ch = backbone_config.base_ch

        heads = {k: len(v) for k, v in dataspecs.heads.class_counts.items()}
        self.heads = model_core.HeadManager(base_ch, heads)

        # domain router
        self.domain_router = model_core.DomainContextRouter(
            domain_ids_num=dataspecs.domains.ids_num,
            domain_vec_dim=dataspecs.domains.vec_dim,
            targets=conditioning_config
        )
        # concat adapter
        concat_config = conditioning_config.get('concat')
        if concat_config is None:
            self.concat = None
            add_dim = 0
        else:
            add_dim = concat_config.ids_embd_dims
            self.concat = unet.ConcatAdapter(concat_dim=add_dim)

        # core UNet body
        body = self._get_model_body(backbone_config.body)
        in_ch = dataspecs.meta.image_specs.num_channels
        self.body: unet_backbones.UNetBackbone
        self.body = body(in_ch + add_dim, base_ch, **backbone_config.conv_params)

        # film conditioning adpater
        film_config = conditioning_config.get('film')
        if film_config is None:
            self.film = None
        else:
            self.film = unet.FilmConditioner(
                embed_dim=film_config.vec_proj_dims,
                bottleneck_ch=self.body.bottleneck_ch,
                hidden_dim=film_config.conditioner_config.get('hidden_dim')
            )

        # default safety
        self.num_safety = model_core.NumericSafety(
            enable_clamp=kwargs.get('enable_clamp', True),
            clamp_range=kwargs.get('clamp_range', (1e-4, 1e4)),
            device=c.DEVICE
        )

        # register logits adjustments
        self._register_logit_adjust(dataspecs.heads.logits_adjust)

    @property
    def spatial_divisor(self) -> int:
        '''Minimum spatial divisor from the model body.'''
        return self.body.spatial_divisor

    def _forward_features(
        self,
        x: torch.Tensor,
        domains: dict[str, model_core.DomainTargetPayload]
    ) -> torch.Tensor:
        '''doc'''

        # feed domain to router
        film = domains.get('film')
        concat = domains.get('concat')

        # concatenate domain channels (if configured)
        if self.concat is not None:
            x = self.concat(x, concat)

        # force float32 with clamping control for gradient stability
        with self.num_safety.autocast_context(dtype=torch.float32):
            # encoders
            x1, x2, x3, x4, xb = self.body.encode(self.num_safety.clamp(x))
            xb = self.num_safety.clamp(xb)
            # FiLM at bottom if provided
            if self.film is not None:
                xb = self.film(xb, film)
                xb = self.num_safety.clamp(xb)
            # decoders
            xs = tuple(self.num_safety.clamp(x) for x in [x1, x2, x3, x4, xb])
            x = self.body.decode(xs)

        # return features
        return x

    @staticmethod
    def _get_model_body(body: str) -> unet_backbones.UNetBackbone:
        '''Retrieve model body by name.'''

        # model body registry
        body_registry = {
            'unet': unet_backbones.UNet,
            'unetpp': unet_backbones.UNetPP,
            'unetppp': unet_backbones.UNetPPP
        }
        if not body in body_registry:
            raise ValueError(f'Invalid base model: {body}')
        return body_registry[body]
