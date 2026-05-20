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

        # base channels
        base_ch = backbone_config.base_ch

        heads = {k: len(v) for k, v in dataspecs.heads.class_counts.items()}
        heads_manager = model_core.HeadManager(base_ch, heads)

        # domain router
        domain_router = model_core.DomainContextRouter(
            domain_vec_dim=dataspecs.domains.vec_dim,
            targets=conditioning_config
        )

        # domain concatenation
        concat_cfg = conditioning_config.get('input')
        self.concat = unet.ConcatAdapter(
            out_dom=concat_cfg.projection.out_dim,
            use_mlp=concat_cfg.projection.use_mlp,
            dim_continuous=dataspecs.domains.ids_max,
            num_categories=dataspecs.domains.vec_dim,
        ) if concat_cfg else None

        # film conditioner
        film_cfg = conditioning_config.get('bottleneck')
        self.film = unet.FilmConditioner(
            embed_dim=film_cfg.projection.out_dim,
            num_categories=dataspecs.domains.ids_max,
            dim_continuous=dataspecs.domains.vec_dim
        ) if film_cfg else None

        # core UNet body
        body = self._get_model_body(backbone_config.body)
        in_ch = dataspecs.meta.image_specs.num_channels
        in_ch += concat_cfg.projection.out_dim if concat_cfg is not None else 0
        self.body = body(in_ch, base_ch, **backbone_config.conv_params)

        # default safety
        num_safety = model_core.NumericSafety(
            enable_clamp=kwargs.get('enable_clamp', True),
            clamp_range=kwargs.get('clamp_range', (1e-4, 1e4)),
            device=c.DEVICE
        )

        # build components
        super().__init__(
            body=body,
            heads_manager=heads_manager,
            num_safety=num_safety,
            domain_router=domain_router
        )

        # register logits adjustments
        self._register_logit_adjust(dataspecs.heads.logits_adjust)

    def _forward_features(
        self,
        x: torch.Tensor,
        domains: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]]
    ) -> torch.Tensor:
        '''doc'''

        # feed domain to router
        concat = domains.get('input', {})
        film = domains.get('bottleneck', {})

        # concatenate domain channels (if configured)
        if self.concat is not None:
            x = self.concat(x, *concat)

        # force float32 with clamping control for gradient stability
        with self.safety.autocast_context(dtype=torch.float32):
            # encoders
            x1, x2, x3, x4, xb = self.body.encode(self.safety.clamp(x))
            xb = self.safety.clamp(xb)
            # FiLM at bottom if provided
            if self.film is not None:
                z = self.film.embed(*film)
                xb = self.film.film_bottleneck(xb, z)
                xb = self.safety.clamp(xb)
            # decoders
            xs = tuple(self.safety.clamp(xx) for xx in [x1, x2, x3, x4, xb])
            x = self.body.decode(xs)

        # return features
        return x

    @staticmethod
    def _get_model_body(body: str) -> backbones.Backbone:
        '''Retrieve model body by name.'''

        # model body registry
        body_registry = {
            'unet': backbones.UNet,
            'unetpp': backbones.UNetPP,
            'unetppp': backbones.UNetPPP
        }
        if not body in body_registry:
            raise ValueError(f'Invalid base model: {body}')
        return body_registry[body]
