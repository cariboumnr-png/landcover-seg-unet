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
Multi-head UNet architectures with optional domain conditioning.

This module provides UNet-based implementations built on top of the
multi-head model framework defined in :mod:`landseg.models.frames`.

The primary implementation, ``MultiHeadUNet``, combines:
- a configurable UNet-style backbone,
- shared feature extraction,
- multiple prediction heads,
- optional domain-aware conditioning,
- numerical safety utilities for stable training and inference.

Domain conditioning is supported through two complementary mechanisms:

- Concatenation conditioning:
  Domain embeddings are appended to the input tensor channels before
  feature extraction. This allows conditioning information to influence
  all encoder stages.

- FiLM conditioning:
  Feature-wise Linear Modulation is applied at the bottleneck using
  learned affine transforms derived from domain embeddings. This enables
  lightweight global adaptation without modifying earlier activations.

The implementation separates responsibilities across composable
components:

- ``MultiHeadUNet``:
    Coordinates backbone execution, domain routing, conditioning, and
    prediction heads.

- ``HeadManager``:
    Manages per-task output heads, active/frozen state, and optional
    logit adjustment.

- ``DomainContextRouter``:
    Routes domain identifiers and vectors to concatenation and/or FiLM
    pathways.

- ``NumericSafety``:
    Centralizes autocast handling and value clamping to improve
    numerical stability.

Expected tensor shapes:
- Input image: N, C_in, H, W)
- Shared feature map: (N, base_ch, H, W)
- Per-head logits: (N, C_head, H, W)

Notes:
- Logit adjustment tensors are stored as non-trainable buffers with shape
    ``(1, C_head, 1, 1)``.
- Backbone implementations are expected to expose compatible ``encode()``
    and ``decode()`` methods.
'''

# third-party imports
import torch
# local imports
import landseg._constants as c
import landseg.core as core
import landseg.models.backbones as backbones
import landseg.models.core as model_core
import landseg.models.frames as frames

class MultiHeadUNet(frames.MultiHeadBaseModel):
    '''
    Multi-head UNet model with optional domain conditioning.

    This model combines:
    - a configurable UNet-style backbone,
    - shared encoder-decoder feature extraction,
    - multiple task-specific prediction heads,
    - optional domain-aware conditioning mechanisms,
    - numerical safety utilities for stable execution.

    Supported conditioning mechanisms:
    - Input concatenation via ``ConcatAdapter``.
    - Bottleneck FiLM modulation via ``FilmConditioner``.

    Main components:
    - ``body``:
        Shared UNet backbone implementing ``encode()`` and ``decode()``.
    - ``heads``:
        Multi-head prediction manager producing task-specific logits.
    - ``domain_router``:
        Routes domain information to conditioning pathways.
    - ``num_safety``:
        Handles autocast control and activation clamping.
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
        Initialize a multi-head UNet model.

        This constructor assembles a complete domain-aware multi-head model
        using externally validated configuration objects. The model itself
        remains configuration-system agnostic and does not depend on Hydra
        or any global runtime state.

        The constructed model includes:
        - a shared UNet backbone,
        - task-specific prediction heads,
        - optional domain-conditioning pathways,
        - numerical safety utilities,
        - optional per-head logit adjustment buffers.

        Conditioning pathways may include:
        - input-level concatenation conditioning,
        - bottleneck FiLM conditioning,
        - both simultaneously,
        - or neither.

        Args:
            dataspecs: Dataset and model specification container
            backbone_config: Backbone configuration describing
            conditioning_config: Mapping of conditioning targets to
                domain-routing configurations
            **kwargs:
                Optional runtime configuration overrides.
                Supported options:
                - enable_clamp: Enable activation clamping for numerical
                    stability. Default: ``True``.
                - clamp_range: Tuple defining minimum and maximum clamp
                    bounds. Default: ``(1e-4, 1e4)``.

        Notes:
        - All arguments are keyword-only to make configuration boundaries
          explicit and order-independent.
        - Backbone implementations are expected to expose compatible
          ``encode()`` and ``decode()`` methods.
        - Logit adjustment tensors are registered as non-trainable buffers
          and broadcast during head inference.
        '''

        super().__init__()

        self.dataspecs = dataspecs
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
            self.concat = model_core.ConcatAdapter(concat_dim=add_dim)

        # core UNet body
        body = self._get_model_body(backbone_config.body)
        in_ch = dataspecs.meta.image_specs.num_channels
        self.body: backbones.UNetBackbone
        self.body = body(in_ch + add_dim, base_ch, **backbone_config.conv_params)

        # validate the spatial divisibility
        if dataspecs.meta.image_specs.height_width % body.spatial_divisor != 0:
            raise RuntimeError(
                'Input spatial size is incompatible with model.\n'
                f'Input size: {dataspecs.meta.image_specs.height_width}\n'
                f'Required divisor: {body.spatial_divisor}'
            )

        # film conditioning adpater
        film_config = conditioning_config.get('film')
        if film_config is None:
            self.film = None
        else:
            self.film = model_core.FilmConditioner(
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

    def build_dummy_batch(
        self,
        batch_size: int = 2,
        device: str = 'cpu'
    ) -> dict[str, torch.Tensor]:
        '''Construct synthetic input tensors for validation etc.'''

        in_ch = self.dataspecs.meta.image_specs.num_channels
        size = self.dataspecs.meta.image_specs.height_width
        ids_num = self.dataspecs.domains.ids_num
        vec_dim = self.dataspecs.domains.vec_dim

        return_dict: dict[str, torch.Tensor] = {}

        # dummy input tensor [B, C, S, S] assume H==W
        return_dict['x'] = torch.randn(
            (batch_size, in_ch, size, size),
            device=device,
            dtype=torch.float32
        )
        # dummy domains (optional depending on dataspecs)
        if ids_num > 0:
            return_dict['ids_domain'] = torch.randint(
                0,
                ids_num,
                (batch_size,),
                dtype=torch.long,
                device=device
            )
        if vec_dim > 0:
            return_dict['vec_domain'] = torch.randn(
                (batch_size, vec_dim),
                dtype=torch.float32,
                device=device
            )

        return return_dict

    def _forward_features(
        self,
        x: torch.Tensor,
        domains: dict[str, model_core.DomainTargetPayload]
    ) -> torch.Tensor:
        '''
        Compute shared UNet feature maps before prediction heads.

        The feature pipeline performs:
        1. Optional domain concatenation at the input level.
        2. Encoder feature extraction through the UNet body.
        3. Optional FiLM conditioning at the bottleneck.
        4. Decoder reconstruction into shared output features.

        Numerical clamping and controlled autocast execution are applied
        throughout the pipeline to improve numerical stability.

        Args:
            x: Input tensor of shape: (batch, channels, height, width)
            domains: Routed domain payloads from the domain router.

        Returns:
            Shared decoder feature tensor consumed by output heads.
        '''

        # feed domain to router
        concat = domains.get('concat')
        film = domains.get('film')

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
    def _get_model_body(body: str) -> backbones.UNetBackbone:
        '''Retrieve a registered UNet backbone implementation.'''

        # model body registry
        body_registry = {
            'unet': backbones.UNet,
            'unetpp': backbones.UNetPP,
            'unetppp': backbones.UNetPPP
        }
        if not body in body_registry:
            raise ValueError(f'Invalid base model: {body}')
        return body_registry[body]
