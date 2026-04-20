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
import torch.nn
# local imports
import landseg._constants as c
import landseg.models.backbones as backbones
import landseg.models.multihead as multihead

class MultiHeadUNet(multihead.BaseMultiheadModel):
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
        dataspecs_config: multihead.DataSpecsConfig,
        backbone_config: multihead.BackboneConfig,
        conditioning: multihead.ConditioningConfig,
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

        # channels
        in_ch = dataspecs_config.in_ch
        base_ch = backbone_config.base_ch

        # logit adjustments
        # scalar strength alpha (1.0 = as-provided priors) for logit adjust
        self.register_buffer('la_alpha', torch.tensor(1.0, dtype=torch.float32))
        # register perhead logit adjustment as buffers (NOT parameters)
        if dataspecs_config.logit_adjust:
            for h, v in dataspecs_config.logit_adjust.items():
                t = torch.tensor(v, dtype=torch.float32).view(1, -1, 1, 1)
                self.register_buffer(f'la_{h}', t)
        # runtime toggle whether to use la for inference (init state)
        self.enable_logit_adjust = kwargs.get('enable_logit_adjust', True)

        # multihead management
        heads = {k: len(v) for k, v in dataspecs_config.heads_w_counts.items()}
        self.heads = _HeadManager(base_ch, heads)

        # domain knowledge router
        self.domain_router = _DomainRouter(dataspecs_config, conditioning)

        # domain concatenation if proviced
        self.concat = multihead.get_concat(
            conditioning,
            domain_ids_num=dataspecs_config.domain_ids_num,
            domain_vec_dim=dataspecs_config.domain_vec_dim
        )
        add = self.concat.output_dim if self.concat is not None else 0

        # core UNet body
        body = self._get_model_body(backbone_config.body)
        self.body = body(in_ch + add, base_ch, **backbone_config.conv_params)

        # film conditioner
        self.film = multihead.get_film(
            base_ch,
            conditioning,
            domain_ids_num=dataspecs_config.domain_ids_num,
            domain_vec_dim=dataspecs_config.domain_vec_dim
        )

        # safety utilities
        self.safety = _NumericSafety(
            enable_clamp=kwargs.get('enable_clamp', True),
            clamp_range=kwargs.get('clamp_range', (1e-4, 1e4))
        )

    @property
    def logit_adjust_alpha(self) -> float:
        '''Returns Global logit adjust alpha scalar.'''
        return float(getattr(self, 'la_alpha').item())

    @property
    def logit_adjust(self) -> dict[str, torch.Tensor]:
        '''
        Returns Lazily gather per-head logit adjustment buffers.

        Buffers are named 'la_{head}'. Excludes the scalar 'la_alpha'.
        '''
        out: dict[str, torch.Tensor] = {}
        for name, buf in self.named_buffers():
            if name.startswith('la_') and name != 'la_alpha':
                head = name.removeprefix('la_')
                out[head] = buf
        return out

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        '''Compute per-head logits with optional domain info.'''

        # numeric safty
        assert torch.isfinite(x).all(), "Input has NaN/Inf"

        # get domain if provided
        dom_ids = kwargs.get('ids', None)
        dom_vec = kwargs.get('vec', None)
        if dom_ids is not None:
            assert isinstance(dom_ids, torch.Tensor)
            # assert dom_ids.type() is torch.int64, dom_ids
        if dom_vec is not None:
            assert isinstance(dom_vec, torch.Tensor)

        # feed domain to router
        concat, film = self.domain_router.forward(dom_ids, dom_vec)

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

        # return head outputs
        return self.heads.forward(
            x,
            self.heads.active,
            self.logit_adjust,
            self.logit_adjust_alpha,
            enable_la=self.enable_logit_adjust
        )

    def set_active_heads(self, active_heads: list[str] | None=None) -> None:
        '''Set the list of active heads used during forward.'''

        self.heads.active = active_heads

    def set_frozen_heads(self, frozen_heads: list[str] | None=None) -> None:
        '''Freeze parameters for selected heads.'''

        self.heads.frozen = frozen_heads
        self.heads.freeze(frozen_heads)

    def reset_heads(self):
        '''Clear active/frozen head selections.'''

        self.heads.active = None
        self.heads.frozen = None

    def set_logit_adjust_enabled(self, enabled: bool):
        '''External toggle on logit adjustment'''

        self.enable_logit_adjust = enabled

    def set_logit_adjust_alpha(self, alpha: float):
        '''Set logit adjust alpha.'''

        la_alpha: torch.Tensor = getattr(self, 'la_alpha')
        la_alpha.fill_(float(alpha))

    @staticmethod
    def _get_model_body(body: str) -> backbones.Backbone:
        '''Retrieve model body by name.'''

        # model body registry
        body_registry = {
            'unet': backbones.UNet,
            'unetpp': backbones.UNetPP
        }
        if not body in body_registry:
            raise ValueError(f'Invalid base model: {body}')
        return body_registry[body]

# internal pieces
class _HeadManager(torch.nn.Module):
    '''
    Manage multiple 1x1 conv heads, activation state, and freezing.

    Each head as `Conv2d(in_ch → num_classes, kernel_size=1)` producing
    per-pixel logits. The manager tracks which heads are active for
    forward passes and supports freezing selected heads' parameters.
    '''

    def __init__(self, in_ch: int, heads: dict[str, int]):
        '''
        Create per-head 1x1 convs and initialize head state.

        Args:
            in_ch: Channel of the shared feature map feeding all heads.
            heads: Mapping from head's name to head's number of classes.
        '''

        super().__init__()
        # output convolution block
        self.outc = torch.nn.ModuleDict({
            head_name: torch.nn.Conv2d(in_ch, num_classes, kernel_size=1)
            for head_name, num_classes in heads.items()
        })
        self.active: list[str] | None = list(self.outc.keys())
        self.frozen: list[str] | None = None

    def forward(
        self,
        x: torch.Tensor,
        active_heads: list[str] | None,
        logit_adjust: dict[str, torch.Tensor],
        logit_adjust_alpha: float,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        '''Run active heads and return a dict of logits.'''

        # if external active heads provided
        if active_heads is not None:
            self.active = active_heads
        # reset logic
        if self.active is None:
            self.active = list(self.outc.keys())
        # parse from kwargs
        use_la = bool(kwargs.get('enable_la', False))

        # iterate through active heads
        output_logits: dict[str, torch.Tensor] = {}
        for head in self.active:
            conv = self.outc[head]
            logits = conv(x)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            # apply logit adjustment
            output_logits[head] = self._apply_logit_adjust(
                head,
                logits,
                logit_adjust,
                use_la=use_la,
                la_alpha=logit_adjust_alpha
            )
        return output_logits

    def freeze(self, frozen_heads: list[str] | None = None) -> None:
        '''Disable gradients of selected heads.'''
        if frozen_heads is None:
            return
        for h in frozen_heads:
            for p in self.outc[h].parameters():
                p.requires_grad = False

    @staticmethod
    def _apply_logit_adjust(
        head: str,
        logits: torch.Tensor,
        logit_adjust: dict[str, torch.Tensor],
        *,
        use_la: bool | None = None,
        la_alpha: float | None = None,
    ) -> torch.Tensor:
        '''
        Apply logit adjustment if enabled and available for this head.
        '''

        prior = logit_adjust.get(head)
        a = float(la_alpha) if la_alpha is not None else 1.0
        # early exit
        if not use_la or prior is None or a == 0.0:
            return logits
        # apply la alpha if provided
        return logits + a * prior

class _DomainRouter(torch.nn.Module):
    '''
    Route domain information to concatenation and FiLM branches.

    This class determines which parts of the provided domain identifiers
    (`ids`) and vectors (`vec`) should be used for:
    - **Concatenation**: optionally project `vec` to `concat.out_dim`
        before channel-wise concatenation at the input.
    - **FiLM**: optionally project `vec` to `film.embed_dim` before
        computing FiLM parameters at the bottleneck.

    Projections are created only if a raw domain vector dimension is
    provided and the corresponding branch (`concat` or `film`) is
    configured to use it.
    '''

    def __init__(
        self,
        dataspecs_config: multihead.DataSpecsConfig,
        conditioning_config: multihead.ConditioningConfig
    ):
        '''
        Initialize optional projections and record routing policy.

        Args:
            cfg: Conditioning configuration describing whether to use
                `ids` and/or `vec` for each branch, and the target
                dimensions for projections.
        '''

        super().__init__()
        self.cfg = conditioning_config
        # set concat and film projection
        self.concat_proj = torch.nn.Linear(
            in_features=dataspecs_config.domain_vec_dim,
            out_features=conditioning_config.concat.out_dim
        ) if dataspecs_config.domain_vec_dim else None
        self.film_proj = torch.nn.Linear(
            in_features=dataspecs_config.domain_vec_dim,
            out_features=conditioning_config.film.embed_dim
        ) if dataspecs_config.domain_vec_dim else None

    def forward(
        self,
        ids: torch.Tensor | None,
        vec: torch.Tensor | None
    ) -> tuple[tuple, tuple]:
        '''Return domain routing according to configuration.'''

        # Decide and shape what goes to concat
        concat_ids = ids if self.cfg.concat.use_ids else None
        # if to use vec for concat and vec is provided
        if self.cfg.concat.use_vec and vec is not None:
            if self.concat_proj is not None:
                concat_vec = self.concat_proj(vec)
            else:
                concat_vec = vec
        else:
            concat_vec = None

        # Decide and shape what goes to film
        film_ids = ids if self.cfg.film.use_ids else None
        # if use vec for fil, and vec is provided
        if self.cfg.film.use_vec and vec is not None:
            if self.film_proj is not None:
                film_vec = self.film_proj(vec)
            else:
                film_vec = vec
        else:
            film_vec = None

        # return
        return (concat_ids, concat_vec), (film_ids, film_vec)

class _NumericSafety():
    '''Autocast and clamping utilities for numerical stability.'''
    def __init__(
        self,
        *,
        enable_clamp: bool,
        clamp_range: tuple[float, float]
    ):
        '''Configure clamping behavior and bounds.'''

        self.enable_clamp = enable_clamp
        self.clamp_range = clamp_range

    def autocast_context(
        self,
        enable: bool = True,
        dtype: torch.dtype = torch.float16
    ) -> torch.autocast:
        '''Create an AMP autocast context for the current device.'''

        return torch.autocast(c.DEVICE, dtype, enable)


    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        '''Clamp tensor values to a safe numeric range.'''

        if not self.enable_clamp:
            return x
        mmin, mmax = self.clamp_range
        return torch.clamp(x, min=mmin, max=mmax)
