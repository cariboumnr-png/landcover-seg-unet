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
import landseg.models.backbones as backbones
import landseg.models.multihead as multihead

class MultiHeadUNet(multihead.BaseMultiheadModel):
    '''
    UNet with 4 down/up levels and multi heads and conditioning support.

    Hybrid domain conditioning support:
      - Concatenation at input (if concat_domain_dim > 0 & mode in
      {'concat','hybrid'})
      - FiLM at bottleneck (if domain_embed_dim set & mode in
      {'film','hybrid'})
    '''

    body_registry = {
        'unet': backbones.UNet,
        'unet++': backbones.UNetPP
    }

    def __init__(
            self,
            body: str,
            config: multihead.ModelConfig,
            *,
            enable_logit_adjust: bool = True,
            enable_clamp: bool = True
        ):
        super().__init__()

        # channels
        in_ch = config.in_ch
        base_ch = config.base_ch

        # logit adjustments
        # scalar strength alpha (1.0 = as-provided priors) for logit adjust
        self.register_buffer('la_alpha', torch.tensor(1.0, dtype=torch.float32))
        # register perhead logit adjustment as buffers (NOT parameters)
        if config.logit_adjust:
            for h, v in config.logit_adjust.items():
                t = torch.tensor(v, dtype=torch.float32).view(1, -1, 1, 1)
                self.register_buffer(f'la_{h}', t)
        # runtime toggle whether to use la for inference (init state)
        self.enable_logit_adjust = bool(enable_logit_adjust)

        # multihead management
        heads_w_num_cls = {k: len(v) for k, v in config.heads_w_counts.items()}
        self.heads = _HeadManager(in_ch=base_ch, heads=heads_w_num_cls)

        # domain knowledge router
        self.domain_router = _DomainRouter(config.conditioning)

        # domain concatenation if proviced
        self.concat = multihead.get_concat(config.conditioning)
        add = self.concat.output_dim if self.concat is not None else 0

        # core UNet body
        assert body in self.body_registry, f'Invalid body type: {body}'
        self.body = self.body_registry[body](in_ch + add, base_ch)

        # conditioner
        self.film = multihead.get_film(config.conditioning, base_ch)

        # safety utilities
        self.safety = _NumericSafety(enable_clamp, config.clamp_range)

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
            self.heads.state.active,
            self.logit_adjust,
            self.logit_adjust_alpha,
            enable_la=self.enable_logit_adjust
        )

    def set_active_heads(self, active_heads: list[str] | None=None) -> None:
        '''Set the list of active heads used during forward.'''

        self.heads.state.active = active_heads

    def set_frozen_heads(self, frozen_heads: list[str] | None=None) -> None:
        '''Freeze parameters for selected heads.'''

        self.heads.state.frozen = frozen_heads
        self.heads.freeze(frozen_heads)

    def reset_heads(self):
        '''Clear active/frozen head selections.'''

        self.heads.state.active = None
        self.heads.state.frozen = None

    def set_logit_adjust_enabled(self, enabled: bool):
        '''External toggle on logit adjustment'''

        self.enable_logit_adjust = enabled

    def set_logit_adjust_alpha(self, alpha: float):
        '''Set logit adjust alpha.'''

        la_alpha: torch.Tensor = getattr(self, 'la_alpha')
        la_alpha.fill_(float(alpha))

    @property
    def logit_adjust_alpha(self) -> float:
        '''Global logit adjust alpha scalar.'''
        return float(getattr(self, 'la_alpha').item())

    @property
    def logit_adjust(self) -> dict[str, torch.Tensor]:
        '''
        Lazily gather per-head logit adjustment buffers. Buffers are
        named 'la_{head}'. Excludes the scalar 'la_alpha'.
        '''
        out: dict[str, torch.Tensor] = {}
        for name, buf in self.named_buffers():
            if name.startswith('la_') and name != 'la_alpha':
                head = name.removeprefix('la_')
                out[head] = buf
        return out

    @property
    def encoder(self) -> list:
        '''Return a list of encoder blocks (incl. bottleneck).'''
        return [self.body.inc, *self.body.downs, self.body.bottleneck]

    @property
    def decoder(self) -> list:
        '''Return a list of decoder (upsampling) blocks.'''
        return [*self.body.ups]

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
        self.state = multihead.HeadsState()
        self.state.active = list(self.outc.keys())
        self.state.frozen = None

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
            self.state.active = active_heads
        # reset logic
        if self.state.active is None:
            self.state.active = list(self.outc.keys())
        # parse from kwargs
        use_la = bool(kwargs.get('enable_la', False))

        # iterate through active heads
        output_logits: dict[str, torch.Tensor] = {}
        for head in self.state.active:
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

    def freeze(self, frozen_heads: list[str] | None=None) -> None:
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

    def __init__(self, cfg: multihead.CondConfig):
        '''
        Initialize optional projections and record routing policy.

        Args:
            cfg: Conditioning configuration describing whether to use
                `ids` and/or `vec` for each branch, and the target
                dimensions for projections.
        '''

        super().__init__()
        self.cfg = cfg
        # set concat and film projection
        self.concat_proj = torch.nn.Linear(
            in_features=cfg.domain_vec_dim,
            out_features=cfg.concat.out_dim
        ) if cfg.domain_vec_dim else None
        self.film_proj = torch.nn.Linear(
            in_features=cfg.domain_vec_dim,
            out_features=cfg.film.embed_dim
        ) if cfg.domain_vec_dim else None

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
        enable_clamp: bool,
        clamp_range: tuple[float, float]
    ):
        '''Configure clamping behavior and bounds.'''

        self.enable_clamp = enable_clamp
        self.clamp_range = clamp_range

    def autocast_context(
        self,
        enable: bool=True,
        dtype: torch.dtype=torch.float16
    ) -> torch.autocast:
        '''Create an AMP autocast context for the current device.'''

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.autocast(device_type, dtype, enable)


    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        '''Clamp tensor values to a safe numeric range.'''

        if not self.enable_clamp:
            return x
        mmin, mmax = self.clamp_range
        return torch.clamp(x, min=mmin, max=mmax)
