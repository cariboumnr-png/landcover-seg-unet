'''FiLM conditioning.'''

# third-party imports
import torch
import torch.nn
# local imports
import models.multihead

class FilmConditioner(torch.nn.Module):
    '''
    Build domain embeddings and FiLM the bottleneck.
    '''

    def __init__(
            self,
            embed_dim: int,
            num_categories: int,
            dim_continuous: int,
            **kwargs):
        '''
        Args:
            embed_dim (int): Width of the domain embedding
                (0 disables FiLM).
            num_categories (int): Number of categorical domain classes
                (0 disables).
            dim_continuous (int): Dimensionality of continuous domain
                vector (0 disables).

            kwargs:
            bottleneck_ch (int): Channel count of UNet bottleneck
                feature map to FiLM.
            hidden (int): Hidden width of the FiLM MLP; if falsy,
                defaults to embed_dim.
        '''
        super().__init__()

        # parse arguments
        bottleneck_ch = kwargs.get('bottleneck_ch', 16)
        hidden = kwargs.get('hidden', 128)

        # early exit
        self.embed_dim = embed_dim
        if embed_dim == 0:  # FiLM disabled
            self.domain_embed = None
            self.domain_mlp = None
            self.bottom_film = None
            return

        # categorical
        self.domain_embed = torch.nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=embed_dim
        ) if num_categories > 0 else None
        # continuous
        self.domain_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_continuous, embed_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(embed_dim, embed_dim)
        ) if dim_continuous > 0 else None

        # FiLM generator
        h = hidden or embed_dim
        self.bottom_film = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, h),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(h, 2 * bottleneck_ch),
        )
        for m in self.bottom_film.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def embed(
            self,
            domain_ids: torch.Tensor | None=None,
            domain_vec: torch.Tensor | None=None
        ) -> torch.Tensor | None:
        '''
        Build domain embedding z (shape [B, embed_dim]) from:
            - domain_ids (categorical) via Embedding, and/or
            - domain_vec (continuous) via MLP.

        Returns:
            z (Tensor of shape [B, embed_dim]) or None if FiLM disabled
            or if neither pathway produced an embedding.
        '''

        if self.embed_dim == 0:
            return None
        z = None
        if domain_ids is not None and self.domain_embed is not None:
            z = self.domain_embed(domain_ids)
        if domain_vec is not None and self.domain_mlp is not None:
            zv = self.domain_mlp(domain_vec)
            z = zv if z is None else z + zv
        if z is None:
            return None
        return torch.nn.functional.layer_norm(z, (z.shape[-1],))

    def film_bottleneck(
            self,
            xb: torch.Tensor,
            z: torch.Tensor | None=None
        ) -> torch.Tensor:
        '''
        Apply FiLM to bottleneck features xb using embedding z.

        Shapes:
            xb: [B, Cb, H, W] with Cb == bottleneck_ch
            z:  [B, embed_dim]

        Returns:
            xb' = xb * (1 + gamma) + beta, same shape as xb.

        If FiLM disabled or z is None, returns xb unchanged.
        '''

        if self.embed_dim == 0 or z is None:
            return xb
        assert self.bottom_film is not None
        gb = self.bottom_film(z)              # [B, 2*Cb]
        gamma, beta = gb.chunk(2, dim=1)      # [B, Cb]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return xb * (1.0 + gamma) + beta


    def forward(
            self,
            xb: torch.Tensor,
            domain_ids: torch.Tensor | None=None,
            domain_vec: torch.Tensor | None=None,
        ) -> torch.Tensor:
        '''
        One-call interface:
        1) Build embedding z from (domain_ids, domain_vec)
        2) FiLM-modulate the bottleneck feature map xb

        Args:
            xb: Bottleneck feature map, shape [B, Cb, H, W]
            domain_ids: Optional categorical IDs, shape [B] or [B, ...]
            domain_vec: Optional continuous features, shape [B, D]

        Returns:
            xb' (FiLM-modulated) with same shape as xb.
        '''

        z = self.embed(domain_ids=domain_ids, domain_vec=domain_vec)
        return self.film_bottleneck(xb, z)

def get_film(
        config: models.multihead.CondConfig,
        base_ch: int
    ) -> FilmConditioner | None:
    '''Generate a film conditioner for the model.'''

    if config.mode in ['film', 'hybrid'] and config.film.embed_dim > 0:
        return FilmConditioner(
            embed_dim=config.film.embed_dim,
            num_categories=config.domain_ids_num,
            dim_continuous=config.domain_vec_dim,
            bottleneck_ch=base_ch * 16,
            hidden=config.film.hidden
        )
    return None
