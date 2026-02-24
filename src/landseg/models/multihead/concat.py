'''Domain knowledge concat.'''

# third-party imports
import torch
import torch.nn
# local imports
import landseg.models.multihead as multihead

class ConcatAdapter(torch.nn.Module):
    '''
    Compress and broadcast domain info -> D channels for input concatenation.
    Supports:
      - Continuous domain vectors (optionally via MLP).
      - Categorical domain IDs (via Embedding).
      - If both provided, they are added together.
    '''

    def __init__(
        self,
        out_dom: int,
        dim_continuous: int,
        num_categories: int,
        use_mlp: bool,
    ):
        '''Init.'''
        super().__init__()

        self.input_dim = dim_continuous
        self.output_dim = out_dom

        # Optional MLP for continuous domain vectors
        if out_dom > 0 and use_mlp:
            assert dim_continuous > 0, 'in_dim must be > 0 when use_mlp=True'
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim_continuous, out_dom),
                torch.nn.ReLU(inplace=True)
            )
        else:
            self.mlp = None

        # Optional embedding for categorical domain
        if out_dom > 0 and num_categories > 0:
            self.embedding = torch.nn.Embedding(num_categories, out_dom)
        else:
            self.embedding = None

    def forward(
        self,
        x: torch.Tensor,
        domain_ids: torch.Tensor | None = None,        # [B] LongTensor
        domain_vectors: torch.Tensor | None = None,  # [B, in_dim] or [B, out_dom] if no MLP
    ) -> torch.Tensor:
        '''Forward.'''
        # No extra domain channels requested -> return input unchanged
        if self.output_dim <= 0:
            return x

        # Must have at least one source of domain info
        if domain_vectors is None and domain_ids is None:
            raise AssertionError('Provide domain_vectors or domain_ids when out_dom>0')

        # Build continuous part (if provided)
        dv_cont = None
        if domain_vectors is not None:
            if self.mlp is not None:
                assert domain_vectors.shape[1] == self.input_dim, \
                    f'Expected domain_vectors dim={self.input_dim}, \
                        got {domain_vectors.shape[1]}'
                dv_cont = self.mlp(domain_vectors.to(x.device))
            else:
                # If no MLP, domain_vectors must already match out_dom
                assert domain_vectors.shape[1] == self.output_dim, \
                    f'Expected domain_vectors dim={self.output_dim} (no MLP), \
                        got {domain_vectors.shape[1]}'
                dv_cont = domain_vectors.to(x.device)

        # Build categorical part (if provided)
        dv_cat = None
        if domain_ids is not None:
            if self.embedding is None:
                raise AssertionError('num_categories must be >0 to use domain_ids')
            dv_cat = self.embedding(domain_ids.to(x.device))

        # Combine: add if both present, else whichever is present
        if dv_cont is not None and dv_cat is not None:
            dv = dv_cont + dv_cat
        else:
            dv = dv_cont if dv_cont is not None else dv_cat

        # Broadcast to spatial and concat
        b, _, h, w = x.shape
        assert dv is not None
        dv = dv.view(b, self.output_dim, 1, 1).expand(b, self.output_dim, h, w)
        return torch.cat([x, dv], dim=1)


def get_concat(config: multihead.CondConfig) -> ConcatAdapter | None:
    '''Generate a concat adapter for the model.'''

    if config.mode in ['concat', 'hybrid'] and config.concat.out_dim > 0:
        return ConcatAdapter(
            out_dom=config.concat.out_dim,
            dim_continuous=config.domain_vec_dim,
            num_categories=config.domain_ids_num,
            use_mlp=config.concat.use_mlp
        )
    return None
