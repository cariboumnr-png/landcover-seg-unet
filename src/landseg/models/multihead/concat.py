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
Domain-feature concatenation adapters for conditional models. Provides a
module to compress/broadcast domain info and append it as extra channels.

Public APIs:
    - ConcatAdapter: Torch module that projects domain and concatenates
      it to input feature maps.
    - get_concat: Factory that builds an adapter from model config.
'''

# third-party imports
import torch
import torch.nn
# local imports
import landseg.models.multihead as multihead

class ConcatAdapter(torch.nn.Module):
    '''
    Compress and broadcast domain info to D ch for input concatenation.

    Supports:
      - Continuous domain vectors (optionally via MLP).
      - Categorical domain IDs (via Embedding).
      - If both provided, they are added together.

    Args:
        out_dom: Number of domain channels to append to the input.
        dim_continuous: Dimensionality of input domain vectors.
        num_categories: Number of categorical domain IDs for embedding.
        use_mlp: If True, apply MLP to domain vectors to reach out_dom.

    Raises:
        valueError: If use_mlp=True but dim_continuous<=0.
        valueError: If forward() receives neither vectors nor IDs.
        valueError: If vector dims mismatch expected sizes.
        valueError: If domain IDs are given but no embedding defined.
    '''

    def __init__(
        self,
        out_dom: int,
        dim_continuous: int,
        num_categories: int,
        use_mlp: bool,
    ):
        '''
        Initialize the domain-feature adapter.

        Args:
            out_dom: Number of domain channels to append to the input.
            dim_continuous: Dimensionality of input domain vectors.
            num_categories: No. of categorical domain IDs for embedding.
            use_mlp: If True, project domain vectors to out_dom via an
                MLP; otherwise vectors must already have size out_dom.

        Raises:
            ValueError: If use_mlp is True but dim_continuous <= 0.
        '''
        super().__init__()

        self.input_dim = dim_continuous
        self.output_dim = out_dom

        # Optional MLP for continuous domain vectors
        if out_dom > 0 and use_mlp:
            if dim_continuous <= 0:
                raise ValueError('in_dim must be > 0 when use_mlp=True')
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
        domain_ids: torch.Tensor | None = None,
        domain_vectors: torch.Tensor | None = None
    ) -> torch.Tensor:
        '''
        Concatenate projected/broadcast domain features to input tensor.

        Args:
            x: Input feature map of shape [B, C, H, W].
            domain_ids: Optional categorical domain IDs of shape [B].
            domain_vectors: Optional continuous domain vectors of shape
                [B, dim_continuous] when use_mlp=True, or [B, out_dom]
                when use_mlp=False.

        Returns:
            torch.Tensor: Output feature map of shape [B, C + out_dom,
                H, W] when out_dom > 0; otherwise returns x unchanged.

        Raises:
            ValueError: If out_dom > 0 and neither domain_vectors nor
                domain_ids is provided.
            ValueError: If provided domain_vectors have mismatched
                dimension for the configured mode (with/without MLP).
            ValueError: If domain_ids are provided but no embedding is
                defined.
        '''
        # No extra domain channels requested -> return input unchanged
        if self.output_dim <= 0:
            return x

        # Must have at least one source of domain info
        if domain_vectors is None and domain_ids is None:
            raise ValueError('No domain_vectors or domain_ids when out_dom>0')

        # Build continuous part (if provided)
        dv_cont = None
        if domain_vectors is not None:
            if self.mlp is not None:
                if domain_vectors.shape[1] != self.input_dim:
                    raise ValueError(
                        f'Expected domain_vectors dim={self.input_dim}, '
                        f'got {domain_vectors.shape[1]}'
                    )
                dv_cont = self.mlp(domain_vectors.to(x.device))
            else:
                # If no MLP, domain_vectors must already match out_dom
                if domain_vectors.shape[1] != self.output_dim:
                    raise ValueError(
                        f'Expected domain_vectors dim={self.output_dim} '
                        f'(no MLP), got {domain_vectors.shape[1]}'
                    )
                dv_cont = domain_vectors.to(x.device)

        # Build categorical part (if provided)
        dv_cat = None
        if domain_ids is not None:
            if self.embedding is None:
                raise ValueError('num_categories must be >0 to use domain_ids')
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


def get_concat(config: multihead.ConditioningConfig) -> ConcatAdapter | None:
    '''
    Build a ConcatAdapter from a conditional-config, if enabled.

    Args:
        config: Conditional model configuration with related settings.

    Returns:
        (ConcatAdapter | None): Adapter when mode is `"concat"` or
            `"hybrid"` and out_dim > 0; otherwise None.
    '''

    if config.mode in ['concat', 'hybrid'] and config.concat.out_dim > 0:
        return ConcatAdapter(
            out_dom=config.concat.out_dim,
            dim_continuous=config.domain_vec_dim,
            num_categories=config.domain_ids_num,
            use_mlp=config.concat.use_mlp
        )
    return None
