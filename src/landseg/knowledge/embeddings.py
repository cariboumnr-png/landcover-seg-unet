# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Generate Sentence Transformer embeddings for tree species ecological
profiles.
'''

# standard imports
import pathlib
import typing
# third-party imports
import numpy
import pandas
import sentence_transformers
import torch
# local imports
import landseg.artifacts as artifacts


class SpeciesEntry(typing.TypedDict):
    '''Per-species code and text prompt dictionary.'''
    index: int
    code: str
    name: str
    text_prompt: str


class SpeciesEmbeddingsMetadata(typing.TypedDict):
    '''Species embeddings metadata dictionary.'''
    model_name: str
    num_classes: int
    embedding_dim: int
    normalized: bool
    key_column: str
    classes: list[SpeciesEntry]


def generate_embeddings_and_matrix(
    csv_path: str | pathlib.Path,
    knowledge_root: str | pathlib.Path = 'knowledge',
    *,
    model_name: str = 'BAAI/bge-base-en-v1.5',
    normalize: bool = True,
) -> None:
    '''
    Generate Sentence Transformer embeddings and similarity matrix from
    species CSV.

    Args:
        csv_path: Path to species/grouped profiles CSV.
        knowledge_root: Root directory path for knowledge base artifacts.
        model_name: HuggingFace sentence transformer model name.
        normalize: Whether to L2-normalize embeddings for cosine similarity.
    '''
    csv_path = pathlib.Path(csv_path)
    csv_stem = csv_path.stem
    kp = artifacts.KnowledgePaths(root=knowledge_root)

    target_dir = kp.profile_dpath(csv_stem)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading species profiles from: {csv_path}')
    print(f'Output artifacts target directory: {target_dir}')
    df = pandas.read_csv(csv_path)

    # determine primary key column (group_code or species_code)
    key_col = 'group_code' if 'group_code' in df.columns else 'species_code'
    name_col = 'group_name' if 'group_name' in df.columns else 'common_name'
    text_col = 'formatted_description'

    keys: list[str] = [str(x) for x in df[key_col]]
    names: list[str] = [str(x) for x in df[name_col]]
    descriptions: list[str] = [str(x) for x in df[text_col]]

    print(f'Loaded {len(keys)} entries. Encoding using "{model_name}"...')
    model = sentence_transformers.SentenceTransformer(model_name)
    embeddings_np = model.encode(
        descriptions, normalize_embeddings=normalize, show_progress_bar=True
    )

    # convert to torch tensor [N, D]
    embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32)

    # compute N x N cosine similarity matrix [N, N]
    if normalize:
        similarity_np = numpy.dot(embeddings_np, embeddings_np.T)
    else:
        norm = numpy.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norm_embeddings = embeddings_np / (norm + 1e-8)
        similarity_np = numpy.dot(norm_embeddings, norm_embeddings.T)

    similarity_tensor = torch.tensor(similarity_np, dtype=torch.float32)

    # prepare metadata dictionary
    metadata: SpeciesEmbeddingsMetadata = {
        'model_name': model_name,
        'num_classes': len(keys),
        'embedding_dim': int(embeddings_tensor.shape[1]),
        'normalized': normalize,
        'key_column': key_col,
        'classes': [
            {
                'index': i,
                'code': keys[i],
                'name': names[i],
                'text_prompt': descriptions[i],
            }
            for i in range(len(keys))
        ],
    }

    # save output artifacts via canonical KnowledgePaths
    emb_path = kp.embeddings_fpath(csv_stem)
    sim_path = kp.similarity_matrix_fpath(csv_stem)
    meta_path = kp.metadata_fpath(csv_stem)
    sim_csv_path = kp.similarity_csv_fpath(csv_stem)

    artifacts.Controller[torch.Tensor](emb_path).persist(embeddings_tensor)
    artifacts.Controller[torch.Tensor](sim_path).persist(similarity_tensor)
    artifacts.Controller[SpeciesEmbeddingsMetadata](meta_path).persist(metadata)

    # save similarity matrix as readable CSV for inspection
    sim_df = pandas.DataFrame(
        similarity_np, index=pandas.Index(keys), columns=pandas.Index(keys)
    )
    artifacts.Controller[pandas.DataFrame](sim_csv_path).persist(sim_df)

    print('\n--- Artifacts Successfully Generated ---')
    print(
        f'Embeddings Tensor [N={len(keys)}, D={embeddings_tensor.shape[1]}]: '
        f'{emb_path}'
    )
    print(
        f'Similarity Matrix Tensor [N={len(keys)}, N={len(keys)}]: '
        f'{sim_path}'
    )
    print(f'Readable Similarity CSV: {sim_csv_path}')
    print(f'Class Metadata JSON: {meta_path}')
