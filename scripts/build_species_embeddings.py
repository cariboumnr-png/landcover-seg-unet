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
import argparse
import pathlib
# local imports
import landseg.knowledge as knowledge


def main() -> None:
    '''CLI entrypoint for building species embedding space.'''
    parser = argparse.ArgumentParser(
        description=(
            'Generate Sentence Transformer embeddings from species CSV '
            'knowledge base.'
        )
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default='knowledge/ontario_tree_species_grouped_profiles.csv',
        help='Path to species/grouped profiles CSV',
    )
    parser.add_argument(
        '--knowledge-root',
        type=str,
        default='knowledge',
        help='Knowledge base root directory',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='BAAI/bge-base-en-v1.5',
        help=(
            'Sentence Transformer model name from HuggingFace '
            '(e.g. BAAI/bge-base-en-v1.5)'
        ),
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable L2 normalization of embedding vectors',
    )

    args = parser.parse_args()

    knowledge.generate_embeddings_and_matrix(
        csv_path=pathlib.Path(args.csv_path),
        knowledge_root=args.knowledge_root,
        model_name=args.model_name,
        normalize=not args.no_normalize,
    )


if __name__ == '__main__':
    main()
