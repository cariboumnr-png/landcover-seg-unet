"""
Generate Sentence Transformer embeddings for tree species ecological
profiles.
"""

# standard imports
import argparse
import json
import pathlib
# third-party imports
import numpy
import pandas
import sentence_transformers
import torch


def generate_embeddings_and_matrix(
    csv_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    model_name: str = "BAAI/bge-base-en-v1.5",
    normalize: bool = True,
) -> None:
    """
    Generate Sentence Transformer embeddings and similarity matrix from
    species CSV.
    """
    csv_path = pathlib.Path(csv_path)
    csv_stem = csv_path.stem
    target_dir = pathlib.Path(output_dir) / csv_stem
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading species profiles from: {csv_path}")
    print(f"Output artifacts target directory: {target_dir}")
    df = pandas.read_csv(csv_path)

    # determine primary key column (group_code or species_code)
    key_col = "group_code" if "group_code" in df.columns else "species_code"
    name_col = "group_name" if "group_name" in df.columns else "common_name"
    text_col = "formatted_description"

    keys: list[str] = [str(x) for x in df[key_col]]
    names: list[str] = [str(x) for x in df[name_col]]
    descriptions: list[str] = [str(x) for x in df[text_col]]

    print(f"Loaded {len(keys)} entries. Encoding using '{model_name}'...")
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
    metadata = {
        "model_name": model_name,
        "num_classes": len(keys),
        "embedding_dim": int(embeddings_tensor.shape[1]),
        "normalized": normalize,
        "key_column": key_col,
        "classes": [
            {
                "index": i,
                "code": keys[i],
                "name": names[i],
                "text_prompt": descriptions[i],
            }
            for i in range(len(keys))
        ],
    }

    # save output artifacts
    emb_path = target_dir / "species_embeddings.pt"
    sim_path = target_dir / "species_similarity_matrix.pt"
    meta_path = target_dir / "species_metadata.json"
    sim_csv_path = target_dir / "species_similarity_matrix.csv"

    torch.save(embeddings_tensor, emb_path)
    torch.save(similarity_tensor, sim_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # save similarity matrix as readable CSV for inspection
    sim_df = pandas.DataFrame(
        similarity_np, index=pandas.Index(keys), columns=pandas.Index(keys)
    )
    sim_df.to_csv(sim_csv_path)

    print("\n--- Artifacts Successfully Generated ---")
    print(
        f"Embeddings Tensor [N={len(keys)}, D={embeddings_tensor.shape[1]}]: "
        f"{emb_path.resolve()}"
    )
    print(
        f"Similarity Matrix Tensor [N={len(keys)}, N={len(keys)}]: "
        f"{sim_path.resolve()}"
    )
    print(f"Readable Similarity CSV: {sim_csv_path.resolve()}")
    print(f"Class Metadata JSON: {meta_path.resolve()}")


def main() -> None:
    """CLI entrypoint for building species embedding space."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Sentence Transformer embeddings from species CSV "
            "knowledge base."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="knowledge/ontario_tree_species_grouped_profiles.csv",
        help="Path to species/grouped profiles CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="knowledge/embeddings",
        help="Output directory to save generated embeddings and matrix",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help=(
            "Sentence Transformer model name from HuggingFace "
            "(e.g. BAAI/bge-base-en-v1.5)"
        ),
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization of embedding vectors",
    )

    args = parser.parse_args()
    generate_embeddings_and_matrix(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
