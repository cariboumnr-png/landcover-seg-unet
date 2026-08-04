# Ontario Tree Species Ecological Knowledge Base

This directory contains the curated ecological domain knowledge base and precomputed vector embedding space for tree species in Ontario. The knowledge base is designed to inject ecological domain knowledge (habitat preferences, silvicultural traits, topographic niches, and soil/moisture requirements) into remote sensing and topographic deep learning segmentation models.

---

## 🌿 Curation Principles & Methodology

The knowledge base was constructed according to four core principles:

### 1. 1:1 Alignment with Ontario FRI Standards
All species codes correspond directly to official **Ontario Ministry of Natural Resources and Forestry (OMNRF) Forest Resources Inventory (FRI)** standard codes (`SPCOMP` / `LEAD_SP` attributes in provincial forest GIS shapefiles).

### 2. Multi-Dimensional Ecological Niche Profiling
Each species or group is profiled across seven key ecological dimensions:
* **Taxonomy & Functional Group:** Conifer vs. Broadleaf Hardwood, growth form.
* **Climate Zone:** Boreal, Great Lakes-St. Lawrence, Carolinian, or introduced/montane.
* **Topographic Position & Elevation Niche:** River valleys, peatland depressions, sandy plains, or rocky ridges.
* **Moisture Regime:** Hydric (wet), Mesic (moderate), to Xeric (dry).
* **Soil Substrate:** Acidic peat, well-drained loams, limestone outcrops, or coarse outwash sands.
* **Shade Tolerance:** Very Intolerant to Extremely Tolerant.
* **Successional Stage:** Fire/disturbance pioneer, mid-successional, to late-successional climax.

### 3. Dual Granularity (Inventory vs. Ecological Similarity)
To balance inventory interest with remote sensing spectral/topographic separability, the knowledge base provides two CSV representations:

* **Full Species Profiles (`ontario_tree_species_profiles.csv`):**
  * Contains all **138 documented FRI species codes**.
  * Ideal for full-spectrum inventory mapping and zero-shot fine-grained classification.

* **Ecologically Grouped Profiles (`ontario_tree_species_grouped_profiles.csv`):**
  * Aggregates the 138 species into **29 ecological groups**.
  * Maintains high granularity for commercially vital softwoods (`SB`, `SW`, `PJ`, `PW`, `PR`, `BF`, `CE`, `LA`, `HE`, `CR`).
  * Groups minor hardwoods by genus or ecological guild (e.g., `OAK_GROUP`, `ASH_GROUP`, `WILLOW_GROUP`, `CAROLINIAN_MINOR_HARDWOODS`) where spectral separability is low or inventory interest is secondary.

### 4. Sentence Transformer Prompt Engineering
Each profile includes a `formatted_description` text string specifically structured for dense retrieval models (e.g., `BAAI/bge-base-en-v1.5` or `all-MiniLM-L6-v2`), enabling cosine similarity in text vector space to mirror true ecological habitat proximity.

---

## 📂 Directory Layout

```
knowledge/
├── README.md                                         <-- This documentation
├── ontario_tree_species_grouped_profiles.csv         <-- 29 Ecologically grouped species profiles
├── ontario_tree_species_profiles.csv                 <-- 138 Full FRI species profiles
└── embeddings/                                       <-- Precomputed Sentence Transformer vectors
    ├── species_embeddings.pt                         <-- PyTorch Tensor [N, D] class embeddings
    ├── species_similarity_matrix.pt                  <-- PyTorch Tensor [N, N] cosine similarity matrix
    ├── species_similarity_matrix.csv                 <-- Human-readable N x N similarity table
    └── species_metadata.json                         <-- Class index to FRI code & prompt metadata
```

---

## ⚡ Generating & Updating Embeddings

Whenever a profile CSV is updated or modified, regenerate the PyTorch embedding tensors by running:

```powershell
# Generate embeddings for 29 ecological groups (default)
.\.venv\Scripts\python.exe scripts/build_species_embeddings.py `
    --csv-path knowledge/ontario_tree_species_grouped_profiles.csv `
    --output-dir knowledge/embeddings `
    --model-name BAAI/bge-base-en-v1.5

# Generate embeddings for full 138 species list
.\.venv\Scripts\python.exe scripts/build_species_embeddings.py `
    --csv-path knowledge/ontario_tree_species_profiles.csv `
    --output-dir knowledge/embeddings_full `
    --model-name BAAI/bge-base-en-v1.5
```
