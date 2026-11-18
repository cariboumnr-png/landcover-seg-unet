# Scripts exécutables du projet

[English](README.md) | [Français](README_fr.md)

Ce répertoire contient les scripts de points d'entrée exécutables pour la génération de plongements vectoriels (*embeddings*), la création de données de test synthétiques et l'exécution de pipelines sur Databricks ou machines virtuelles (VM).

---

## 📜 Index des scripts et exemples d'utilisation

### 1. `build_species_embeddings.py`
**Description :**
Encode les profils écologiques des espèces d'arbres à partir de bases de connaissances CSV à l'aide de Sentence Transformers (par défaut : `BAAI/bge-base-en-v1.5`). Génère des tenseurs d'embeddings PyTorch normalisés L2 (`[N, D]`), des matrices de similarité cosinus (`[N, N]`), des fichiers JSON de métadonnées de classes et des fichiers CSV de similarité lisibles par l'humain dans des sous-dossiers nommés d'après le CSV source sous `knowledge/embeddings/`.

**Exemples d'utilisation :**

```powershell
# 1. Générer les embeddings pour les 29 groupes écologiques (par défaut)
.\.venv\Scripts\python.exe scripts/build_species_embeddings.py

# 2. Générer les embeddings pour la liste complète des 138 espèces de l'IFI (FRI)
.\.venv\Scripts\python.exe scripts/build_species_embeddings.py `
    --csv-path knowledge/ontario_tree_species_profiles.csv

# 3. Spécifier un modèle HuggingFace personnalisé ou un répertoire de sortie
.\.venv\Scripts\python.exe scripts/build_species_embeddings.py `
    --csv-path knowledge/ontario_tree_species_grouped_profiles.csv `
    --output-dir knowledge/embeddings `
    --model-name BAAI/bge-base-en-v1.5
```

---

### 2. `generate_dummy_data.py`
**Description :**
Génère des jeux de données raster géospatiaux synthétiques (fichiers GeoTIFF représentant l'imagerie multispectrale, le MNA/DEM, le MNS/DSM et les masques de segmentation) dans `./experiment/input` pour tester localement les pipelines d'entraînement UNet sans avoir besoin de rasters satellites complets.

**Exemple d'utilisation :**

```powershell
.\.venv\Scripts\python.exe scripts/generate_dummy_data.py
```

---

### 3. `run.py`
**Description :**
Script de lancement (*runner/bootstrap*) conçu pour les clusters de travaux (*jobs*) Databricks, les machines virtuelles dans le cloud et l'exécution de pipelines conteneurisés. Résout dynamiquement le répertoire `src/` de l'espace de travail et invoque l'interface en ligne de commande (CLI) de `landseg`.

**Exemple d'utilisation :**

```powershell
# Exécuter la CLI du pipeline via le script de lancement
.\.venv\Scripts\python.exe scripts/run.py --help
```

---

### 4. `build_raster_mosaic.py`
**Description :**
Harmonise et assemble plusieurs dalles raster disjointes ou adjacentes (telles que les unités de gestion forestière de la Couronne de l'Ontario à travers différentes zones UTM) en une seule mosaïque virtuelle VRT (`.vrt`) alignée sur le système de coordonnées de référence provincial standard (`EPSG:3161`), sans matérialiser d'imagerie intermédiaire sur le disque.

**Exemples d'utilisation :**

```powershell
# 1. Mosaïquer des dalles GeoTIFF adjacentes en un raster virtuel unique EPSG:3161
.\.venv\Scripts\python.exe scripts/build_raster_mosaic.py `
    data/fmu_tiles/*.tif `
    --output experiment/input/feature/mosaic_sentinel2.vrt

# 2. Mosaïquer des masques de classification catégorielle avec rééchantillonnage au plus proche voisin
.\.venv\Scripts\python.exe scripts/build_raster_mosaic.py `
    data/fmu_labels/*.tif `
    --output experiment/input/label/mosaic_labels.vrt `
    --categorical `
    --relative-paths

# 3. Spécifier une résolution cible (ex. 20m) et un algorithme de rééchantillonnage personnalisé
.\.venv\Scripts\python.exe scripts/build_raster_mosaic.py `
    data/dem_tiles/*.tif `
    --output experiment/input/feature/mosaic_dem.vrt `
    --target-resolution 20.0 `
    --resampling cubic
```
