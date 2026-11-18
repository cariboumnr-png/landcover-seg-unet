# Base de connaissances écologiques des espèces d'arbres de l'Ontario

[English](README.md) | [Français](README_fr.md)

Ce répertoire contient la base de connaissances écologiques de domaine et l'espace précalculé de plongements vectoriels (*embeddings*) pour les espèces d'arbres en Ontario. Cette base de connaissances est conçue pour injecter des connaissances écologiques (préférences d'habitat, caractéristiques sylvicoles, niches topographiques et exigences en sol/humidité) dans les modèles d'apprentissage profond pour la segmentation de données de télédétection et d'entrées topographiques.

---

## 🌿 Principes de conception et méthodologie

La base de connaissances a été construite selon quatre principes directeurs :

### 1. Alignement 1:1 avec les normes FRI / IFI de l'Ontario
Tous les codes d'espèces correspondent directement aux codes officiels de l'**Inventaire des ressources forestières (FRI / IFI)** du ministère des Richesses naturelles et des Forêts de l'Ontario (MRNFO) (attributs `SPCOMP` / `LEAD_SP` dans les fichiers SIG shapefiles forestiers provinciaux).

### 2. Profilage multidimensionnel des niches écologiques
Chaque espèce ou groupe fait l'objet d'un profil selon sept dimensions écologiques clés :
* **Taxonomie et groupe fonctionnel :** Conifère vs feuillu, forme biologique.
* **Zone climatique :** Boréale, Grands Lacs-Saint-Laurent, carolinienne ou introduite/montagnarde.
* **Position topographique et niche altitudinale :** Vallées fluviales, dépressions de tourbières, plaines sablonneuses ou crêtes rocheuses.
* **Régime hydrique :** Hydrique (humide), mésique (modéré) à xérique (sec).
* **Substrat pédologique :** Tourbe acide, loams bien drainés, affleurements calcaires ou sables d'épandage grossiers.
* **Tolérance à l'ombre :** Très intolérant à extrêmement tolérant.
* **Stade de succession :** Pionnier post-incendie/perturbation, mi-successionnel à climax de fin de succession.

### 3. Double granularité (Inventaire vs similarité écologique)
Pour équilibrer les besoins de l'inventaire forestier avec la séparabilité spectrale et topographique en télédétection, la base de connaissances propose deux représentations CSV :

* **Profils complets des espèces (`ontario_tree_species_profiles.csv`) :**
  * Contient l'ensemble des **138 codes d'espèces documentés dans l'IFI (FRI)**.
  * Idéal pour la cartographie d'inventaire exhaustive et la classification fine *zero-shot*.

* **Profils écologiquement regroupés (`ontario_tree_species_grouped_profiles.csv`) :**
  * Agrège les 138 espèces en **29 groupes écologiques**.
  * Conserve une haute granularité pour les résineux d'importance commerciale (`SB`, `SW`, `PJ`, `PW`, `PR`, `BF`, `CE`, `LA`, `HE`, `CR`).
  * Regroupe les feuillus secondaires par genre ou guilde écologique (ex. `OAK_GROUP`, `ASH_GROUP`, `WILLOW_GROUP`, `CAROLINIAN_MINOR_HARDWOODS`) lorsque la séparabilité spectrale est faible ou que l'intérêt pour l'inventaire est secondaire.

### 4. Ingénierie des invites (*Prompt Engineering*) pour Sentence Transformers
Chaque profil intègre une chaîne descriptive `formatted_description` spécifiquement formulée pour les modèles de recherche dense (ex. `BAAI/bge-base-en-v1.5` ou `all-MiniLM-L6-v2`), permettant à la similarité cosinus dans l'espace vectoriel textuel de refléter fidèlement la proximité d'habitat écologique réelle.

---

## 📂 Structure du répertoire

```text
knowledge/
├── README.md                                         <-- Documentation en anglais
├── README_fr.md                                      <-- Documentation en français
├── ontario_tree_species_grouped_profiles.csv         <-- 29 profils d'espèces regroupés écologiquement
├── ontario_tree_species_profiles.csv                 <-- 138 profils complets d'espèces FRI
└── embeddings/                                       <-- Vecteurs précalculés Sentence Transformers
    ├── species_embeddings.pt                         <-- Tenseur PyTorch [N, D] d'embeddings de classes
    ├── species_similarity_matrix.pt                  <-- Tenseur PyTorch [N, N] matrice de similarité cosinus
    ├── species_similarity_matrix.csv                 <-- Table de similarité N x N lisible par l'humain
    └── species_metadata.json                         <-- Métadonnées reliant l'indice de classe au code FRI et au prompt
```

---

## ⚡ Génération et mise à jour des embeddings

Chaque fois qu'un fichier CSV de profils est mis à jour ou modifié, régénérez les tenseurs de plongements vectoriels PyTorch en exécutant :

```powershell
# Générer les embeddings pour les 29 groupes écologiques (par défaut)
.\.venv\Scripts\python.exe scripts/build_species_embeddings.py `
    --csv-path knowledge/ontario_tree_species_grouped_profiles.csv `
    --output-dir knowledge/embeddings `
    --model-name BAAI/bge-base-en-v1.5

# Générer les embeddings pour la liste complète des 138 espèces
.\.venv\Scripts\python.exe scripts/build_species_embeddings.py `
    --csv-path knowledge/ontario_tree_species_profiles.csv `
    --output-dir knowledge/embeddings_full `
    --model-name BAAI/bge-base-en-v1.5
```
