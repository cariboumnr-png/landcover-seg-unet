# Vegetation-Focused Importance Ranking for Coarse Ecosite Groups

This ranking is **a priori (ecological + remote sensing–driven)** and is **not based on pixel counts**.
It reflects how useful each group is for **vegetation discrimination with Landsat + DEM**.

---

## 🥇 Tier 1 — Core vegetation signal (highest priority)

These systems provide the **strongest, most separable, and most informative canopy signal**.

**8 — fresh_loam**

* Tolerant hardwood dominance (e.g., sugar maple)
* Strong phenology and canopy closure
* Low edaphic noise
  → *Benchmark deciduous forest class*

**9 — moist_fine**

* Clear red maple / mixed hardwood gradients
* Moisture-driven productivity differences
* Strong reflectance contrast

**6 — moist_coarse**

* Conifer ↔ mixedwood transitions
* Moisture visible in DEM + spectral response
* High structural variability

**5 — dryfresh_coarse**

* Classic mixedwood / pine systems
* SWIR dryness signal
* Good conifer vs hardwood separability

---

## 🥈 Tier 2 — Very important (slightly harder)

**4 — dry_sand**

* Pine / oak systems
* Distinct when pure
* Open canopy introduces soil background noise

**10 — organic_wet** *(priority depends on wetland interest)*

* Treed swamp, thicket swamp, fen vegetation
* Unique phenology
* Very strong moisture + texture signal
* Scientifically important but more complex

---

## 🥉 Tier 3 — Secondary vegetation environments

**3 — vshallow**

* Thin soils → stressed / sparse vegetation
* Rock / lichen background contamination
* Lower productivity signal

**7 — fresh_clay**

* Ecologically important
* Spectrally similar to fresh_loam
* Separation relies more on terrain than canopy

---

## 🪨 Tier 4 — Context / constraint classes (low vegetation value)

These help **model spatial logic** but are not vegetation targets.

**11 — rock_barren**
Minimal canopy signal → coarse head only.

**2 — bluff_dune**
Unstable geomorphic edge systems → sparse vegetation.

**13 — coastal** *(if present in study area)*
Vegetation exists but spectrally dominated by substrate / salinity.

**12 — anthro**
No ecological vegetation value (use as mask/context).

**1 — water**
Purely contextual.

---

# 🎯 Final Priority Order (vegetation-focused)

## Highest value — allocate most model capacity

* fresh_loam (8)
* moist_fine (9)
* moist_coarse (6)
* dryfresh_coarse (5)
* dry_sand (4)

## High value (optional emphasis depending on goals)

* organic_wet (10)

## Secondary vegetation systems

* vshallow (3)
* fresh_clay (7)

## Coarse-head / context only

* rock_barren (11)
* bluff_dune (2)
* coastal (13)
* anthro (12)
* water (1)

---

# 🧠 How to use this in the model

### Model capacity

* Larger / deeper fine heads → Tier 1
* Medium → Tier 2–3
* Coarse head only → Tier 4

### Loss weighting

Upweight:

* fresh_loam
* moist_fine
* moist_coarse

### Evaluation focus

Report detailed metrics for Tier 1 — these carry the main ecological signal and scientific value.
