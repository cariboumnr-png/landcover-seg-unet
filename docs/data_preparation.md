## 🧊 Data Foundation

The system operates on **Landsat imagery** and **DEM‑derived terrain metrics**.
The dataprep pipeline:

- generates spectral indices (NDVI, NDMI, NBR)
- produces slope, aspect, TPI from DEM
- builds label hierarchies
- normalizes features globally using Welford statistics
- bundles everything into stable `.npz` blocks