# COAST

**C**ontinuous **O**rdering of **A**natomical **S**tructures from **T**ranscriptomics.

Many tissues are built from one structure repeated many times: nephrons in the kidney, crypts in the gut,
lobules in the liver. A tissue section cuts these structures into many separate cross-sections. COAST orders those
cross-sections along one continuous axis, the **pseudostructure** coordinate, using expression alone. It then tests
which genes change along that axis, and which change beyond the steps between discrete clusters.

COAST starts from a unit × gene count matrix: one row per segmented cross-section (a *unit*), with raw counts and
a sample label. Segmentation, and assigning spots or bins to units, happen upstream.

## Install

```bash
pip install git+https://github.com/edmundtsou/COAST.git
```

Python ≥ 3.10. To run the tutorial, install the extras with `pip install "coast[tutorial] @ git+https://github.com/edmundtsou/COAST.git"`.

## Quick start

```python
import anndata as ad
import scanpy as sc
import coast

adata = ad.read_h5ad("units.h5ad")                 # raw counts, obs["sample"]

# 1. filter and normalise
adata = coast.pp.filter_units(adata)
coast.pp.normalize(adata)

# 2. integrate, label clusters, keep the structure, integrate again
coast.pp.integrate(adata, batch_key="sample")
sc.tl.leiden(adata)
coast.tl.label_clusters(adata, panels={"S1": [...], "S2": [...], "S3": [...], "other": [...]})
adata = adata[adata.obs["label"] != "other"].copy()
coast.pp.integrate(adata, batch_key="sample")

# 3. the pseudostructure coordinate
coast.tl.pseudostructure(adata, early_genes=[...], late_genes=[...])

# 4. genes along it (one sample at a time; keep genes that replicate)
result = coast.tl.test_genes(adata[adata.obs["sample"] == "A"])
curves = coast.tl.fit_curves(adata, result.index[result["significant"]])
coast.pl.heatmap(curves)
```

See [`tutorials/getting_started.ipynb`](tutorials/getting_started.ipynb) for a full run on healthy human and mouse
kidney (Visium HD), including the restriction to 1:1 orthologs for comparing species. The input matrices will be
released with the paper.

## What each step does

| Function | What it does |
|---|---|
| `pp.filter_units` | Drops units below the hard floors (counts; optionally bin coverage, for units clipped by the capture edge). Then drops units whose log counts, log counts per bin (optional) or genes-vs-counts residual is more than 3 robust SDs from the median of their group (sample, or sample x any class column). |
| `pp.doublets` | Optional. Flags units holding two structures: two incompatible identities in the whole unit (1) or in two places of it (2, a permutation test on its bins; 2a when the identities are adjacent), large and not convex (3), or two pieces joined by a neck (4). |
| `pp.filter_genes` | Optional. Keeps genes detected in >= 5 % of units with >= 20 counts, plus listed markers. |
| `pp.normalize` | Keeps raw counts in `layers["counts"]` and sets `X` to log1p(counts per 10k). |
| `pp.integrate` | Variable genes (ranked within each sample; with `group_key`, only genes variable in every group) → PCA → Harmony → neighbours → UMAP. |
| `tl.label_clusters` | Gives each cluster the name of its best-scoring marker panel (any panels; control genes from the variable genes) and returns the score table with each cluster's margin over the runner-up. |
| `tl.pseudostructure` | Fits an elastic principal curve ([ElPiGraph](https://github.com/j-bac/elpigraph-python)) through the integrated embedding and projects units onto it. Units beyond the curve's ends go on the extension of the end edge. Marker genes set the direction. The coordinate is scaled to [0, 1]. |
| `tl.test_genes` | Fits a negative-binomial GLM per gene (raw counts, log-library offset, cubic spline in the coordinate) and compares it with an intercept-only model, or with one mean per cluster when `groupby` is given. Genes are called against a decoy with the coordinate shuffled. |
| `tl.fit_curves` | Fitted expression of each gene along the coordinate (log2 counts per 10k): a negative-binomial GAM, one penalised spline per sample, smoothing and dispersion by REML (tradeSeq's model). |
| `tl.compare_groups` | Which genes differ between two groups of samples along the coordinate: gap curves, replicated by every sample pair and larger than mixed-sample splits. |
| `pl.curve`, `pl.genes`, `pl.heatmap` | The curve in the fitted embedding or on the UMAP (`basis="X_umap"`), genes along the coordinate per sample, and curves ordered by peak. |
