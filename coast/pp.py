"""Preprocessing: filter units, normalise, and integrate samples into one embedding."""
from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData

# Mitochondrial and ribosomal genes (mouse and human symbols). They are kept out of the integration features.
DEFAULT_EXCLUDE = r"^(?:mt-|MT-|Rp[ls]\d|RP[LS]\d|Mrp[ls]\d|MRP[LS]\d)"


def filter_units(adata: AnnData, min_counts: float = 500, n_mads: float = 3.0, groupby: str | None = "sample") -> AnnData:
    """Remove units with too few counts and units whose depth or complexity is an outlier in their group.

    A unit is kept when it has at least ``min_counts`` counts and both log total counts and log genes detected
    lie within ``n_mads`` robust SDs (1.4826 x MAD) of the median of its ``groupby`` group.

    Returns a filtered copy; ``adata`` is not modified.
    """
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    detected = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    floor = totals >= min_counts
    keep = floor.copy()
    groups = adata.obs[groupby].to_numpy() if groupby else np.zeros(adata.n_obs)
    for values in (np.log1p(totals), np.log1p(detected)):
        for g in np.unique(groups):
            in_group = groups == g
            median = np.median(values[in_group & floor])
            sd = 1.4826 * np.median(np.abs(values[in_group & floor] - median))
            keep &= ~(in_group & (np.abs(values - median) > n_mads * sd))
    return adata[keep].copy()


def normalize(adata: AnnData, target_sum: float = 1e4) -> None:
    """Keep raw counts in ``layers['counts']`` and set ``X`` to log1p(counts per ``target_sum``)."""
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def integrate(adata: AnnData, batch_key: str = "sample", group_key: str | None = None, n_pcs: int = 50,
              theta: float = 6, exclude: str | None = DEFAULT_EXCLUDE, n_neighbors: int = 25,
              random_state: int = 0) -> None:
    """Integration features -> PCA -> Harmony -> neighbour graph -> UMAP.

    Features are Seurat-flavour highly variable genes. With ``group_key`` (for example condition or species)
    they are selected inside each group and only genes variable in every group are kept, so the correction
    never sees the genes that differ between groups, which are the genes tested later. Genes matching the
    regular expression ``exclude`` are never features.

    Writes ``var['coast_feature']``, ``obsm['X_pca']``, ``obsm['X_harmony']``, the neighbour graph and
    ``obsm['X_umap']``. ``X`` must hold log-normalised values (see `normalize`).

    The paper used R harmony 2.0.5 (theta 6, lambda 1, 30 iterations). This uses harmonypy, the Python port,
    whose numerics differ slightly; to match the paper exactly, run R harmony on ``obsm['X_pca']`` and store
    the result in ``obsm['X_harmony']``.
    """
    import harmonypy

    sc.pp.highly_variable_genes(adata, flavor="seurat", min_mean=0.0125, max_mean=3, min_disp=0.5,
                                batch_key=group_key)
    if group_key is None:
        features = adata.var["highly_variable"].to_numpy()
    else:
        features = (adata.var["highly_variable_nbatches"] == adata.obs[group_key].nunique()).to_numpy()
    if exclude:
        features &= ~adata.var_names.str.contains(exclude)
    adata.var["coast_feature"] = features

    sc.pp.pca(adata, n_comps=n_pcs, mask_var="coast_feature", random_state=random_state)
    harmony = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, batch_key, theta=theta, lamb=1,
                                    max_iter_harmony=30, random_state=random_state, verbose=False)
    adata.obsm["X_harmony"] = np.asarray(harmony.Z_corr, dtype=float)
    # exact neighbours: deterministic, and scanpy 1.11 fails on the sparse arrays pynndescent>=0.7 returns
    sc.pp.neighbors(adata, use_rep="X_harmony", n_neighbors=n_neighbors, transformer="sklearn",
                    random_state=random_state)
    sc.tl.umap(adata, min_dist=0.3, random_state=random_state)
