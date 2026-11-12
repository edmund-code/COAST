"""Plots of the curve and of genes along the pseudostructure coordinate."""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from .tl import fit_curves


def curve(adata: AnnData, color: str = "pseudostructure", basis: str | None = None, dims: tuple[int, int] = (0, 1),
          ax=None):
    """Units in two dimensions of an embedding, with the curve drawn through them.

    By default the embedding is the one the curve was fitted in (``uns['coast']['use_rep']``), whose first two
    dimensions can fold the structure over itself. With ``basis`` (for example ``"X_umap"``), units are drawn in that
    embedding instead, and each curve node at the median position of the units nearest to it in the fitted
    embedding; nodes no unit is nearest to are skipped. This places the curve for display only; the fit is unchanged.

    Categorical colours are scanpy's (``uns[color + '_colors']``) when set; units are drawn in random order so no
    category hides another.
    """
    from scipy.spatial import cKDTree

    ax = ax or plt.figure(figsize=(4.5, 4)).gca()
    fitted, nodes = adata.obsm[adata.uns["coast"]["use_rep"]], adata.uns["coast"]["nodes"]
    if basis is None:
        X, nodes = fitted[:, list(dims)], nodes[:, list(dims)]
        xlabel, ylabel = f"dim {dims[0] + 1}", f"dim {dims[1] + 1}"
    else:
        X = adata.obsm[basis][:, list(dims)]
        nearest = cKDTree(nodes).query(fitted)[1]
        nodes = np.array([np.median(X[nearest == k], axis=0) for k in range(len(nodes)) if (nearest == k).any()])
        xlabel, ylabel = (f"{basis.removeprefix('X_')} {d + 1}" for d in dims)
    values = adata.obs[color]
    if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object:
        values = values.astype("category")
        levels = values.cat.categories
        palette = adata.uns.get(f"{color}_colors")
        if palette is None or len(palette) != len(levels):
            palette = [plt.cm.tab20(i % 20) for i in range(len(levels))]
        order = np.random.default_rng(0).permutation(len(values))
        codes = values.cat.codes.to_numpy()[order]
        ax.scatter(*X[order].T, s=3, c=[palette[c] for c in codes], rasterized=True)
        for level, colour in zip(levels, palette):
            ax.scatter([], [], s=3, color=colour, label=str(level))
        ax.legend(markerscale=4, frameon=False, fontsize=8)
    else:
        points = ax.scatter(*X.T, c=values, s=3, cmap="viridis", rasterized=True)
        plt.colorbar(points, ax=ax, label=color)
    ax.plot(*nodes.T, "-o", color="black", ms=2, lw=1)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    return ax


def genes(adata: AnnData, genes: Sequence[str], x: str = "pseudostructure", sample_key: str | None = None,
          layer: str = "counts", library: str | None = None, n_bins: int = 20, df: int = 6, axes=None):
    """One panel per gene: equal-count bin means (points) and the fitted curve (line), per sample.

    Curves and points cover the 5th to 95th percentile of ``obs[x]`` over all units, the same range for every
    sample (see `coast.tl.fit_curves`).
    """
    genes = list(genes)
    limits = np.percentile(adata.obs[x], [5, 95])
    if axes is None:
        _, axes = plt.subplots(1, len(genes), figsize=(3 * len(genes), 2.6), squeeze=False)
        axes = axes[0]
    if sample_key is None:
        samples, palette = [None], [None]
    else:
        samples = list(adata.obs[sample_key].astype("category").cat.categories)
        palette = adata.uns.get(f"{sample_key}_colors")
        if palette is None or len(palette) != len(samples):
            palette = [None] * len(samples)
    for sample, colour in zip(samples, palette):
        sub = adata if sample is None else adata[adata.obs[sample_key] == sample]
        counts = sub[:, genes].layers[layer] if layer else sub[:, genes].X
        counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
        lib = sub.obs[library].to_numpy(float) if library else np.asarray(
            (sub.layers[layer] if layer else sub.X).sum(axis=1)).ravel()
        position = sub.obs[x].to_numpy()
        inside = np.flatnonzero((position >= limits[0]) & (position <= limits[1]))
        bins = np.array_split(inside[np.argsort(position[inside])], n_bins)
        bin_x = [position[b].mean() for b in bins]
        curves = fit_curves(sub, genes, x=x, layer=layer, library=library, df=df, limits=limits)
        for j, (gene, ax) in enumerate(zip(genes, axes)):
            with np.errstate(divide="ignore"):
                bin_y = [np.log2(1e4 * counts[b, j].sum() / lib[b].sum()) for b in bins]
            line, = ax.plot(curves.columns, curves.loc[gene], label=sample, color=colour)
            ax.scatter(bin_x, bin_y, s=8, color=line.get_color())
            ax.set(title=gene, xlabel=x, ylabel="log2 counts per 10k")
    if sample_key is not None:
        axes[-1].legend(frameon=False, fontsize=8)
    return axes


def heatmap(curves: pd.DataFrame, ax=None):
    """Fitted curves (genes x positions, from `coast.tl.fit_curves`) z-scored per gene, ordered by peak."""
    curves = curves.dropna(how="all")   # genes without a fit
    z = curves.sub(curves.mean(axis=1), axis=0).div(curves.std(axis=1), axis=0)
    z = z.iloc[np.argsort(np.nanargmax(z.to_numpy(), axis=1), kind="stable")]
    ax = ax or plt.figure(figsize=(4, min(8, max(2, 0.12 * len(z))))).gca()
    image = ax.imshow(z.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                      extent=(z.columns[0], z.columns[-1], len(z), 0), interpolation="nearest")
    if len(z) <= 60:
        ax.set_yticks(np.arange(len(z)) + 0.5, z.index, fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_xlabel(curves.columns.name or "position")
    plt.colorbar(image, ax=ax, label="z")
    return ax
