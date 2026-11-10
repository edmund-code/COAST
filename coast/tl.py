"""Tools: cluster labels, the pseudostructure coordinate, and gene shapes along it."""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from scipy.stats import spearmanr


def label_clusters(adata: AnnData, panels: Mapping[str, Sequence[str]], cluster_key: str = "leiden",
                   key_added: str = "label", min_score: float = 0.5) -> pd.DataFrame:
    """Give every cluster the name of the marker panel with the highest mean score in it.

    Each panel is scored per unit with `scanpy.tl.score_genes` on ``X``. A cluster whose best mean score is
    below ``min_score`` is labelled ``"Unassigned"``. Writes ``obs[key_added]`` and returns the cluster x panel
    table of mean scores, so every label can be checked by eye.
    """
    scores = pd.DataFrame(index=adata.obs_names)
    for name, genes in panels.items():
        present = [g for g in genes if g in adata.var_names]
        if not present:
            raise ValueError(f"no gene of panel {name!r} is in the data")
        sc.tl.score_genes(adata, present, score_name="_coast_score", random_state=0)
        scores[name] = adata.obs.pop("_coast_score")
    means = scores.groupby(adata.obs[cluster_key].to_numpy()).mean()
    labels = means.idxmax(axis=1).where(means.max(axis=1) >= min_score, "Unassigned")
    adata.obs[key_added] = pd.Categorical(adata.obs[cluster_key].map(labels))
    return means


def _marker_score(adata: AnnData, genes: Sequence[str] | None) -> np.ndarray:
    if not genes:
        return np.zeros(adata.n_obs)
    present = [g for g in genes if g in adata.var_names]
    if not present:
        raise ValueError(f"none of {list(genes)} is in the data")
    sc.tl.score_genes(adata, present, score_name="_coast_score", random_state=0)
    return adata.obs.pop("_coast_score").to_numpy(float)


def pseudostructure(adata: AnnData, use_rep: str = "X_harmony", early_genes: Sequence[str] | None = None,
                    late_genes: Sequence[str] | None = None, n_nodes: int = 50, lam: float = 0.01, mu: float = 0.1,
                    end_stretch: float = 2.0, seed: int = 0, key_added: str = "pseudostructure") -> None:
    """Order units along one elastic principal curve through the integrated embedding.

    1. Fit an elastic principal curve (ElPiGraph; ``n_nodes`` nodes, stretching ``lam``, bending ``mu``)
       through ``obsm[use_rep]`` and project every unit onto it.
    2. The projection clips every unit beyond an end node to that node, which piles units up at the two ends.
       Such units are instead placed on the straight continuation of the end edge, at most ``end_stretch``
       end-edge lengths out (the princurve / Slingshot convention). No other unit moves.
    3. Orient the coordinate so it rises with (late - early) marker score, and scale it to [0, 1].

    Writes ``obs[key_added]`` and ``uns['coast']`` (curve nodes in order and their coordinate). Without marker
    genes the direction is arbitrary.
    """
    import elpigraph
    import networkx as nx

    X = np.asarray(adata.obsm[use_rep], dtype=float)
    np.random.seed(seed)  # elpigraph draws the start vector of its first principal component from numpy's global state
    graph = elpigraph.computeElasticPrincipalCurve(X, n_nodes, Lambda=lam, Mu=mu, Do_PCA=False)[0]
    chain = nx.Graph(graph["Edges"][0].tolist())
    ends = [node for node, degree in chain.degree if degree == 1]
    path = nx.shortest_path(chain, ends[0], ends[1])
    elpigraph.utils.getPseudotime(X, graph, path[0])
    t = np.asarray(graph["pseudotime"], dtype=float).copy()
    nodes = graph["NodePositions"][path]
    node_t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))])

    tolerance = 1e-9 * node_t[-1]
    for tip, neighbour, direction in ((0, 1, -1.0), (-1, -2, 1.0)):
        edge = nodes[tip] - nodes[neighbour]
        length = np.linalg.norm(edge)
        along = (X - nodes[tip]) @ (edge / length)
        past = (np.abs(t - node_t[tip]) <= tolerance) & (along > 0)
        t[past] = node_t[tip] + direction * np.minimum(along[past], end_stretch * length)

    axis = _marker_score(adata, late_genes) - _marker_score(adata, early_genes)
    if (early_genes or late_genes) and spearmanr(t, axis).statistic < 0:
        t, node_t, nodes = -t, -node_t[::-1], nodes[::-1]
    low, high = t.min(), t.max()
    adata.obs[key_added] = (t - low) / (high - low)
    adata.uns["coast"] = {"use_rep": use_rep, "nodes": nodes, "node_pseudostructure": (node_t - low) / (high - low)}


# ---------------------------------------------------------------------------------------------------
# gene shapes along the coordinate
# ---------------------------------------------------------------------------------------------------
def _counts_and_offset(adata: AnnData, genes, layer: str, library: str | None):
    counts = adata.layers[layer] if layer else adata.X
    lib = adata.obs[library].to_numpy(float) if library else np.asarray(counts.sum(axis=1)).ravel()
    genes = list(adata.var_names) if genes is None else list(genes)
    y = counts[:, adata.var_names.get_indexer(genes)]
    y = y.toarray() if sparse.issparse(y) else np.asarray(y)
    return genes, y, np.log(lib)


def _spline(x: np.ndarray, df: int):
    """Intercept + cubic B-spline with ``df`` columns (knots at quantiles of x)."""
    import patsy

    return patsy.dmatrix(f"bs(x, df={df}, degree=3)", {"x": x})


def _nb_fit(y, design, offset):
    """ML negative-binomial GLM (NB2). Returns (coefficients, dispersion alpha, converged)."""
    import statsmodels.api as sm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.NegativeBinomial(y, design, offset=offset).fit(disp=0, maxiter=200)
    alpha = fit.params[-1]
    converged = bool(fit.mle_retvals["converged"]) and np.isfinite(alpha)
    return fit.params[:-1], max(alpha, 1e-8) if converged else 1e-8, converged   # floor: a Poisson-like gene


def _nb_lr(y, null, full, offset):
    """Likelihood ratio of ``full`` against ``null`` (the dispersion estimated under ``full``), the dispersion,
    convergence, and the log2 range (5th to 95th percentile over units) of the terms ``full`` adds to ``null``."""
    import statsmodels.api as sm

    try:
        _, alpha, converged = _nb_fit(y, full, offset)
        family = sm.families.NegativeBinomial(alpha=alpha)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(y, full, family=family, offset=offset).fit()
            lr = sm.GLM(y, null, family=family, offset=offset).fit().deviance - fit.deviance
    except (ValueError, np.linalg.LinAlgError):   # a gene with almost no counts: report it as not converged
        return np.nan, np.nan, False, np.nan
    added = full[:, null.shape[1]:] @ fit.params[null.shape[1]:]
    return lr, alpha, converged, np.subtract(*np.percentile(added, [95, 5])) / np.log(2)


def _lr_all(y, null, full, offset, n_jobs):
    from joblib import Parallel, delayed

    rows = Parallel(n_jobs=n_jobs)(delayed(_nb_lr)(y[:, j], null, full, offset) for j in range(y.shape[1]))
    return np.array(rows, dtype=float).T


def decoy_threshold(real: np.ndarray, decoy: np.ndarray, fdr: float = 0.05) -> float:
    """Smallest statistic c with #(decoy >= c) / #(real >= c) <= ``fdr``; inf when none qualifies."""
    real, decoy = np.sort(real[np.isfinite(real)]), np.sort(decoy[np.isfinite(decoy)])
    for c in real:
        if (len(decoy) - np.searchsorted(decoy, c)) / (len(real) - np.searchsorted(real, c)) <= fdr:
            return float(c)
    return np.inf


def test_genes(adata: AnnData, x: str = "pseudostructure", groupby: str | None = None, genes=None,
               layer: str = "counts", library: str | None = None, df: int = 6, fdr: float = 0.05, seed: int = 0,
               n_jobs: int = 1) -> pd.DataFrame:
    """Which genes change along the coordinate (or, with ``groupby``, change beyond the steps between clusters).

    Every gene gets a negative-binomial GLM on raw counts with a log(library) offset:
      null  intercept, or one level per ``groupby`` cluster (for example S1/S2/S3)
      full  null + a cubic B-spline in ``obs[x]`` with ``df`` degrees of freedom
    The dispersion is estimated under ``full`` and held fixed for ``null``; the statistic is the likelihood
    ratio ``lr``.

    Units from one structure are not independent, so ``p_chi2`` is reported only for reference. Genes are
    called against a decoy instead: ``obs[x]`` is shuffled (within each ``groupby`` cluster), every fit is
    repeated, and ``significant`` marks genes above the smallest ``lr`` at which decoy calls / real calls
    <= ``fdr``. Run each sample separately and ask that calls replicate across samples.

    With thousands of units, tiny shifts are significant. ``log2_range`` gives the size of the effect: the range
    (5th to 95th percentile over units) of the spline part of the fit, in log2 units.

    ``library`` names an obs column of library sizes (default: total counts of ``layer``).
    Returns one row per gene; the threshold is in ``result.attrs['threshold']``.
    """
    from scipy.stats import chi2

    genes, y, offset = _counts_and_offset(adata, genes, layer, library)
    position = adata.obs[x].to_numpy(float)
    if groupby is None:
        null, strata = np.ones((adata.n_obs, 1)), np.zeros(adata.n_obs)
    else:
        strata = adata.obs[groupby].to_numpy()
        null = pd.get_dummies(strata).to_numpy(float)
    rng = np.random.default_rng(seed)
    decoy_position = position.copy()
    for level in np.unique(strata):
        idx = np.flatnonzero(strata == level)
        decoy_position[idx] = position[rng.permutation(idx)]

    lr, alpha, converged, log2_range = _lr_all(y, null, np.column_stack([null, _spline(position, df)[:, 1:]]), offset, n_jobs)
    decoy_lr, _, decoy_converged, _ = _lr_all(y, null, np.column_stack([null, _spline(decoy_position, df)[:, 1:]]),
                                           offset, n_jobs)
    ok = converged.astype(bool)
    threshold = decoy_threshold(lr[ok], decoy_lr[ok & decoy_converged.astype(bool)], fdr)
    result = pd.DataFrame({"lr": lr, "p_chi2": chi2.sf(lr, df), "decoy_lr": decoy_lr, "log2_range": log2_range,
                           "alpha": alpha,
                           "converged": ok, "significant": ok & (lr >= threshold)}, index=pd.Index(genes, name="gene"))
    result.attrs["threshold"] = threshold
    return result


def fit_curves(adata: AnnData, genes, x: str = "pseudostructure", layer: str = "counts", library: str | None = None,
               df: int = 6, n_points: int = 101, limits: tuple[float, float] | None = None) -> pd.DataFrame:
    """Fitted expression of each gene along ``obs[x]``, in log2 counts per 10k.

    Negative-binomial GLM with a cubic B-spline (``df`` degrees of freedom) and a log(library) offset, fitted on
    every unit and evaluated on ``n_points`` evenly spaced positions between ``limits``. By default the limits are
    the 5th and 95th percentiles of ``obs[x]``. A spline is poorly constrained near the first and last units, so
    the outer 5% at each end still anchor the fit but are not shown. Returns genes x positions.
    """
    import patsy

    genes, y, offset = _counts_and_offset(adata, genes, layer, library)
    position = adata.obs[x].to_numpy(float)
    design = _spline(position, df)
    limits = np.percentile(position, [5, 95]) if limits is None else limits
    grid = np.linspace(*limits, n_points)
    on_grid = np.asarray(patsy.build_design_matrices([design.design_info], {"x": grid})[0])
    curves = [on_grid @ _nb_fit(y[:, j], np.asarray(design), offset)[0] for j in range(y.shape[1])]
    return pd.DataFrame((np.array(curves) + np.log(1e4)) / np.log(2), index=pd.Index(genes, name="gene"),
                        columns=pd.Index(np.round(grid, 4), name=x))
