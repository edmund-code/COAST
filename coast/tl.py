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
                   key_added: str = "label", min_score: float = 0.5, n_top_genes: int | None = 2000) -> pd.DataFrame:
    """Give every cluster the name of the marker panel with the highest mean score in it.

    ``panels`` maps any structure name to its marker genes. Each panel is scored per unit with
    `scanpy.tl.score_genes` on ``X``, its control genes drawn from the ``n_top_genes`` Seurat-flavour variable genes
    plus the panel genes (None: all genes), so thousands of barely detected genes do not set the background. A
    cluster whose best mean score is below ``min_score`` is labelled ``"Unassigned"``. Writes ``obs[key_added]`` and
    returns the cluster x panel table of mean scores with ``label`` and ``margin`` (best minus second-best score;
    a small margin means two panels fit almost equally well), so every label can be checked by eye.
    """
    pool = None
    if n_top_genes:
        hv = sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.n_vars), flavor="seurat",
                                         inplace=False)["highly_variable"].reindex(adata.var_names).to_numpy(bool)
        pool = list(adata.var_names[hv | adata.var_names.isin([g for gs in panels.values() for g in gs])])
    scores = pd.DataFrame(index=adata.obs_names)
    for name, genes in panels.items():
        present = [g for g in genes if g in adata.var_names]
        if not present:
            raise ValueError(f"no gene of panel {name!r} is in the data")
        sc.tl.score_genes(adata, present, score_name="_coast_score", gene_pool=pool, random_state=0)
        scores[name] = adata.obs.pop("_coast_score")
    means = scores.groupby(adata.obs[cluster_key].to_numpy()).mean()
    top2 = np.sort(means.to_numpy(), axis=1)[:, ::-1]
    labels = means.idxmax(axis=1).where(means.max(axis=1) >= min_score, "Unassigned")
    adata.obs[key_added] = pd.Categorical(adata.obs[cluster_key].map(labels))
    return means.assign(label=labels, margin=top2[:, 0] - top2[:, 1])


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


def _cr_basis(x: np.ndarray, knots: np.ndarray):
    """Cubic regression spline (mgcv ``bs='cr'``): the coefficients are the curve's values at the knots.
    Returns the basis at ``x`` and the penalty matrix of the integrated squared second derivative."""
    k, h = len(knots), np.diff(knots)
    D, B = np.zeros((k - 2, k)), np.zeros((k - 2, k - 2))
    for i in range(k - 2):
        D[i, i:i + 3] = 1 / h[i], -1 / h[i] - 1 / h[i + 1], 1 / h[i + 1]
        B[i, i] = (h[i] + h[i + 1]) / 3
        if i < k - 3:
            B[i, i + 1] = B[i + 1, i] = h[i + 1] / 6
    F = np.vstack([np.zeros(k), np.linalg.solve(B, D), np.zeros(k)])   # second derivatives at the knots
    j = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, k - 2)
    lo, hi = knots[j + 1] - x, x - knots[j]
    X = ((lo ** 3 / h[j] - h[j] * lo) / 6)[:, None] * F[j] + ((hi ** 3 / h[j] - h[j] * hi) / 6)[:, None] * F[j + 1]
    X[np.arange(len(x)), j] += lo / h[j]
    X[np.arange(len(x)), j + 1] += hi / h[j]
    return X, D.T @ np.linalg.solve(B, D)


def _nb_gam(y: np.ndarray, blocks: list, S: np.ndarray):
    """Negative-binomial GAM with one penalised cubic regression spline per sample, all sharing one smoothing
    parameter and one dispersion (tradeSeq's model). ``blocks`` holds (rows, basis, offset) per sample. The smoothing
    parameter and theta maximise the Laplace-approximate restricted likelihood (REML); coefficients by penalised IRLS.
    Returns samples x knots coefficients (log rate at the knots), or None when the fit fails."""
    from scipy.optimize import minimize
    from scipy.special import gammaln

    rank = S.shape[0] - 2
    beta = [np.full(S.shape[0], np.log((y[r].sum() + 0.5) / np.exp(o).sum())) for r, _, o in blocks]
    lam0 = np.mean([np.trace(X.T @ X) for _, X, _ in blocks]) / np.trace(S)

    def criterion(par):
        lam, theta = lam0 * np.exp(par[0]), np.exp(par[1])
        value = -0.5 * len(blocks) * rank * np.log(lam)
        for b, (r, X, o) in enumerate(blocks):
            yb, bb, step = y[r], beta[b], np.inf
            for it in range(101):   # penalised IRLS (a tiny ridge keeps a sample without counts solvable)
                eta = np.clip(X @ bb + o, -50, 50)
                mu = np.exp(eta)
                w = mu / (1 + mu / theta)
                H = X.T @ (w[:, None] * X) + lam * S + 1e-8 * np.eye(len(S))
                if it == 100 or np.max(np.abs(step)) < 1e-8:
                    break
                step = np.linalg.solve(H, X.T @ (w * (eta - o + (yb - mu) / mu))) - bb
                bb = bb + step
            ll = (gammaln(yb + theta) - gammaln(theta) - gammaln(yb + 1) + theta * np.log(theta / (theta + mu))
                  + yb * (eta - np.log(theta + mu))).sum()
            value += -ll + 0.5 * lam * bb @ S @ bb + 0.5 * np.linalg.slogdet(H)[1]
            beta[b] = bb
        return value if np.isfinite(value) else 1e300

    try:
        with np.errstate(all="ignore"):
            res = minimize(criterion, [0.0, 0.0], method="Nelder-Mead", bounds=[(-25, 25), (-8, 12)],
                           options=dict(xatol=1e-3, fatol=1e-6, maxiter=400))
            criterion(res.x)   # coefficients at the optimum
    except np.linalg.LinAlgError:
        return None
    out = np.array(beta)
    return out if np.isfinite(out).all() else None


def fit_curves(adata: AnnData, genes, x: str = "pseudostructure", sample_key: str | None = None, layer: str = "counts",
               library: str | None = None, k: int = 10, n_points: int = 101, limits: tuple[float, float] | None = None,
               n_jobs: int = 1) -> pd.DataFrame:
    """Fitted expression of each gene along ``obs[x]``, in log2 counts per 10k, one curve per sample.

    Negative-binomial GAM on raw counts with a log(library) offset (tradeSeq's model): one penalised cubic regression
    spline per ``sample_key`` level (one in all without it), ``k`` knots at quantiles of ``obs[x]`` shared by all
    samples, one smoothing parameter and one dispersion per gene chosen by REML. The penalty keeps genes that switch
    off along the axis finite. Evaluated on ``n_points`` evenly spaced positions between ``limits`` (default: the
    5th and 95th percentiles of ``obs[x]``). Positions outside a sample's range of ``obs[x]`` and genes whose fit
    fails are NaN. Returns genes x positions, or (sample, gene) x positions with ``sample_key``.
    """
    from joblib import Parallel, delayed

    genes, y, offset = _counts_and_offset(adata, genes, layer, library)
    position = adata.obs[x].to_numpy(float)
    knots = np.quantile(position, np.linspace(0, 1, k))
    if len(np.unique(knots)) < k:
        raise ValueError("tied knots: too few distinct positions for k knots")
    limits = np.percentile(position, [5, 95]) if limits is None else limits
    grid = np.linspace(*limits, n_points)
    samples = [None] if sample_key is None else list(adata.obs[sample_key].astype("category").cat.categories)
    rows = [np.arange(adata.n_obs) if s is None else np.flatnonzero(adata.obs[sample_key].to_numpy() == s)
            for s in samples]
    basis, S = _cr_basis(position, knots)
    blocks = [(r, basis[r], offset[r]) for r in rows]
    on_grid = _cr_basis(grid, knots)[0]
    outside = np.array([(grid < position[r].min()) | (grid > position[r].max()) for r in rows])
    fits = Parallel(n_jobs=n_jobs)(delayed(_nb_gam)(y[:, j], blocks, S) for j in range(y.shape[1]))
    curves = np.full((len(samples), len(genes), n_points), np.nan)
    for j, beta in enumerate(fits):
        if beta is not None:
            curves[:, j] = np.where(outside, np.nan, beta @ on_grid.T)
    curves = (curves + np.log(1e4)) / np.log(2)
    columns = pd.Index(np.round(grid, 4), name=x)
    if sample_key is None:
        return pd.DataFrame(curves[0], index=pd.Index(genes, name="gene"), columns=columns)
    index = pd.MultiIndex.from_product([samples, genes], names=[sample_key, "gene"])
    return pd.DataFrame(curves.reshape(-1, n_points), index=index, columns=columns)


def compare_groups(adata: AnnData, group_key: str, case: str, reference: str, sample_key: str = "sample",
                   x: str = "pseudostructure", genes=None, layer: str = "counts", library: str | None = None, k: int = 10,
                   n_points: int = 201, prior: float = 0.3, min_null_ratio: float = 2.0, min_reproduction: float = 0.5,
                   floor_quantile: float = 0.9, window: float = 0.02, min_window_coverage: float = 0.6,
                   crossover_share: float = 0.25, max_splits: int = 20, seed: int = 0, n_jobs: int = 1):
    """Which genes differ between two groups of samples (e.g. disease vs healthy) along ``obs[x]``.

    1. Axis: the stretch both groups cover (the larger 1st and the smaller 99th percentile of the two groups'
       positions) on ``n_points`` points; averages along it are weighted by the overlap of the two groups' unit
       densities, f_ref f_case / (f_ref + f_case). The units on that stretch are fitted with `fit_curves`, one curve
       per sample (NaN outside the sample's own 1st-99th percentile).
    2. Gap = log2((case rate + c) / (reference rate + c)) with c = ``prior`` x the gene's mean rate, from the group
       means of the sample curves (a sample's missing ends filled from its group); the case curve is re-referenced to
       the typical gene (median log2 fold change over genes removed at every position). The same gap is computed for
       every case x reference sample pair and for mixed splits (each side holds half of both groups, so the group
       difference cancels: the null; at most ``max_splits``).
    3. A gene is selected when, on the whole axis or in its strongest window of half-width ``window``, the gap's
       root mean square is >= the ``floor_quantile`` of the mixed-split sizes of genes of the same expression decile,
       >= ``min_null_ratio`` x the gene's own largest mixed split, and every pair reproduces >= ``min_reproduction``
       of it (pair . gap / gap . gap). Direction: the sign carrying most of the squared gap; ``crossover`` when each
       sign carries >= ``crossover_share``. Needs at least two samples per group.

    Returns (table, gap): one row per gene (rms, level, null_rms, null_ratio, min_pair_reproduction, direction,
    selected, selected_whole_axis, selected_local_window, local_x, expression_per_10k) and the gap curves
    (genes x positions).
    """
    import itertools

    from scipy.ndimage import gaussian_filter1d

    adata = adata[adata.obs[group_key].isin([case, reference])]
    pos, is_case = adata.obs[x].to_numpy(float), (adata.obs[group_key] == case).to_numpy()
    lo = max(np.percentile(pos[is_case], 1), np.percentile(pos[~is_case], 1))
    hi = min(np.percentile(pos[is_case], 99), np.percentile(pos[~is_case], 99))
    grid = np.linspace(lo, hi, n_points)
    step = grid[1] - grid[0]
    f = [gaussian_filter1d(np.histogram(pos[m], np.r_[grid - step / 2, hi + step / 2])[0] / m.sum(),
                           2 * (hi - lo) / 100 / step, mode="nearest") for m in (~is_case, is_case)]
    w = f[0] * f[1] / np.maximum(f[0] + f[1], 1e-300)
    w /= w.sum()
    wmean = lambda a: (np.where(np.isfinite(a), a, 0) * np.where(np.isfinite(a), w, 0)).sum(-1) / np.maximum(  # noqa: E731
        np.where(np.isfinite(a), w, 0).sum(-1), 1e-300)

    inside = (pos >= lo) & (pos <= hi)
    curves = fit_curves(adata[inside], genes, x=x, sample_key=sample_key, layer=layer, library=library, k=k,
                        n_points=n_points, limits=(lo, hi), n_jobs=n_jobs)
    genes = curves.index.get_level_values("gene").unique()
    by_sample = adata.obs.groupby(sample_key, observed=True)[group_key].first()
    cs, rs = sorted(by_sample.index[by_sample == case]), sorted(by_sample.index[by_sample == reference])
    if min(len(cs), len(rs)) < 2:
        raise ValueError("compare_groups needs at least two samples per group (no mixed split otherwise)")
    sp = {}
    for s in cs + rs:
        p = pos[adata.obs[sample_key].to_numpy() == s]
        out = (grid < np.percentile(p, 1)) | (grid > np.percentile(p, 99))
        sp[s] = np.where(out, np.nan, curves.loc[s].loc[genes].to_numpy() - np.log2(1e4))   # log2 rate per count

    filled = {}   # a sample's missing ends: the other samples of its group, shifted to meet it
    for group in (cs, rs):
        for s in group:
            total, count = np.zeros_like(sp[s]), np.zeros_like(sp[s])
            for o in set(group) - {s}:
                both = np.isfinite(sp[s]) & np.isfinite(sp[o])
                first = np.where(both.any(1), both.argmax(1), 0)
                last = np.where(both.any(1), n_points - 1 - both[:, ::-1].argmax(1), n_points - 1)
                d = np.nan_to_num(sp[s] - sp[o])
                i, rows = np.arange(n_points)[None], np.arange(len(genes))
                fill = sp[o] + np.where(i < first[:, None], d[rows, first][:, None],
                                        np.where(i > last[:, None], d[rows, last][:, None], 0))
                total, count = total + np.nan_to_num(fill), count + np.isfinite(sp[o])
            filled[s] = np.where(np.isfinite(sp[s]) | (count == 0), sp[s], total / np.maximum(count, 1))
    typical = lambda low, high: (low, high - np.nanmedian(high - low, axis=0))  # noqa: E731
    mean = lambda names: np.mean([filled[s] for s in names], axis=0)  # noqa: E731
    pooled = typical(mean(rs), mean(cs))
    rate = lambda v: 1e4 * 2.0 ** v  # noqa: E731
    expr = wmean((rate(pooled[0]) + rate(pooled[1])) / 2)
    gap = lambda lh: np.log2((rate(lh[1]) + prior * expr[:, None]) / (rate(lh[0]) + prior * expr[:, None]))  # noqa: E731
    D = gap(pooled)
    P = [gap(typical(sp[r], sp[c])) for c in cs for r in rs]
    splits = [(hc, hr) for nc in sorted({len(cs) // 2, (len(cs) + 1) // 2})
              for hc in itertools.combinations(cs, nc) if cs[0] in hc
              for nr in sorted({len(rs) // 2, (len(rs) + 1) // 2}) for hr in itertools.combinations(rs, nr)]
    if len(splits) > max_splits:
        splits = [splits[i] for i in np.sort(np.random.default_rng(seed).choice(len(splits), max_splits, False))]
    side = lambda c_, r_: (mean(c_) + mean(r_)) / 2  # noqa: E731
    N = [gap(typical(side([s for s in cs if s not in hc], [s for s in rs if s not in hr]), side(hc, hr)))
         for hc, hr in splits]

    decile = pd.qcut(pd.Series(expr).rank(method="first"), 10, labels=False).to_numpy()
    floor = lambda stat: np.array([np.quantile(stat[decile == d], floor_quantile) for d in range(10)])[decile]  # noqa: E731
    near = (np.abs(grid[:, None] - grid[None, :]) <= window) * w[None, :]   # window weights around every point
    ends = (grid - window >= grid[0] - 1e-9) & (grid + window <= grid[-1] + 1e-9)

    def strongest(Y):   # root mean square of Y in its strongest window, and that window's centre
        ok = np.isfinite(Y)
        den = ok @ near.T
        v = np.where((den / near.sum(1) >= min_window_coverage) & ends, (np.where(ok, Y, 0) ** 2 @ near.T)
                     / np.maximum(den, 1e-300), -np.inf)
        return np.sqrt(np.maximum(v.max(1), 0)), v.argmax(1)

    M, at = strongest(D)
    Wd = (np.abs(grid[None, :] - grid[at][:, None]) <= window) * w[None, :]
    wm = lambda Y: (np.nan_to_num(Y) * Wd * np.isfinite(Y)).sum(1) / np.maximum((Wd * np.isfinite(Y)).sum(1), 1e-300)  # noqa: E731
    M_null = np.max([strongest(n)[0] for n in N], 0)
    local = ((M >= floor(M_null)) & (M >= min_null_ratio * M_null)
             & (np.min([wm(p * D) for p in P], 0) / np.maximum(wm(D ** 2), 1e-24) >= min_reproduction))
    rms = np.sqrt(wmean(D ** 2))
    null = np.max([np.sqrt(wmean(n ** 2)) for n in N], 0)
    reproduction = np.min([wmean(p * D) for p in P], 0) / np.maximum(rms ** 2, 1e-24)
    whole = (rms >= floor(null)) & (rms >= min_null_ratio * null) & (reproduction >= min_reproduction)
    up = wmean(np.where(D > 0, D, 0) ** 2) / np.maximum(rms ** 2, 1e-24)
    direction = np.where((up >= crossover_share) & (up <= 1 - crossover_share), "crossover", np.where(up > 0.5, "up", "down"))
    direction = np.where(local & ~whole, np.where(wm(D) > 0, "up", "down"), direction)
    table = pd.DataFrame({"rms": rms, "level": wmean(D), "null_rms": null, "null_ratio": rms / np.maximum(null, 1e-12),
                          "min_pair_reproduction": reproduction, "direction": direction, "selected": whole | local,
                          "selected_whole_axis": whole, "selected_local_window": local, "local_x": grid[at],
                          "expression_per_10k": expr}, index=pd.Index(genes, name="gene"))
    return table, pd.DataFrame(D, index=table.index, columns=pd.Index(np.round(grid, 4), name=x))
