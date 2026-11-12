"""Preprocessing: filter units, normalise, and integrate samples into one embedding."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

# Mitochondrial and ribosomal genes (mouse and human symbols). They are kept out of the integration features.
DEFAULT_EXCLUDE = r"^(?:mt-|MT-|Rp[ls]\d|RP[LS]\d|Mrp[ls]\d|MRP[LS]\d)"


def filter_units(adata: AnnData, min_counts: float = 500, min_coverage: float | None = None,
                 coverage_key: str = "bin_coverage", n_mads: float | None = 3.0,
                 groupby: str | Sequence[str] | None = "sample", bins_key: str | None = None,
                 min_group_size: int = 50) -> AnnData:
    """Remove units that fail the hard floors and units whose depth or complexity is an outlier in their group.

    Hard floors: at least ``min_counts`` counts and, with ``min_coverage``, ``obs[coverage_key]`` >= ``min_coverage``
    (the share of the unit covered by bins, e.g. bins x bin area / polygon area; lower = clipped by the capture edge).
    Outliers (skipped when ``n_mads`` is None), among the units that pass the floors: any of log total counts,
    log counts per bin (only with ``bins_key``, an obs column of bins per unit) and the residual of
    log genes detected ~ log total counts (a line fitted per group) more than ``n_mads`` robust SDs
    (1.4826 x MAD) from its group median, either side. Groups are the combinations of the ``groupby`` columns; a
    combination with fewer than ``min_group_size`` units is pooled with the other small ones of its first column.

    Returns a filtered copy; ``adata`` is not modified.
    """
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    keep = totals >= min_counts
    if min_coverage is not None:
        keep &= adata.obs[coverage_key].to_numpy(float) >= min_coverage
    if n_mads is None:
        return adata[keep].copy()
    cols = [groupby] if isinstance(groupby, str) else list(groupby or [])
    key = adata.obs[cols].astype(str).agg("|".join, axis=1) if cols else pd.Series("all", index=adata.obs_names)
    small = key.map(key[keep].value_counts()).fillna(0).to_numpy() < min_group_size
    key[small] = adata.obs.loc[small, cols[0]].astype(str) + "|all" if cols else "all"
    with np.errstate(divide="ignore"):
        log_t = np.log(totals)
        log_g = np.log(np.asarray((adata.X > 0).sum(axis=1)).ravel())
        metrics = [log_t] + ([log_t - np.log(adata.obs[bins_key].to_numpy(float))] if bins_key else [])
    residual = np.full(adata.n_obs, np.nan)
    for g in key[keep].unique():
        m = keep & (key == g).to_numpy()
        residual[m] = log_g[m] - np.polyval(np.polyfit(log_t[m], log_g[m], 1), log_t[m])
    out = np.zeros(adata.n_obs, bool)
    for values in metrics + [residual]:
        for g in key[keep].unique():
            m = keep & (key == g).to_numpy()
            median = np.median(values[m])
            out |= m & (np.abs(values - median) > n_mads * 1.4826 * np.median(np.abs(values[m] - median)))
    return adata[keep & ~out].copy()


def filter_genes(adata: AnnData, min_fraction: float = 0.05, min_counts: float = 20, keep: Sequence[str] = ()) -> AnnData:
    """Keep genes detected in at least ``min_fraction`` of the units with at least ``min_counts`` counts, plus the
    genes in ``keep`` (e.g. marker panels). Counts come from ``layers['counts']`` when present; the returned copy has
    raw counts in ``X``, so call `normalize` again (the library then sums the kept genes only)."""
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    detected = np.asarray((counts > 0).sum(axis=0)).ravel()
    mask = ((detected >= max(1, int(np.ceil(min_fraction * adata.n_obs))))
            & (np.asarray(counts.sum(axis=0)).ravel() >= min_counts)) | adata.var_names.isin(list(keep))
    out = adata[:, mask].copy()
    out.X = out.layers.pop("counts") if "counts" in out.layers else out.X
    out.uns.pop("log1p", None)
    return out


def normalize(adata: AnnData, target_sum: float = 1e4) -> None:
    """Keep raw counts in ``layers['counts']`` and set ``X`` to log1p(counts per ``target_sum``)."""
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def integrate(adata: AnnData, batch_key: str = "sample", group_key: str | None = None, n_pcs: int = 50,
              theta: float = 6, exclude: str | None = DEFAULT_EXCLUDE, n_neighbors: int = 25,
              random_state: int = 0) -> None:
    """Integration features -> PCA -> Harmony -> neighbour graph -> UMAP.

    Features are Seurat-flavour highly variable genes, ranked within each ``batch_key`` level and combined. With
    ``group_key`` (for example condition or species) they are selected inside each group and only genes variable
    in every group are kept, so the correction never sees the genes that differ between groups, which are the
    genes tested later. Genes matching the regular expression ``exclude`` are never features.

    Writes ``var['coast_feature']``, ``obsm['X_pca']``, ``obsm['X_harmony']``, the neighbour graph and
    ``obsm['X_umap']``. ``X`` must hold log-normalised values (see `normalize`).

    The paper used R harmony 2.0.5 (theta 6, lambda 1, 30 iterations). This uses harmonypy, the Python port,
    whose numerics differ slightly; to match the paper exactly, run R harmony on ``obsm['X_pca']`` and store
    the result in ``obsm['X_harmony']``.
    """
    import harmonypy

    eligible = ~adata.var_names.str.contains(exclude) if exclude else np.ones(adata.n_vars, bool)
    features = eligible.copy()
    for group in [None] if group_key is None else adata.obs[group_key].unique():
        sub = adata[:, eligible] if group is None else adata[adata.obs[group_key] == group, eligible]
        batch = batch_key if sub.obs[batch_key].nunique() > 1 else None
        hv = sc.pp.highly_variable_genes(sub, flavor="seurat", min_mean=0.0125, max_mean=3, min_disp=0.5,
                                         batch_key=batch, inplace=False)
        features[eligible] &= hv["highly_variable"].reindex(sub.var_names).to_numpy(bool)
    adata.var["coast_feature"] = features

    sc.pp.pca(adata, n_comps=n_pcs, mask_var="coast_feature", random_state=random_state)
    harmony = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, batch_key, theta=theta, lamb=1,
                                    max_iter_harmony=30, random_state=random_state, verbose=False)
    adata.obsm["X_harmony"] = np.asarray(harmony.Z_corr, dtype=float)
    # exact neighbours: deterministic, and scanpy 1.11 fails on the sparse arrays pynndescent>=0.7 returns
    sc.pp.neighbors(adata, use_rep="X_harmony", n_neighbors=n_neighbors, transformer="sklearn",
                    random_state=random_state)
    sc.tl.umap(adata, min_dist=0.3, random_state=random_state)


def doublets(adata: AnnData, bins: AnnData, polygons: Sequence, regions: Mapping[str, Sequence[str]],
             adjacency: Mapping[str, Sequence[str]], sample_key: str = "sample", class_key: str | None = None,
             exempt: Sequence[str] = (),
             discard: Sequence[str] = ("2", "3", "4"), min_detected: float = 0.10, confident: float = 0.5,
             split_min_counts: float = 1000, part_min_counts: float = 400, n_shuffles: int = 99, alpha: float = 0.05,
             n_mads: float = 3.0, max_solidity: float = 0.75, erosion: float = 0.5, min_piece: float = 0.2,
             seed: int = 0) -> None:
    """Flag units that hold two structures. Writes ``obs['doublet_tier<t>']`` per tier, ``obs['doublet_split_p']`` and
    ``obs['doublet']`` (any tier in ``discard``); nothing is removed. Run it on the units that pass the hard floors
    (`filter_units` with ``n_mads=None``): they are the reference of the identity scores and of the size outliers.
    Units of the ``class_key`` classes in ``exempt`` (e.g. glomeruli, which hold several cell types) are never flagged.

    ``regions`` maps a broad identity to marker genes and ``adjacency`` lists the identities that may meet inside
    one structure (e.g. consecutive nephron segments). A unit or a part of one is scored per identity as the mean z
    of log1p(counts per 10k) of the identity's markers detected in >= ``min_detected`` of the sample's units (z
    against those units); its identity is the best score, confident above ``confident``.

    1   the whole unit: two confident, non-adjacent identities
    2   two PLACES: ``bins`` (bins x genes with raw counts, ``obs['unit']`` = unit obs name, ``obs['x']``, ``obs['y']``,
        optional ``obs['total_counts']`` over all genes) of units with >= ``split_min_counts`` counts are cut six ways
        (2-means, across the long axis at 25 / 75 %, along it at 33 / 50 / 67 %); with both parts >= ``part_min_counts``,
        separation D = [s_A(a) - s_B(a)] - [s_A(b) - s_B(b)] (A, B the parts' identities); the best cut is tested
        against ``n_shuffles`` permutations of the bins (max |D| over the same cuts; stops once p <= ``alpha`` is out of
        reach). Flagged: p <= ``alpha``, both parts confident and different, non-adjacent (``2a``: adjacent).
    3   size (number of bins) above +``n_mads`` MAD within its ``class_key`` class and solidity < ``max_solidity``
    4   falls into >= 2 pieces, each >= ``min_piece`` of the remaining area, when eroded by ``erosion`` x its
        inscribed radius (two structures joined by a neck)

    ``polygons`` are shapely polygons in the order of ``obs`` and in the coordinates of the bins.
    """
    import shapely
    from scipy import sparse

    adjacent = lambda a, b: (a == b) | np.array([y in adjacency.get(x, ()) for x, y in zip(a, b)], bool)  # noqa: E731
    names, markers = np.array(list(regions)), sorted({g for gs in regions.values() for g in gs if g in adata.var_names})
    lognorm = lambda c, t: np.log1p(np.asarray(c, float) * (1e4 / np.maximum(t, 1e-12))[:, None])  # noqa: E731
    flags = pd.DataFrame(False, index=adata.obs_names, columns=["1", "2", "2a", "3", "4"])
    split_p = pd.Series(np.nan, index=adata.obs_names)
    total = np.asarray(adata.X.sum(axis=1)).ravel()
    classes = adata.obs[class_key].astype(str).to_numpy() if class_key else np.zeros(adata.n_obs, str)
    candidate, size = ~np.isin(classes, list(exempt)), np.zeros(adata.n_obs)
    for sample in adata.obs[sample_key].unique():
        rows = np.flatnonzero(adata.obs[sample_key].to_numpy() == sample)
        X = adata.X[rows][:, adata.var_names.get_indexer(markers)]
        expr = lognorm(X.toarray() if sparse.issparse(X) else X, total[rows])
        mean, sd = expr.mean(0), expr.std(0)
        ok = (expr > 0).mean(0) >= min_detected
        cols = [np.flatnonzero(np.isin(markers, list(regions[r])) & ok) for r in names]

        def scores(e):   # parts x identities
            z = np.where(sd > 0, (e - mean) / np.where(sd > 0, sd, 1), 0)
            return np.stack([z[:, c].mean(1) if len(c) else np.zeros(len(z)) for c in cols], 1)

        s = scores(expr)
        top = np.argsort(-s, 1)[:, :2]
        both = np.take_along_axis(s, top, 1).min(1) > confident
        flags.iloc[rows, 0] = both & ~adjacent(names[top[:, 0]], names[top[:, 1]])

        # tier 2: bins of the testable units, sorted by unit (stable), cut six ways
        unit = pd.Index(adata.obs_names[rows]).get_indexer(bins.obs["unit"].astype(str))
        size[rows] = unit_bins = np.bincount(unit[unit >= 0], minlength=len(rows))
        b = np.flatnonzero((unit >= 0) & np.isin(unit, np.flatnonzero((total[rows] >= split_min_counts)
                                                                      & (unit_bins >= 8) & candidate[rows])))
        b = b[np.argsort(unit[b], kind="stable")]
        pos = np.unique(unit[b], return_inverse=True)[1]
        xy = bins.obs[["x", "y"]].to_numpy(float)[b]
        Mb = bins.X[b][:, bins.var_names.get_indexer(markers)].tocsr().astype(np.float32)
        cb = (bins.obs["total_counts"].to_numpy(float) if "total_counts" in bins.obs
              else np.asarray(bins.X.sum(1)).ravel())[b]
        starts = np.flatnonzero(np.r_[True, np.diff(pos) != 0]) if len(b) else np.zeros(0, int)
        cuts = (np.concatenate([_cuts(c) for c in np.split(xy, starts[1:])]) if len(b) else np.zeros((0, 6), np.int8))
        n, r = len(starts), np.arange(len(starts))

        def parts(idx, p, m_units, M, c):   # per cut: counts and identity scores of parts a and b of each unit
            agg = lambda m: sparse.csr_matrix((np.ones(m.sum(), np.float32), (p[m], np.flatnonzero(m))),  # noqa: E731
                                              shape=(m_units, len(p)))
            whole, wt = (agg(np.ones(len(p), bool)) @ M).toarray(), np.bincount(p, c, m_units)
            out = []
            for k in range(cuts.shape[1]):
                m = cuts[idx, k] == 1
                pb, tb = (agg(m) @ M).toarray(), np.bincount(p[m], c[m], m_units)
                out.append((wt - tb, tb, scores(lognorm(whole - pb, wt - tb)), scores(lognorm(pb, tb))))
            return out

        obs_parts = parts(np.arange(len(b)), pos, n, Mb, cb)
        D = np.full((n, cuts.shape[1]), -np.inf)
        ia, ib = np.zeros(D.shape, int), np.zeros(D.shape, int)
        for k, (ca, cb_, sa, sb) in enumerate(obs_parts):
            ia[:, k], ib[:, k] = sa.argmax(1), sb.argmax(1)
            valid = np.minimum(ca, cb_) >= part_min_counts
            D[valid, k] = (sa[r, ia[:, k]] - sa[r, ib[:, k]] - sb[r, ia[:, k]] + sb[r, ib[:, k]])[valid]
        stat, rng = D.max(1), np.random.default_rng(seed)
        exceed, done, active = np.zeros(n), np.zeros(n), np.isfinite(stat)
        for shuffle in range(n_shuffles):   # null: the unit's bins permuted; units that cannot reach alpha drop out
            if shuffle in (0, 10, 20, 40, 70):
                keep = np.flatnonzero(active[pos])
                units, p = np.unique(pos[keep], return_inverse=True)
            if not len(units):
                break
            perm = np.empty(len(p), np.int64)
            perm[np.lexsort((rng.random(len(p)), p))] = np.arange(len(p))
            null, rr = np.full((len(units), cuts.shape[1]), -np.inf), np.arange(len(units))
            for k, (_, _, sa, sb) in enumerate(parts(keep, p, len(units), Mb[keep][perm], cb[keep][perm])):
                a_, b_, fin = ia[units, k], ib[units, k], np.isfinite(D[units, k])
                null[fin, k] = np.abs(sa[rr, a_] - sa[rr, b_] - sb[rr, a_] + sb[rr, b_])[fin]
            still = active[units]
            exceed[units[still]] += (null.max(1) >= stat[units])[still]
            done[units[still]] += 1
            active &= exceed < np.floor(alpha * (n_shuffles + 1))
        best = np.where(np.isfinite(stat), D.argmax(1), 0)
        A, B = ia[r, best], ib[r, best]
        sa = np.stack([o[2] for o in obs_parts], 2)[r, A, best]
        sb = np.stack([o[3] for o in obs_parts], 2)[r, B, best]
        hit = (np.isfinite(stat) & (sa > confident) & (sb > confident) & (A != B)
               & ((1 + exceed) / (1 + np.maximum(done, 1)) <= alpha))
        near = adjacent(names[A], names[B])
        tested = rows[np.unique(unit[b])] if len(b) else rows[:0]
        flags.iloc[tested, 1], flags.iloc[tested, 2] = hit & ~near, hit & near
        split_p.iloc[tested] = np.where(np.isfinite(stat), (1 + exceed) / (1 + np.maximum(done, 1)), np.nan)

    # tiers 3 and 4: geometry, pooled over samples
    polygons = np.asarray(polygons, dtype=object)
    solidity = shapely.area(polygons) / shapely.area(shapely.convex_hull(polygons))
    with np.errstate(divide="ignore"):
        log_size = np.log(size)
    for c in np.unique(classes[candidate]):
        m = candidate & (classes == c)
        median = np.median(log_size[m])
        upper = median + n_mads * 1.4826 * np.median(np.abs(log_size[m] - median))
        flags.loc[m, "3"] = (log_size[m] > upper) & (solidity[m] < max_solidity)
    radius = shapely.length(shapely.maximum_inscribed_circle(polygons))
    pieces, owner = shapely.get_parts(shapely.buffer(polygons, -erosion * radius), return_index=True)
    area = shapely.area(pieces)
    big = (area > 0) & (area >= min_piece * np.bincount(owner, area, len(polygons))[owner])
    flags["4"] = np.bincount(owner[big], minlength=len(polygons)) >= 2
    flags &= candidate[:, None]
    for t in flags:
        adata.obs[f"doublet_tier{t}"] = flags[t].to_numpy()
    adata.obs["doublet_split_p"] = split_p.to_numpy()
    adata.obs["doublet"] = flags[list(discard)].any(axis=1).to_numpy()


def _cuts(xy: np.ndarray) -> np.ndarray:
    """Six two-part cuts of one unit's bins (0 / 1 per bin): 2-means, across the long axis at 25 / 75 %, along it at
    33 / 50 / 67 %."""
    c = xy - xy.mean(0)
    vectors = np.linalg.eigh(c.T @ c)[1]
    along, across = c @ vectors[:, 1], c @ vectors[:, 0]
    label = along > 0          # 2-means started from the long axis (Lloyd iterations)
    for _ in range(25):
        if label.all() or not label.any():
            break
        new = ((xy - xy[label].mean(0)) ** 2).sum(1) < ((xy - xy[~label].mean(0)) ** 2).sum(1)
        if (new == label).all():
            break
        label = new
    return np.stack([label] + [along > np.quantile(along, q) for q in (0.25, 0.75)]
                    + [across > np.quantile(across, q) for q in (0.33, 0.5, 0.67)], 1).astype(np.int8)
