# final-method: before / after

`pl.genes` and `pl.heatmap` on 14,659 mouse proximal-tubule units, drawn with `fix-fit-curves` (left) and `final-method`
(right). `fit_curves` is now a negative-binomial GAM with one penalised spline per sample (shared smoothing and
dispersion, REML), so sparse genes get finite curves in every sample; `pl.genes` scales each gene to 0-1 by default
(`scale="log"` gives counts per 10k on a log axis, case 5).
