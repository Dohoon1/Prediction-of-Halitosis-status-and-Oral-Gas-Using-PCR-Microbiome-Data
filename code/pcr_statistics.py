"""PCR-only descriptive statistics and prespecified representation checks.

Independent total-bacteria measurements are never used. All normalizations use
the sum of the sixteen targeted assays and must not be read as whole-community
relative abundance. Importing this module performs no analysis or I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _json(path, obj):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [clean(v) for v in value]
        if isinstance(value, (np.floating, float)) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    Path(path).write_text(
        json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def _count(config, key, default):
    value = config.get(key, default)
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer.")
    return int(value)


def _adjust(p, method="bh"):
    p = np.asarray(p, dtype=float)
    valid = np.isfinite(p)
    values = np.where(valid, p, 1.0)
    if method == "bonferroni":
        adjusted = np.minimum(1, values * len(values))
    else:
        order = np.argsort(values, kind="stable")
        ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
        adjusted = np.empty(len(values))
        adjusted[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    adjusted[~valid] = np.nan
    return adjusted


def _data(dataset):
    frame, columns = dataset.frame.copy(), list(dataset.pcr_columns)
    if (
        len(columns) != 16
        or len(set(columns)) != 16
        or not all(c.startswith("PCR_") for c in columns)
    ):
        raise ValueError("Exactly sixteen distinct PCR_ assay columns are required.")
    if frame.participant_id.isna().any() or frame.participant_id.astype(str).duplicated().any():
        raise ValueError("Unique nonmissing participant IDs are required.")
    a = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("PCR concentrations must be finite/nonnegative; missing is not zero.")
    y = pd.to_numeric(frame.Halitosis, errors="raise").to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("The supplied binary labels must contain both 0 and 1.")
    return frame, columns, a, y.astype(int)


def panel_indices(a):
    """The validated standalone definitions: S, natural-log H, and H/ln(S)."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] != 16 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Expected a finite nonnegative participant-by-16 PCR matrix.")
    total = a.sum(axis=1)
    s = (a > 0).sum(axis=1)
    p = np.divide(a, total[:, None], out=np.zeros_like(a), where=total[:, None] > 0)
    terms = np.zeros_like(p)
    positive = p > 0
    terms[positive] = p[positive] * np.log(p[positive])
    h = -terms.sum(axis=1)
    h[total == 0] = np.nan
    j = np.full(len(a), np.nan)
    j[s > 1] = h[s > 1] / np.log(s[s > 1])
    return s, h, j


def representations(a, config):
    pseudo = float(config.get("sensitivity", {}).get("clr_pseudocount_copies_ml", 1.0))
    if not np.isfinite(pseudo) or pseudo <= 0:
        raise ValueError("CLR pseudocount must be a positive finite concentration in copies/mL.")
    total = a.sum(axis=1)
    relative = np.divide(a, total[:, None], out=np.full_like(a, np.nan), where=total[:, None] > 0)
    log_for_clr = np.log(a + pseudo)
    return {
        "raw": a.copy(),
        "log10": np.log10(a + 1),
        "relative": relative,
        "clr": log_for_clr - log_for_clr.mean(axis=1, keepdims=True),
    }


def _summary(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return dict(
        n=len(v),
        mean=float(v.mean()) if len(v) else np.nan,
        sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
        median=float(np.median(v)) if len(v) else np.nan,
        q25=float(np.percentile(v, 25)) if len(v) else np.nan,
        q75=float(np.percentile(v, 75)) if len(v) else np.nan,
    )


def _tests(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    welch = np.nan
    if min(len(a), len(b)) >= 2 and (np.var(a) > 0 or np.var(b) > 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            welch = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    mw = (
        float(stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").pvalue)
        if min(len(a), len(b))
        else np.nan
    )
    return welch, mw


def _cliff(a, b):
    """Positive delta means larger values in b (halitosis) than in a."""
    return float(np.sign(b[:, None] - a[None, :]).mean())


def _effect_interval(a, b, repeats, seed):
    if not min(len(a), len(b)):
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    differences, deltas = [], []
    for _ in range(repeats):
        ar, br = rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True)
        differences.append(br.mean() - ar.mean())
        deltas.append(_cliff(ar, br))
    if not differences:
        return np.nan, np.nan, np.nan, np.nan
    return (*np.percentile(differences, [2.5, 97.5]), *np.percentile(deltas, [2.5, 97.5]))


def _diversity(frame, a, y, out_dir):
    names = ["PCR_detected_targets", "PCR_panel_Shannon_ln", "PCR_panel_Pielou_evenness"]
    values = dict(zip(names, panel_indices(a)))
    participant = pd.DataFrame(
        {
            "participant_id": frame.participant_id,
            "Halitosis": y,
            "PCR_panel_total": a.sum(axis=1),
            **values,
        }
    )
    participant.to_csv(out_dir / "PCR_panel_diversity_participants.csv", index=False)
    rows = []
    for metric, v in values.items():
        row = {"metric": metric}
        groups = []
        for label, g in [("nonhalitosis", 0), ("halitosis", 1)]:
            group = v[(y == g) & np.isfinite(v)]
            groups.append(group)
            row.update({f"{label}_{key}": value for key, value in _summary(group).items()})
        row["welch_p_two_sided"], row["mannwhitney_p_two_sided"] = _tests(*groups)
        rows.append(row)
    summary = pd.DataFrame(rows)
    for test in ["welch", "mannwhitney"]:
        summary[f"{test}_bonferroni_3"] = _adjust(summary[f"{test}_p_two_sided"], "bonferroni")
    summary.to_csv(out_dir / "PCR_panel_diversity_summary.csv", index=False)
    return participant, summary


def _species(columns, spaces, y, config, out_dir):
    cfg = config.get("statistics", {})
    repeats, seed = _count(cfg, "effect_bootstrap_repeats", 1000), int(cfg.get("seed", 42))
    rows = []
    log_effects = {}
    for j, taxon in enumerate(columns):
        a, b = spaces["log10"][y == 0, j], spaces["log10"][y == 1, j]
        lo, hi, _, _ = _effect_interval(a, b, repeats, seed + j)
        log_effects[taxon] = (b.mean() - a.mean(), lo, hi)
    for space, matrix in spaces.items():
        for j, taxon in enumerate(columns):
            a = matrix[(y == 0) & np.isfinite(matrix[:, j]), j]
            b = matrix[(y == 1) & np.isfinite(matrix[:, j]), j]
            p_welch, p_mw = _tests(a, b)
            lo, hi, dlo, dhi = _effect_interval(
                a, b, repeats, seed + 100 * list(spaces).index(space) + j
            )
            ref, reflo, refhi = log_effects[taxon]
            rows.append(
                dict(
                    representation=space,
                    taxon=taxon,
                    n0=len(a),
                    n1=len(b),
                    mean0=float(a.mean()) if len(a) else np.nan,
                    mean1=float(b.mean()) if len(b) else np.nan,
                    mean_difference=float(b.mean() - a.mean()) if min(len(a), len(b)) else np.nan,
                    mean_difference_ci_low=lo,
                    mean_difference_ci_high=hi,
                    reference_mean_log10_copies_plus1_difference=ref,
                    reference_log10_difference_ci_low=reflo,
                    reference_log10_difference_ci_high=refhi,
                    cliffs_delta=_cliff(a, b) if min(len(a), len(b)) else np.nan,
                    cliffs_delta_ci_low=dlo,
                    cliffs_delta_ci_high=dhi,
                    p_welch=p_welch,
                    p_mannwhitney=p_mw,
                    bootstrap_repeats=repeats,
                    effect_direction="halitosis_minus_nonhalitosis",
                    multiplicity_family="16 taxa within representation and test",
                )
            )
    table = pd.DataFrame(rows)
    for test in ["welch", "mannwhitney"]:
        table[f"{test}_bonferroni_16"] = table.groupby("representation", sort=False)[
            f"p_{test}"
        ].transform(lambda p: _adjust(p, "bonferroni"))
        table[f"{test}_bh_16"] = table.groupby("representation", sort=False)[f"p_{test}"].transform(
            lambda p: _adjust(p, "bh")
        )
    table.to_csv(out_dir / "species_comparisons.csv", index=False)


def _correlation(matrix, method):
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    result = np.full((matrix.shape[1], matrix.shape[1]), np.nan)
    if len(matrix) < 3:
        return result, len(matrix)
    values = stats.rankdata(matrix, axis=0) if method == "spearman" else matrix
    keep = np.std(values, axis=0) > 0
    if keep.sum() > 1:
        result[np.ix_(keep, keep)] = np.corrcoef(values[:, keep], rowvar=False)
    elif keep.sum() == 1:
        result[np.ix_(keep, keep)] = 1
    return result, len(matrix)


def _memberships(corr, cutoff, strategy, absolute=False):
    n = len(corr)
    if strategy == "hierarchical_average_1_abs_r":
        distance = 1 - np.clip(np.abs(np.nan_to_num(corr, nan=0)), 0, 1)
        np.fill_diagonal(distance, 0)
        z = linkage(squareform((distance + distance.T) / 2, checks=False), method="average")
        labels = fcluster(z, t=1 - cutoff, criterion="distance")
    else:
        values = np.abs(corr) if absolute else corr
        adjacency = np.isfinite(values) & (values > cutoff)
        labels = np.zeros(n, dtype=int)
        component = 0
        for start in range(n):
            if labels[start]:
                continue
            component += 1
            pending = [start]
            while pending:
                node = pending.pop()
                if labels[node]:
                    continue
                labels[node] = component
                pending.extend(j for j in np.flatnonzero(adjacency[node]) if not labels[j])
    # Label ordering must not depend on scipy's arbitrary cluster labels.
    mapping = {value: i + 1 for i, value in enumerate(dict.fromkeys(labels))}
    return np.asarray([mapping[v] for v in labels])


def _partial(matrix, load, method):
    rows = []
    for i in range(matrix.shape[1]):
        for j in range(i + 1, matrix.shape[1]):
            keep = np.isfinite(matrix[:, i]) & np.isfinite(matrix[:, j]) & np.isfinite(load)
            values = np.c_[matrix[keep, i], matrix[keep, j], load[keep]]
            n = len(values)
            r, p = np.nan, np.nan
            if n > 3:
                if method == "spearman":
                    values = stats.rankdata(values, axis=0)
                x = np.c_[np.ones(n), values[:, 2]]
                residual = values[:, :2] - x @ np.linalg.lstsq(x, values[:, :2], rcond=None)[0]
                if np.linalg.matrix_rank(x) == 2 and (residual.std(axis=0) > 1e-12).all():
                    r = float(np.corrcoef(residual, rowvar=False)[0, 1])
                    t = r * np.sqrt((n - 3) / max(1 - r * r, np.finfo(float).eps))
                    p = float(2 * stats.t.sf(abs(t), n - 3))
            rows.append(dict(i=i, j=j, n=n, r=r, p=p))
    return rows


def _clusters(columns, spaces, a, config, out_dir):
    cfg = config.get("sensitivity", {})
    cutoffs = list(cfg.get("correlation_thresholds", [0.3, 0.4, 0.5, 0.6]))
    if not cutoffs or not all(np.isfinite(t) and 0 < t < 1 for t in cutoffs):
        raise ValueError("Correlation thresholds must lie strictly between zero and one.")
    repeats = _count(cfg, "cluster_bootstrap_repeats", 100)
    seed = int(cfg.get("seed", 42))
    absolute = bool(cfg.get("absolute_correlation", False))
    corrs, reference, records, partial, memberships = {}, {}, [], [], []
    for space, matrix in spaces.items():
        for method in ["pearson", "spearman"]:
            corr, n = _correlation(matrix, method)
            corrs[space, method] = corr
            pd.DataFrame(corr, index=columns, columns=columns).to_csv(
                out_dir / f"correlation_{space}_{method}.csv", index_label="taxon_index"
            )
            for i in range(len(columns)):
                for j in range(i + 1, len(columns)):
                    records.append(
                        dict(
                            representation=space,
                            method=method,
                            taxon_a=columns[i],
                            taxon_b=columns[j],
                            r=corr[i, j],
                            n=n,
                        )
                    )
            for row in _partial(matrix, np.log10(a.sum(axis=1) + 1), method):
                partial.append(
                    dict(
                        representation=space,
                        method=method,
                        taxon_a=columns[row.pop("i")],
                        taxon_b=columns[row.pop("j")],
                        **row,
                        control="log10(sum_of_16_PCR_assays + 1)",
                        caveat="Panel sum contains the tested taxa: part-whole adjustment, not independent total-load control.",
                    )
                )
            for cutoff in cutoffs:
                for strategy in ["connected_components", "hierarchical_average_1_abs_r"]:
                    key = f"{space}|{method}|{cutoff:g}|{strategy}"
                    labels = _memberships(corr, cutoff, strategy, absolute)
                    reference[key] = (space, method, cutoff, strategy, labels)
                    for taxon, cluster in zip(columns, labels):
                        memberships.append(
                            dict(
                                configuration=key,
                                representation=space,
                                method=method,
                                threshold=cutoff,
                                strategy=strategy,
                                absolute_correlation=(
                                    absolute if strategy == "connected_components" else True
                                ),
                                taxon=taxon,
                                cluster=int(cluster),
                                cluster_size=int((labels == cluster).sum()),
                            )
                        )
    pd.DataFrame(records).to_csv(out_dir / "correlation_pairs.csv", index=False)
    partial_table = pd.DataFrame(partial)
    partial_table["p_bh_120"] = partial_table.groupby(["representation", "method"])["p"].transform(
        lambda p: _adjust(p)
    )
    partial_table.to_csv(out_dir / "partial_correlations.csv", index=False)
    pd.DataFrame(memberships).to_csv(out_dir / "cluster_memberships.csv", index=False)
    accum = {key: np.zeros((len(columns), len(columns))) for key in reference}
    jaccard_rows = []
    rng = np.random.default_rng(seed)
    triangle = np.triu_indices(len(columns), 1)
    for repeat in range(repeats):
        sample = a[rng.integers(len(a), size=len(a))]
        bootstrap_spaces = representations(sample, config)
        bcorrs = {
            (space, method): _correlation(matrix, method)[0]
            for space, matrix in bootstrap_spaces.items()
            for method in ["pearson", "spearman"]
        }
        for key, (space, method, cutoff, strategy, original) in reference.items():
            labels = _memberships(bcorrs[space, method], cutoff, strategy, absolute)
            same, original_same = (
                labels[:, None] == labels[None, :],
                original[:, None] == original[None, :],
            )
            accum[key] += same
            union = (same | original_same)[triangle].sum()
            overlap = (same & original_same)[triangle].sum()
            jaccard_rows.append(
                dict(
                    configuration=key,
                    repeat=repeat + 1,
                    reference_cluster="partition_pairs",
                    reference_size=len(columns),
                    jaccard=float(overlap / union) if union else 1.0,
                )
            )
            boot_sets = [set(np.flatnonzero(labels == group)) for group in np.unique(labels)]
            for group in np.unique(original):
                members = set(np.flatnonzero(original == group))
                if len(members) >= 2:
                    best = max(len(members & other) / len(members | other) for other in boot_sets)
                    jaccard_rows.append(
                        dict(
                            configuration=key,
                            repeat=repeat + 1,
                            reference_cluster=str(group),
                            reference_size=len(members),
                            jaccard=best,
                        )
                    )
    coclustering = []
    if repeats:
        for key, matrix in accum.items():
            for i in range(len(columns)):
                for j in range(i, len(columns)):
                    coclustering.append(
                        dict(
                            configuration=key,
                            taxon_a=columns[i],
                            taxon_b=columns[j],
                            probability=matrix[i, j] / repeats,
                            bootstrap_repeats=repeats,
                        )
                    )
    pd.DataFrame(
        coclustering,
        columns=["configuration", "taxon_a", "taxon_b", "probability", "bootstrap_repeats"],
    ).to_csv(out_dir / "cluster_coclustering.csv", index=False)
    boot = pd.DataFrame(
        jaccard_rows,
        columns=["configuration", "repeat", "reference_cluster", "reference_size", "jaccard"],
    )
    boot.to_csv(out_dir / "cluster_bootstrap_jaccard.csv", index=False)
    summaries = []
    for (key, group), rows in boot.groupby(["configuration", "reference_cluster"], sort=False):
        summaries.append(
            dict(
                configuration=key,
                reference_cluster=group,
                bootstrap_repeats=len(rows),
                jaccard_mean=float(rows.jaccard.mean()),
                jaccard_p025=float(rows.jaccard.quantile(0.025)),
                jaccard_p975=float(rows.jaccard.quantile(0.975)),
            )
        )
    pd.DataFrame(
        summaries,
        columns=[
            "configuration",
            "reference_cluster",
            "bootstrap_repeats",
            "jaccard_mean",
            "jaccard_p025",
            "jaccard_p975",
        ],
    ).to_csv(out_dir / "cluster_stability_summary.csv", index=False)
    return dict(configurations=len(reference), cluster_bootstrap_repeats=repeats)


def _metadata(frame, config, out_dir):
    cfg = config.get("statistics", {})
    filename = cfg.get("metadata_file") or config.get("sensitivity", {}).get("metadata_file")
    requested = list(cfg.get("extra_covariates", []))

    def excluded_total(column):
        normalized = re.sub(r"[^a-z0-9]", "", column.lower())
        return "totalbacteria" in normalized or normalized in {
            "totalz",
            "universalqpcrtotal",
            "universal16stotal",
            "universal16srrnatotal",
        }

    if any(excluded_total(c) for c in requested):
        raise ValueError("Independent total-bacteria covariates are excluded from this analysis.")
    inventory = []
    if filename:
        path = Path(filename)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        meta = pd.read_csv(path, dtype={"participant_id": str})
        if (
            "participant_id" not in meta
            or meta.participant_id.isna().any()
            or meta.participant_id.duplicated().any()
        ):
            raise ValueError("Metadata require unique, nonmissing participant_id values.")
        frame = frame.copy()
        frame["participant_id"] = frame.participant_id.astype(str)
        if set(meta.participant_id) != set(frame.participant_id):
            raise ValueError(
                "Metadata IDs must exactly match the PCR cohort; no silent exclusions or additions."
            )
        overlap = (set(meta.columns) & set(frame.columns)) - {"participant_id"}
        if overlap:
            raise ValueError(f"Metadata may not overwrite existing columns: {sorted(overlap)}")
        forbidden = [c for c in meta if excluded_total(c)]
        if forbidden:
            raise ValueError("Independent total-bacteria metadata are excluded from this analysis.")
        frame = frame.merge(
            meta, on="participant_id", how="left", validate="one_to_one", sort=False
        )
    for field in ["periodontal_status", "PD", "BOP", "tongue_coating", "salivary_flow", *requested]:
        inventory.append(
            dict(
                variable=field,
                available=field in frame,
                n_missing=int(frame[field].isna().sum()) if field in frame else None,
                source=(
                    "supplied metadata or existing frame"
                    if field in frame
                    else "not supplied; cannot fabricate"
                ),
            )
        )
    pd.DataFrame(inventory).drop_duplicates("variable").to_csv(
        out_dir / "covariate_inventory.csv", index=False
    )
    return frame, requested


def _associations(frame, columns, extras, config, out_dir):
    cfg = config.get("statistics", {})
    binary_method = cfg.get("binary_association_method", "logit")
    if binary_method not in {"logit", "firth"}:
        raise ValueError("binary_association_method must be logit or firth.")
    default_taxa = [
        c
        for c in columns
        if any(
            s in re.sub(r"[^a-z]", "", c.lower())
            for s in ["eubacteriumnodatum", "porphyromonasgingivalis", "campylobacterrectus"]
        )
    ]
    selected = list(cfg.get("association_taxa", default_taxa))
    covariates = list(
        dict.fromkeys([c for c in ["Age", "Sex", "Smoking", "Oral_hygiene", *extras] if c in frame])
    )
    missing_covariates = [c for c in extras if c not in frame]
    rows = []
    try:
        import statsmodels.api as sm
    except ImportError:
        rows.append(
            dict(
                status="not_run",
                reason="optional statsmodels is not installed; Firth is not implemented; no substitute estimates were fabricated",
                method="adjusted_associations",
                requested_binary_method=binary_method,
            )
        )
    else:
        for taxon in selected:
            if taxon not in columns:
                raise ValueError(
                    f"Association taxon is not one of the sixteen PCR targets: {taxon}"
                )
            for target in ["Halitosis", "H2S_ppb", "CH3SH_ppb", "VSCs_ppb"]:
                base = dict(
                    taxon=taxon,
                    target=target,
                    covariates=json.dumps(covariates),
                    predictor_transform="log10(copies/mL + 1)",
                    response_transform="none" if target == "Halitosis" else "log1p(ppb)",
                )
                if missing_covariates:
                    rows.append(
                        dict(
                            **base,
                            status="not_run",
                            reason="Requested covariates absent: " + ", ".join(missing_covariates),
                        )
                    )
                    continue
                if target == "Halitosis" and binary_method == "firth":
                    rows.append(
                        dict(
                            **base,
                            status="not_run",
                            method="Firth",
                            reason="Firth is not implemented; conventional Logit is not relabeled as Firth.",
                        )
                    )
                    continue
                selected_frame = frame[[taxon, target, *covariates]].dropna()
                x = pd.get_dummies(selected_frame[covariates], drop_first=True, dtype=float)
                x.insert(
                    0, "taxon_log10", np.log10(selected_frame[taxon].to_numpy(dtype=float) + 1)
                )
                x = sm.add_constant(x, has_constant="add").astype(float)
                y = selected_frame[target].astype(float)
                if target != "Halitosis":
                    y = np.log1p(y)
                if len(x) <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
                    rows.append(
                        dict(
                            **base,
                            status="not_run",
                            n=len(x),
                            reason="Insufficient observations or rank-deficient design.",
                        )
                    )
                    continue
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fitted = (
                            sm.Logit(y, x).fit(disp=0, maxiter=200)
                            if target == "Halitosis"
                            else sm.OLS(y, x).fit(cov_type="HC3")
                        )
                    converged = bool(getattr(fitted, "mle_retvals", {}).get("converged", True))
                    separation = any("separation" in str(w.message).lower() for w in caught)
                    ci = fitted.conf_int().loc["taxon_log10"]
                    beta, p = float(fitted.params["taxon_log10"]), float(
                        fitted.pvalues["taxon_log10"]
                    )
                    valid = converged and not separation and np.isfinite([beta, p, *ci]).all()
                    row = dict(
                        **base,
                        status="ok" if valid else "diagnostic_failure",
                        n=len(x),
                        n_parameters=x.shape[1],
                        events=int(y.sum()) if target == "Halitosis" else None,
                        converged=converged,
                        separation_warning=separation,
                        condition_number=float(np.linalg.cond(x)),
                        method=(
                            "unweighted_Logit_Wald"
                            if target == "Halitosis"
                            else "OLS_HC3_log1p_gas"
                        ),
                        warnings=" | ".join(str(w.message) for w in caught),
                        coefficient=beta if valid else np.nan,
                        ci_low=float(ci.iloc[0]) if valid else np.nan,
                        ci_high=float(ci.iloc[1]) if valid else np.nan,
                        p=p if valid else np.nan,
                    )
                    if target == "Halitosis" and valid:
                        row.update(
                            odds_ratio=float(np.exp(beta)),
                            or_ci_low=float(np.exp(ci.iloc[0])),
                            or_ci_high=float(np.exp(ci.iloc[1])),
                        )
                    rows.append(row)
                except Exception as exc:
                    rows.append(
                        dict(**base, status="failed", reason=f"{type(exc).__name__}: {exc}")
                    )
    table = pd.DataFrame(
        rows or [dict(status="not_run", reason="No prespecified taxa matched the PCR targets.")]
    )
    if "p" in table:
        table["p_bh_within_target"] = table.groupby("target")["p"].transform(lambda p: _adjust(p))
    table.to_csv(out_dir / "adjusted_associations.csv", index=False)
    return dict(
        optional_statsmodels_available="sm" in locals(),
        association_rows=len(table),
        firth_implemented=False,
        requested_binary_association_method=binary_method,
    )


def run_statistics(dataset, config, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, columns, a, y = _data(dataset)
    original_labels = frame.Halitosis.copy()
    spaces = representations(a, config)
    pd.DataFrame(
        {"participant_id": frame.participant_id, "Halitosis": y, "PCR_panel_total": a.sum(axis=1)}
    ).to_csv(out_dir / "PCR_panel_totals.csv", index=False)
    pd.DataFrame(
        [
            dict(
                group=g,
                definition="sum of sixteen targeted PCR concentrations",
                units="copies/mL",
                **_summary(a.sum(axis=1)[y == g]),
            )
            for g in [0, 1]
        ]
    ).to_csv(out_dir / "PCR_panel_total_summary.csv", index=False)
    pd.DataFrame(
        [
            dict(
                taxon=c,
                assay_index=i + 1,
                included=True,
                display_alias=(
                    "Lacticaseibacillus casei" if "casei" in c.lower() else c.removeprefix("PCR_")
                ),
                multiplicity_family_size=16,
            )
            for i, c in enumerate(columns)
        ]
    ).to_csv(out_dir / "PCR_target_inventory.csv", index=False)
    _, diversity_summary = _diversity(frame, a, y, out_dir)
    reference_path = config.get("statistics", {}).get("reference_diversity_summary")
    reference_match = None
    if reference_path:
        reference_path = Path(reference_path)
        if not reference_path.is_absolute():
            reference_path = Path(__file__).resolve().parents[1] / reference_path
        reference = pd.read_csv(reference_path, encoding="utf-8-sig").set_index("metric")
        current = diversity_summary.set_index("metric")
        if set(reference.index) != set(current.index):
            raise ValueError("Standalone diversity reference has different metric definitions.")
        common = [c for c in reference.columns if c in current.columns]
        for metric in current.index:
            if not np.allclose(
                current.loc[metric, common].to_numpy(dtype=float),
                reference.loc[metric, common].to_numpy(dtype=float),
                rtol=1e-9,
                atol=1e-10,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Integrated PCR diversity differs from the standalone reference: {metric}"
                )
        reference_match = True
    _species(columns, spaces, y, config, out_dir)
    cluster_info = _clusters(columns, spaces, a, config, out_dir)
    merged, extras = _metadata(frame, config, out_dir)
    association_info = _associations(merged, columns, extras, config, out_dir)
    if not frame.Halitosis.equals(original_labels) or not dataset.frame.Halitosis.equals(
        original_labels
    ):
        raise AssertionError("Original labels changed during descriptive analysis.")
    audit = dict(
        status="complete",
        n=len(frame),
        pcr_targets=16,
        class_counts={str(g): int((y == g).sum()) for g in [0, 1]},
        existing_labels_preserved=True,
        independent_total_used=False,
        panel_total_definition="sum of the sixteen targeted PCR concentrations",
        representations={
            "raw": "copies/mL",
            "log10": "log10(copies/mL + 1)",
            "relative": "assay / sum of sixteen assays; undefined if panel sum zero",
            "clr": "ln(x + fixed pseudocount) minus row mean ln(x + pseudocount)",
        },
        clr_pseudocount_copies_ml=float(
            config.get("sensitivity", {}).get("clr_pseudocount_copies_ml", 1)
        ),
        diversity="Targeted panel signal indices; natural-log Shannon; Chao1 not computed.",
        standalone_diversity_reference_match=reference_match,
        uncertainty="Effect CIs resample participants separately within the supplied label groups.",
        correlation_caveat="Partial correlations control the tested taxa's own panel sum; part-whole dependence remains.",
        cluster_caveat="Connected components are not cliques; hierarchical thresholds are average-link distances, not all-pair correlation requirements.",
        **cluster_info,
        **association_info,
    )
    _json(out_dir / "statistics_manifest.json", audit)
    return audit
