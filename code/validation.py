"""Participant-block uncertainty and resumable, full-pipeline outcome tests.

The main estimand is the mean of the pooled OOF metric computed separately
within each repetition. Averaging predictions first is a different estimand
and is reported separately. Bootstrap intervals condition on the fitted OOF
predictions: they do not reproduce model training or selection uncertainty.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score


_BASE_KEYS = ["target", "model", "algorithm"]
_OUTCOMES = ["Halitosis", "H2S_ppb", "CH3SH_ppb", "VSCs_ppb"]
_SCOPE = "conditional_fixed_OOF_participant_block_percentile"


def _holm(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Holm correction requires finite p-values in [0, 1].")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values)))))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _seed(seed, *parts):
    token = json.dumps([int(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def _read_csv(path):
    try:
        return pd.read_csv(path, dtype={"participant_id": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _keys(task):
    return _BASE_KEYS + (["target_scale"] if task == "regression" else [])


def _load_oof(out_dir, task, expected_repeats=None):
    candidates = [Path(out_dir) / f"{task}_oof.csv"]
    candidates.append(Path(out_dir) / ("class_oof.csv" if task == "classification" else "reg_oof.csv"))
    present = [p for p in candidates if p.exists()]
    if not present:
        return pd.DataFrame()
    frame = _read_csv(present[0])
    if frame.empty:
        return frame
    required = set(_BASE_KEYS + ["participant_id", "repeat", "fold", "y_true",
                                "y_prob" if task == "classification" else "y_pred"])
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"{task} OOF missing columns: {missing}")
    if task == "regression" and "target_scale" not in frame:
        frame["target_scale"] = "raw"
    for column in ["participant_id", *_keys(task)]:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{task} OOF has missing/empty {column}.")
    for column in ["repeat", "fold"]:
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} must contain finite integer identifiers.")
        frame[column] = numeric.astype(int)
    numeric_columns = ["y_true", "y_prob" if task == "classification" else "y_pred"]
    if task == "regression":
        numeric_columns += [c for c in ["y_true_transformed", "y_pred_transformed",
            "train_mean_prediction", "train_mean_transformed"] if c in frame]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{task} OOF contains non-finite {column}.")
    if task == "classification":
        if not set(frame.y_true.unique()).issubset({0, 1}):
            raise ValueError("Classification OOF labels must be 0 and 1.")
        if ((frame.y_prob < 0) | (frame.y_prob > 1)).any():
            raise ValueError("Classification probabilities must lie in [0, 1].")
    if frame.duplicated(_keys(task) + ["participant_id", "repeat"]).any():
        raise ValueError("Duplicate participant OOF prediction within a model/repetition.")
    # A different cohort or fold layout across models invalidates paired claims.
    reference = {}
    for key, group in frame.groupby(_keys(task), sort=True, dropna=False):
        repeats = sorted(group["repeat"].unique())
        if expected_repeats is not None and len(repeats) != expected_repeats:
            raise ValueError(f"OOF repetition count for {key} is {len(repeats)}, expected {expected_repeats}.")
        ids = None
        for repeat, part in group.groupby("repeat", sort=True):
            ordered = part.sort_values("participant_id")
            current = tuple(ordered.participant_id)
            if ids is None:
                ids = current
            elif ids != current:
                raise ValueError("Participant set changes across repetitions.")
            reference_key = (str(key[0]), int(repeat))
            audit = ordered[["participant_id", "fold", "y_true"]].reset_index(drop=True)
            if reference_key in reference and not audit.equals(reference[reference_key]):
                raise ValueError("Models do not share identical participant IDs, outcomes and test folds.")
            reference.setdefault(reference_key, audit)
        if (group.groupby("participant_id").y_true.nunique() != 1).any():
            raise ValueError("A participant outcome changes across repetitions.")
    return frame


def _bundle(group, task):
    ids = sorted(group.participant_id.unique())
    repeats = sorted(group["repeat"].unique())
    def matrix(column):
        result = group.pivot(index="repeat", columns="participant_id", values=column)
        return result.reindex(index=repeats, columns=ids).to_numpy(dtype=float)
    result = {"ids": ids, "repeats": repeats, "task": task,
              "y_raw": matrix("y_true"), "fold": matrix("fold")}
    if task == "classification":
        result["p"] = matrix("y_prob")
        result["y"] = result["y_raw"]
    else:
        scale = str(group.target_scale.iloc[0])
        result["scale"] = scale
        result["p_raw"] = matrix("y_pred")
        if scale != "raw" and not {"y_true_transformed", "y_pred_transformed"}.issubset(group):
            raise ValueError("Transformed-target OOF must retain transformed outcomes and predictions.")
        result["y"] = matrix("y_true_transformed") if "y_true_transformed" in group else result["y_raw"]
        result["p"] = matrix("y_pred_transformed") if "y_pred_transformed" in group else result["p_raw"]
        result["dummy"] = matrix("train_mean_prediction") if "train_mean_prediction" in group else None
        result["dummy_t"] = matrix("train_mean_transformed") if "train_mean_transformed" in group else result["dummy"] if scale == "raw" else None
    return result


def _auc(y, p):
    positive = np.asarray(y) == 1
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = stats.rankdata(p, method="average")
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _r2_rows(y, prediction):
    denominator = np.sum((y - y.mean(axis=1, keepdims=True)) ** 2, axis=1)
    numerator = np.sum((y - prediction) ** 2, axis=1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0) * -1 + 1


def _all_finite_mean(values):
    return float(np.mean(values)) if np.isfinite(values).all() else np.nan


def _core(bundle, indices=None):
    indices = np.arange(len(bundle["ids"])) if indices is None else indices
    y, p = bundle["y"][:, indices], bundle["p"][:, indices]
    if bundle["task"] == "classification":
        return {"auc": _all_finite_mean(np.array([_auc(a, b) for a, b in zip(y, p)]))}
    yr, pr = bundle["y_raw"][:, indices], bundle["p_raw"][:, indices]
    values = {"r2": _all_finite_mean(_r2_rows(y, p)),
              "r2_raw_ppb": _all_finite_mean(_r2_rows(yr, pr)),
              "mae": float(np.abs(yr - pr).mean()),
              "rmse": float(np.sqrt(np.mean((yr - pr) ** 2, axis=1)).mean())}
    for name, truth, prediction, dummy in [
        ("skill", y, p, bundle["dummy_t"]),
        ("skill_raw_ppb", yr, pr, bundle["dummy"]),
    ]:
        if dummy is None:
            values[name] = np.nan
        else:
            denominator = np.sum((truth - dummy[:, indices]) ** 2, axis=1)
            numerator = np.sum((truth - prediction) ** 2, axis=1)
            skill = 1 - np.divide(numerator, denominator,
                out=np.full_like(numerator, np.nan), where=denominator > 0)
            values[name] = _all_finite_mean(skill)
    return values


def _classification_extra(y, p, threshold):
    label = p >= threshold
    positive, negative = y == 1, y == 0
    tp, tn = int((label & positive).sum()), int((~label & negative).sum())
    fp, fn = int((label & negative).sum()), int((~label & positive).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return dict(accuracy=(tp + tn) / len(y), sensitivity=sens, specificity=spec,
                balanced_accuracy=(sens + spec) / 2,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                average_precision=float(average_precision_score(y, p)))


def _draws(y, iterations, seed, classification):
    rng = np.random.default_rng(seed)
    if classification:
        groups = [np.flatnonzero(y == value) for value in (0, 1)]
        if any(len(group) == 0 for group in groups):
            raise ValueError("Stratified bootstrap requires both classes.")
    for _ in range(iterations):
        if classification:
            yield np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        else:
            yield rng.integers(0, len(y), size=len(y))


def _interval(values, iterations):
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return dict(low=np.nan, high=np.nan, iterations_requested=iterations,
                    iterations_valid=0, iterations_invalid=iterations,
                    status="disabled" if iterations == 0 else "undefined")
    low, high = np.percentile(valid, [2.5, 97.5])
    return dict(low=float(low), high=float(high), iterations_requested=iterations,
                iterations_valid=len(valid), iterations_invalid=iterations - len(valid),
                status="complete" if len(valid) == iterations else "degenerate_resamples_excluded")


def _bootstrap(bundle, iterations, seed, other=None):
    scores = {metric: [] for metric in _core(bundle)}
    if other is not None:
        if bundle["ids"] != other["ids"] or bundle["repeats"] != other["repeats"]:
            raise ValueError("Paired bootstrap requires identical IDs and repetitions.")
        if not np.array_equal(bundle["y_raw"], other["y_raw"]) or not np.array_equal(bundle["fold"], other["fold"]):
            raise ValueError("Paired models have different outcomes or fold assignments.")
    for indices in _draws(bundle["y_raw"][0], iterations, seed, bundle["task"] == "classification"):
        values = _core(bundle, indices)
        if other is not None:
            reference = _core(other, indices)
            values = {metric: value - reference[metric] for metric, value in values.items()}
        for metric, value in values.items():
            scores[metric].append(value)
    return {metric: _interval(values, iterations) for metric, values in scores.items()}


def _repeat_records(frame, task, threshold):
    records = []
    for key, group in frame.groupby(_keys(task), sort=True):
        meta = dict(zip(_keys(task), key))
        for repeat, part in group.groupby("repeat", sort=True):
            bundle = _bundle(part, task)
            row = dict(task=task, **meta, repeat=int(repeat), n_participants=len(part),
                       **_core(bundle))
            if task == "classification":
                row.update(_classification_extra(bundle["y"][0], bundle["p"][0], threshold))
            records.append(row)
    return records


def _average_predictions(bundle):
    result = dict(bundle)
    for name in ["y", "p", "y_raw", "p_raw", "dummy", "dummy_t", "fold"]:
        if name in bundle and bundle[name] is not None:
            result[name] = bundle[name].mean(axis=0, keepdims=True)
    result["repeats"] = ["participant_average"]
    return result


def _regression_comparisons(frame, out_dir):
    pairs, omnibus = [], []
    if not frame.empty:
        for (target, scale), family in frame.groupby(["target", "target_scale"], sort=True):
            family = family.copy()
            family["absolute_error"] = np.abs(family.y_true - family.y_pred)
            errors = family.groupby(["participant_id", "model", "algorithm"], sort=True).absolute_error.mean().unstack(["model", "algorithm"])
            if errors.isna().any().any():
                raise ValueError("Regression comparisons have unmatched participant IDs.")
            for left, right in itertools.combinations(errors.columns, 2):
                difference = errors[left].to_numpy() - errors[right].to_numpy()
                if np.equal(difference, 0).all():
                    statistic, p = 0.0, 1.0
                else:
                    statistic, p = stats.wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
                if not np.isfinite(p):
                    raise ValueError("Non-finite paired Wilcoxon result.")
                pairs.append(dict(target=target, target_scale=scale, model_a=left[0], algorithm_a=left[1],
                    model_b=right[0], algorithm_b=right[1], n_participants=len(errors),
                    mean_absolute_error_difference=float(difference.mean()), statistic=float(statistic), p=float(p),
                    method="participant_mean_across_repeats_absolute_error_Wilcoxon"))
            if len(errors.columns) >= 3:
                arrays = [errors[column].to_numpy() for column in errors]
                if np.all(errors.to_numpy() == errors.iloc[:, 0].to_numpy()[:, None]):
                    statistic, p, status = 0.0, 1.0, "all_model_errors_identical"
                else:
                    statistic, p = stats.friedmanchisquare(*arrays)
                    status = "complete"
                if not np.isfinite(p):
                    raise ValueError("Non-finite Friedman result.")
                omnibus.append(dict(target=target, target_scale=scale, n_participants=len(errors),
                    n_models=len(errors.columns), statistic=float(statistic), p=float(p), status=status,
                    method="participant_mean_across_repeats_absolute_error_Friedman"))
            else:
                omnibus.append(dict(target=target, target_scale=scale, n_participants=len(errors),
                    n_models=len(errors.columns), statistic=np.nan, p=np.nan, status="requires_at_least_three_models"))
    table = pd.DataFrame(pairs)
    if not table.empty:
        table["p_holm"] = table.groupby(["target", "target_scale"]).p.transform(
            lambda series: _holm(series.to_numpy()))
        table["family_comparisons"] = table.groupby(["target", "target_scale"]).p.transform("size")
    _atomic_csv(table, Path(out_dir) / "regression_pairwise.csv")
    _atomic_csv(pd.DataFrame(omnibus), Path(out_dir) / "regression_friedman.csv")


def _dummy_comparisons(bundles, task, iterations, seed):
    """Secondary paired reference, separate from conditional clinical increment."""
    rows, omissions = [], []
    for key, bundle in bundles.items():
        meta = dict(zip(_keys(task), key))
        if meta['algorithm'] == 'dummy':
            continue
        candidates = [(k,b) for k,b in bundles.items()
            if dict(zip(_keys(task),k))['algorithm']=='dummy'
            and all(dict(zip(_keys(task),k))[c]==meta[c] for c in ['target']+(['target_scale'] if task=='regression' else []))]
        if not candidates:
            omissions.append(dict(task=task,**meta,comparison_role='secondary_dummy_reference_not_conditional_PCR_increment',status='not_run',reason='dummy_reference_not_requested_or_not_available'))
            continue
        if len(candidates)!=1:
            raise ValueError('More than one Dummy comparator in an outcome/scale family.')
        reference_key, baseline = candidates[0]
        if bundle['ids'] != baseline['ids'] or bundle['repeats'] != baseline['repeats']:
            raise ValueError('Dummy paired comparison has mismatched participants/repeats.')
        point, reference = _core(bundle), _core(baseline)
        ci = _bootstrap(bundle, iterations, _seed(seed,'dummy_reference',task,*key),other=baseline)
        for metric in (['auc'] if task=='classification' else ['r2','mae','skill','r2_raw_ppb']):
            rows.append(dict(task=task,**meta,
                comparator=dict(zip(_keys(task),reference_key))['model'],comparator_algorithm='dummy',
                comparison_role='secondary_dummy_reference_not_conditional_PCR_increment',
                metric=metric,model_metric=point[metric],comparator_metric=reference[metric],
                delta=point[metric]-reference[metric],improvement_direction='negative' if metric=='mae' else 'positive',
                n_participants=len(bundle['ids']),n_repeats=len(bundle['repeats']),method=_SCOPE,**ci[metric]))
    return rows, omissions


def finalize_results(out_dir, config):
    """Finalize saved OOF results without fitting models or reading source data."""
    out_dir = Path(out_dir)
    iterations = int(config.get("bootstrap", {}).get("iterations", 10000))
    if iterations < 0:
        raise ValueError("bootstrap.iterations must be nonnegative.")
    seed = int(config.get("bootstrap", {}).get("seed", config.get("cv", {}).get("seed", 42)))
    expected = int(config.get("cv", {}).get("outer_repeats", 1))
    threshold = float(config.get("modeling", {}).get("threshold", .5))
    repeat_rows, intervals, delta_rows, omissions, participant_rows = [], [], [], [], []
    dummy_rows, dummy_omissions = [], []
    counts = {}
    for task in ["classification", "regression"]:
        frame = _load_oof(out_dir, task, expected)
        if frame.empty:
            counts[task] = 0
            continue
        repeat_rows.extend(_repeat_records(frame, task, threshold))
        bundles, summaries = {}, []
        existing_path = out_dir / f"{task}_summary.csv"
        existing = _read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
        for key, group in frame.groupby(_keys(task), sort=True):
            meta = dict(zip(_keys(task), key))
            bundle = _bundle(group, task)
            bundles[key] = bundle
            point = _core(bundle)
            average = _core(_average_predictions(bundle))
            per_repeat = [_core(_bundle(part, task)) for _, part in group.groupby("repeat", sort=True)]
            ci = _bootstrap(bundle, iterations, _seed(seed, task, *key))
            old = existing
            for column, value in meta.items():
                if column in old:
                    old = old[old[column].astype(str) == str(value)]
            row = old.iloc[0].to_dict() if len(old) == 1 else {}
            row.update(meta, n_participants=len(bundle["ids"]), n_repeats=len(bundle["repeats"]),
                       ci_scope=_SCOPE, primary_summary="mean_of_repeat_pooled_OOF_metrics")
            for metric, value in point.items():
                row[f"repeat_mean_{metric}"] = value
                row[f"repeat_sd_{metric}"] = float(np.std([record[metric] for record in per_repeat], ddof=1)) if len(per_repeat) > 1 else 0.0
                row[f"participant_mean_prediction_{metric}"] = average[metric]
                row[f"{metric}_ci_low"] = ci[metric]["low"]
                row[f"{metric}_ci_high"] = ci[metric]["high"]
                intervals.append(dict(task=task, **meta, metric=metric, method=_SCOPE, **ci[metric]))
            primary = "auc" if task == "classification" else "r2"
            row[primary] = point[primary]
            row[f"pooled_oof_{primary}"] = average[primary]
            if task == "classification":
                extras = [_classification_extra(a, b, threshold) for a, b in zip(bundle["y"], bundle["p"])]
                row.update({name: float(np.mean([record[name] for record in extras])) for name in extras[0]})
            else:
                row.update(mae=point["mae"], rmse=point["rmse"], r2_raw_ppb=point["r2_raw_ppb"], skill=point["skill"])
            summaries.append(row)
            for index, participant in enumerate(bundle["ids"]):
                item = dict(task=task, **meta, participant_id=participant,
                            y_true=float(bundle["y_raw"][0, index]), n_repeats=len(bundle["repeats"]))
                if task == "classification":
                    item["y_prob"] = float(bundle["p"][:, index].mean())
                else:
                    item.update(y_pred=float(bundle["p_raw"][:, index].mean()),
                        y_true_transformed=float(bundle["y"][0, index]),
                        y_pred_transformed=float(bundle["p"][:, index].mean()),
                        participant_mean_absolute_error=float(np.abs(bundle["y_raw"][:, index] - bundle["p_raw"][:, index]).mean()),
                        absolute_error_of_mean_prediction=float(abs(bundle["y_raw"][0, index] - bundle["p_raw"][:, index].mean())))
                participant_rows.append(item)
        for key, bundle in bundles.items():
            meta = dict(zip(_keys(task), key))
            if meta["model"] == "Clinical_only" or meta["algorithm"] == "dummy":
                continue
            baseline_key = tuple("Clinical_only" if name == "model" else meta[name] for name in _keys(task))
            if meta["algorithm"] == "selector":
                omissions.append(dict(task=task, **meta, reason="selector_has_no_single_matched_algorithm_clinical_comparator"))
                continue
            if baseline_key not in bundles:
                raise ValueError(f"Missing Clinical_only comparator with matched algorithm/scale for {key}.")
            baseline = bundles[baseline_key]
            point, reference = _core(bundle), _core(baseline)
            ci = _bootstrap(bundle, iterations, _seed(seed, "paired", task, *key), other=baseline)
            for metric in (["auc"] if task == "classification" else ["r2", "mae", "skill", "r2_raw_ppb"]):
                delta_rows.append(dict(task=task, **meta, comparator="Clinical_only", comparator_algorithm=meta["algorithm"],
                    metric=metric, model_metric=point[metric], comparator_metric=reference[metric],
                    delta=point[metric] - reference[metric], improvement_direction="negative" if metric == "mae" else "positive",
                    n_participants=len(bundle["ids"]), n_repeats=len(bundle["repeats"]), method=_SCOPE, **ci[metric]))
        reference_rows, reference_omissions = _dummy_comparisons(bundles,task,iterations,seed)
        dummy_rows.extend(reference_rows);dummy_omissions.extend(reference_omissions)
        _atomic_csv(pd.DataFrame(summaries), out_dir / f"{task}_summary.csv")
        if task == "regression":
            _regression_comparisons(frame, out_dir)
        counts[task] = len(summaries)
    _atomic_csv(pd.DataFrame(repeat_rows), out_dir / "repeat_metrics.csv")
    _atomic_csv(pd.DataFrame(intervals), out_dir / "conditional_bootstrap.csv")
    _atomic_csv(pd.DataFrame(delta_rows), out_dir / "paired_performance_deltas.csv")
    _atomic_csv(pd.DataFrame(omissions), out_dir / "paired_comparison_omissions.csv")
    dummy_columns = ["task", "target", "model", "algorithm", "target_scale", "comparator",
        "comparator_algorithm", "comparison_role", "metric", "model_metric", "comparator_metric",
        "delta", "improvement_direction", "n_participants", "n_repeats", "method", "low", "high",
        "iterations_requested", "iterations_valid", "iterations_invalid", "status"]
    _atomic_csv(pd.DataFrame(dummy_rows).reindex(columns=dummy_columns), out_dir / "dummy_reference_deltas.csv")
    _atomic_csv(pd.DataFrame(dummy_omissions, columns=["task", "target", "model", "algorithm", "target_scale",
        "comparison_role", "status", "reason"]), out_dir / "dummy_reference_omissions.csv")
    participant_table = pd.DataFrame(participant_rows)
    _atomic_csv(participant_table, out_dir / "participant_average_oof.csv")
    _atomic_csv(participant_table[participant_table.task == "regression"] if not participant_table.empty else participant_table,
                out_dir / "regression_participant_errors.csv")
    methods = dict(primary_summary="Mean of pooled OOF metrics calculated separately within each repetition.",
        separate_summary="Metric of participant-averaged predictions is reported separately, not substituted for the primary metric.",
        bootstrap_scope=_SCOPE, bootstrap_iterations=iterations,
        disclosure="Participants are resampled as blocks; every repetition and paired model stays with the participant. Classification resampling preserves class counts. Intervals condition on fixed OOF predictions and exclude retraining/selection uncertainty. Undefined regression resamples are counted explicitly.",
        regression_comparisons="Absolute raw-ppb errors are averaged across repetitions within participant before paired Wilcoxon/Friedman tests; Holm family is all pairs within gas and target scale.",
        r2_scope="r2 uses the declared target scale; r2_raw_ppb is separate. Train-mean skill is 1-SSE_model/SSE_training-mean-dummy and is not conventional R-squared.",
        dummy_reference="Secondary paired Dummy contrasts are separate from matched-algorithm clinical increment, and do not test conditional PCR contribution.",
        counts=counts)
    _atomic_json(out_dir / "validation_methods.json", methods)
    return methods


def _permutation_config(config):
    result = copy.deepcopy(config)
    settings = result.get("permutation", {})
    result.setdefault("cv", {}).update(outer_folds=int(settings.get("outer_folds", 5)),
        outer_repeats=int(settings.get("outer_repeats", 2)), inner_folds=int(settings.get("inner_folds", 3)))
    result.setdefault("bootstrap", {})["iterations"] = 0
    result.setdefault("importance", {})["permutations"] = 0
    result["_permutation_run"] = True
    return result


def _fingerprint(dataset, config):
    frozen = copy.deepcopy(config)
    frozen.pop("_config_path", None)
    for field in ["max_new_permutations", "batch_limit", "batch_size"]:
        frozen.get("permutation", {}).pop(field, None)
    frame = dataset.frame
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    digest.update(json.dumps(list(frame.columns)).encode())
    digest.update(json.dumps([str(dtype) for dtype in frame.dtypes]).encode())
    data_hash = digest.hexdigest()
    sources = {}
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    source = Path(dataset.source_path)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    payload = dict(config=frozen, frame_hash=data_hash, source_file_hash=source_hash,
        pcr_columns=list(dataset.pcr_columns), clinical_columns=list(dataset.clinical_columns),
        code_hashes=sources)
    payload["fingerprint"] = hashlib.sha256(json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return payload


def _score_key(record):
    return tuple(str(record.get(key, "")) for key in ["task", "target", "model", "algorithm", "target_scale"])


def _scores_from_run(folder, config):
    records = []
    for task in ["classification", "regression"]:
        frame = _load_oof(folder, task, int(config["cv"]["outer_repeats"]))
        if frame.empty:
            continue
        for key, group in frame.groupby(_keys(task), sort=True):
            bundle = _bundle(group, task)
            metric = "auc" if task == "classification" else "r2"
            value = _core(bundle)[metric]
            if not np.isfinite(value):
                raise ValueError(f"Undefined/non-finite permutation statistic for {task}, {key}.")
            records.append(dict(task=task, **dict(zip(_keys(task), key)), metric=metric,
                score=float(value), n_participants=len(bundle["ids"]), n_repeats=len(bundle["repeats"])))
    if not records:
        raise ValueError("Runner produced no valid OOF model scores.")
    return records


def _family(record, settings):
    if record["task"] == "classification":
        return f"classification:{record['target']}"
    scale = record.get("target_scale", "raw")
    if settings.get("gas_family", "joint") == "joint":
        return f"regression:all_gases:{scale}"
    return f"regression:{record['target']}:{scale}"


def _families(record, settings):
    primary = _family(record, settings)
    if record["task"] == "regression" and settings.get("gas_family", "joint") == "joint":
        return [primary, f"regression:{record['target']}:{record.get('target_scale', 'raw')}"]
    return [primary]


def _write_permutation_summary(root, observed, completed, settings, total):
    by_key = {_score_key(record): record for record in observed}
    candidates = [record for record in observed if record["algorithm"] != "dummy"]
    family_observed = {}
    for record in candidates:
        for family in _families(record, settings):
            family_observed[family] = max(family_observed.get(family, -np.inf), record["score"])
    null_maxima, all_scores = [], []
    for number, records in sorted(completed.items()):
        maxima = {}
        for record in records:
            all_scores.append(dict(permutation_id=number, **record))
            if record["algorithm"] != "dummy":
                for family in _families(record, settings):
                    maxima[family] = max(maxima.get(family, -np.inf), record["score"])
        for family, score in maxima.items():
            null_maxima.append(dict(permutation_id=number, family=family, maximum_score=score))
    n = len(completed)
    final = n == total
    rows = []
    for key, observation in by_key.items():
        values = [next(record["score"] for record in records if _score_key(record) == key) for records in completed.values()]
        family = _family(observation, settings)
        max_values = [record["maximum_score"] for record in null_maxima if record["family"] == family]
        per_gas = f"regression:{observation['target']}:{observation.get('target_scale', 'raw')}"
        per_gas_values = [record["maximum_score"] for record in null_maxima if record["family"] == per_gas]
        count = sum(value >= observation["score"] for value in values)
        corrected_count = sum(value >= observation["score"] for value in max_values)
        lower_count = sum(value <= observation["score"] for value in values)
        rows.append(dict(**observation, family=family, requested_iterations=total, completed_iterations=n,
            pending_iterations=total - n, status="complete" if final else "partial_not_final",
            upper_exceedances=count, maximum_exceedances=corrected_count,
            p_raw=(count + 1) / (total + 1) if final else np.nan,
            p_maximum_adjusted=(corrected_count + 1) / (total + 1) if final and max_values and observation["algorithm"] != "dummy" else np.nan,
            p_maximum_adjusted_per_gas=(sum(value >= observation["score"] for value in per_gas_values) + 1) / (total + 1)
                if final and per_gas_values and observation["algorithm"] != "dummy" else np.nan,
            p_lower=(lower_count + 1) / (total + 1) if final else np.nan,
            interim_p_raw=(count + 1) / (n + 1) if n else np.nan,
            interpretation="global outcome-feature independence; not PCR incremental effect conditional on clinical predictors"))
    families = []
    for family, observed_max in family_observed.items():
        values = [record["maximum_score"] for record in null_maxima if record["family"] == family]
        count = sum(value >= observed_max for value in values)
        families.append(dict(family=family, observed_maximum=observed_max,
            n_candidates=sum(family in _families(record, settings) for record in candidates),
            candidate_keys=json.dumps([_score_key(record) for record in candidates if family in _families(record, settings)]),
            requested_iterations=total, completed_iterations=n, pending_iterations=total - n,
            status="complete" if final else "partial_not_final", exceedances=count,
            p=(count + 1) / (total + 1) if final else np.nan,
            interim_p=(count + 1) / (n + 1) if n else np.nan,
            null_q025=np.percentile(values, 2.5) if values else np.nan,
            null_q975=np.percentile(values, 97.5) if values else np.nan))
    _atomic_csv(pd.DataFrame(rows), root / "permutation_model_tests.csv")
    _atomic_csv(pd.DataFrame(families), root / "maximum_statistic_tests.csv")
    _atomic_csv(pd.DataFrame(all_scores), root / "permutation_scores.csv")
    _atomic_csv(pd.DataFrame(null_maxima), root / "maximum_statistic_null.csv")
    return dict(status="complete" if final else "partial_not_final", requested_iterations=total,
                completed_iterations=n, pending_iterations=total - n)


def run_permutation(dataset, config, out_dir, runner=None, max_new_permutations=None):
    """Run/resume full-pipeline tests; a batch limit never changes the target B.

    Only outcome columns are shuffled, jointly by participant. Predictors,
    clinical variables, IDs and their order are preserved. Classification
    stratification is regenerated by the runner for each permuted label vector.
    """
    settings = config.get("permutation", {})
    total = int(settings.get("iterations", 1000))
    if total < 1:
        raise ValueError("permutation.iterations must be positive.")
    if max_new_permutations is None:
        max_new_permutations = settings.get("max_new_permutations")
    if max_new_permutations is not None:
        if isinstance(max_new_permutations, bool) or int(max_new_permutations) != max_new_permutations or max_new_permutations < 0:
            raise ValueError("max_new_permutations must be a nonnegative integer or None.")
        max_new_permutations = int(max_new_permutations)
    model_settings = config.get("modeling", {})
    if model_settings.get("panels", "fixed_exploratory") == "fixed_exploratory":
        if not settings.get("fixed_panels_prespecified", False) or not settings.get("fixed_panels_scope"):
            raise ValueError("Outcome tests require training-derived panels. A truly prespecified fixed family requires explicit fixed_panels_prespecified and fixed_panels_scope; same-cohort exploratory panels do not qualify.")
    if runner is None:
        from modeling import run_modeling
        runner = run_modeling
    root = Path(out_dir) / "permutation"
    root.mkdir(parents=True, exist_ok=True)
    frozen = _fingerprint(dataset, config)
    manifest_path = root / "checkpoint_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != frozen["fingerprint"]:
            raise ValueError("Permutation checkpoint mismatch: data, candidate configuration or code changed. Use a new output directory.")
    else:
        manifest = dict(fingerprint=frozen["fingerprint"], requested_iterations=total,
                        completed_ids=[], failures={}, observed_complete=False)
        _atomic_json(root / "frozen_design.json", frozen)
        _atomic_json(manifest_path, manifest)
    run_config = _permutation_config(config)
    observed_path = root / "observed_scores.json"
    if manifest.get("observed_complete"):
        if not observed_path.exists():
            raise ValueError("Completed observation checkpoint is missing its score file.")
        observed_payload = json.loads(observed_path.read_text(encoding="utf-8"))
        if observed_payload.get("fingerprint") != frozen["fingerprint"]:
            raise ValueError("Observed checkpoint fingerprint mismatch.")
        observed = observed_payload["scores"]
    else:
        runner(copy.deepcopy(dataset), copy.deepcopy(run_config), root / "observed")
        observed = _scores_from_run(root / "observed", run_config)
        _atomic_json(observed_path, dict(fingerprint=frozen["fingerprint"], scores=observed))
        manifest["observed_complete"] = True
        _atomic_json(manifest_path, manifest)
    expected_keys = {_score_key(record) for record in observed}
    if len(expected_keys) != len(observed) or not all(np.isfinite(record["score"]) for record in observed):
        raise ValueError("Observed candidate catalog contains duplicate score keys.")
    _atomic_csv(pd.DataFrame(observed), root / "observed_reference_scores.csv")
    if run_config["cv"]["outer_folds"] == 5 and run_config["cv"]["outer_repeats"] == 2:
        _atomic_csv(pd.DataFrame(observed), root / "observed_reference_5x2.csv")
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    completed = {}
    for path in sorted(checkpoints.glob("perm_*.json")):
        match = re.fullmatch(r"perm_(\d+)\.json", path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        number = int(match.group(1))
        if payload.get("fingerprint") != frozen["fingerprint"] or payload.get("permutation_id") != number or not 0 <= number < total:
            raise ValueError("Permutation checkpoint ID/fingerprint mismatch.")
        if len(payload["scores"]) != len(observed) or {_score_key(record) for record in payload["scores"]} != expected_keys:
            raise ValueError("Checkpoint candidate catalog differs from the observation.")
        if not all(np.isfinite(record["score"]) for record in payload["scores"]):
            raise ValueError("Checkpoint contains a non-finite score.")
        completed[number] = payload["scores"]
    if not set(manifest.get("completed_ids", [])).issubset(completed):
        raise ValueError("Manifest references a missing completed permutation checkpoint.")
    outcomes = [column for column in _OUTCOMES if column in dataset.frame]
    if not outcomes or "participant_id" not in dataset.frame or dataset.frame.participant_id.duplicated().any():
        raise ValueError("Permutation input requires unique participant IDs and outcome columns.")
    seed = int(settings.get("seed", 2026))
    new_count = 0
    for number in range(total):
        if number in completed:
            continue
        if max_new_permutations is not None and new_count >= max_new_permutations:
            break
        shuffled = copy.deepcopy(dataset)
        shuffled.frame = dataset.frame.copy(deep=True)
        order = np.random.default_rng(np.random.SeedSequence([seed, number])).permutation(len(dataset.frame))
        shuffled.frame.loc[:, outcomes] = dataset.frame.iloc[order][outcomes].to_numpy()
        try:
            runner(shuffled, copy.deepcopy(run_config), root / "runs" / f"perm_{number:06d}")
            scores = _scores_from_run(root / "runs" / f"perm_{number:06d}", run_config)
            if {_score_key(record) for record in scores} != expected_keys or len(scores) != len(observed):
                raise ValueError("Permutation changed the candidate catalog; failed/missing models cannot be silently dropped.")
            _atomic_json(checkpoints / f"perm_{number:06d}.json", dict(
                fingerprint=frozen["fingerprint"], permutation_id=number, scores=scores))
            completed[number] = scores
            new_count += 1
            manifest["failures"].pop(str(number), None)
        except Exception as exc:
            manifest["failures"][str(number)] = dict(type=type(exc).__name__, message=str(exc))
            manifest["completed_ids"] = sorted(completed)
            manifest["status"] = "failed_requires_retry"
            _atomic_json(manifest_path, manifest)
            _write_permutation_summary(root, observed, completed, settings, total)
            raise RuntimeError(f"Permutation {number} failed; it was not counted as a null observation.") from exc
        manifest["completed_ids"] = sorted(completed)
        _atomic_json(manifest_path, manifest)
    result = _write_permutation_summary(root, observed, completed, settings, total)
    manifest.update(result, completed_ids=sorted(completed))
    _atomic_json(manifest_path, manifest)
    _atomic_json(root / "permutation_methods.json", dict(**result,
        observed_cv=run_config["cv"], main_cv=config.get("cv", {}),
        primary_statistic="mean_of_repeat_pooled_OOF_AUROC_or_target_scale_R2",
        procedure="Every observation/permutation reruns the same training preprocessing, sampling, cluster discovery, candidate selection, tuning and refitting. Only post-fit uncertainty/importance are disabled in both branches.",
        hypothesis="Global outcome-feature independence, not the incremental PCR effect conditional on clinical variables.",
        family_scope=settings.get("gas_family", "joint"),
        gas_permutation="Same participant row permutation jointly applied to Halitosis and all available gas outcomes; predictors and participant IDs remain fixed.",
        p_formula="(exceedances_or_ties + 1)/(requested_iterations + 1), only after all requested iterations complete; partial estimates are explicitly interim.",
        fixed_panel_scope=settings.get("fixed_panels_scope")))
    return result
