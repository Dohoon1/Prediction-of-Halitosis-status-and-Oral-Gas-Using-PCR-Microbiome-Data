"""Export definitions, formatting and provenance without making new statistical tests."""

from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd


def _read(path):
    try:
        return pd.read_csv(path, dtype={"participant_id": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _seed_stability(out_dir, config):
    """Aggregate completed seed runs only; never average CIs or tune a seed."""
    base_seed = int(config.get("neural", {}).get("seed", 42))
    planned = sorted({base_seed, *map(int, config.get("sensitivity", {}).get("neural_seeds", []))})

    def normalized(c):
        c = copy.deepcopy(c)
        c.setdefault("neural", {})["seed"] = "SEED_ONLY"
        c.setdefault("sensitivity", {}).pop("active_model_variant", None)
        return json.dumps(c, sort_keys=True, default=str)

    wanted = normalized(config)
    mapping = {}
    registry = _read(out_dir / "sensitivity_model_variants.csv")
    for row in registry.to_dict("records"):
        c = json.loads(row["config"])
        seed = int(c.get("neural", {}).get("seed", 42))
        if normalized(c) == wanted and seed in planned:
            mapping[row["name"]] = seed
    scores = _read(out_dir / "sensitivity_model_summary.csv")
    if not scores.empty:
        scores = scores.loc[
            (scores.algorithm == "transformer") & scores.sensitivity_variant.isin(mapping)
        ].copy()
        scores["training_seed"] = scores.sensitivity_variant.map(mapping)
    rows = []
    if not scores.empty:
        if "target_scale" not in scores:
            scores["target_scale"] = "raw"
        keys = ["task", "target", "target_scale", "model", "algorithm"]
        if scores.duplicated(keys + ["training_seed"]).any():
            raise ValueError("Duplicate results for one training seed/protocol.")
        for key, g in scores.groupby(keys, dropna=False):
            for metric in [
                "repeat_mean_auc",
                "repeat_mean_r2",
                "mae",
                "rmse",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "f1",
                "average_precision",
                "brier",
            ]:
                if metric not in g:
                    continue
                values = pd.to_numeric(g[metric], errors="raise").dropna()
                if values.empty:
                    continue
                rows.append(
                    dict(zip(keys, key))
                    | dict(
                        metric=metric,
                        n_seeds_completed=len(values),
                        n_seeds_planned=len(planned),
                        mean_across_seeds=float(values.mean()),
                        sd_across_seeds=float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                        minimum=float(values.min()),
                        maximum=float(values.max()),
                        status="complete" if len(values) == len(planned) else "partial",
                        scope="Training-seed variation with fixed CV/protocol; not a confidence interval or independent participants",
                    )
                )
    scores.to_csv(out_dir / "transformer_seed_results.csv", index=False)
    pd.DataFrame(
        rows,
        columns=[
            "task",
            "target",
            "target_scale",
            "model",
            "algorithm",
            "metric",
            "n_seeds_completed",
            "n_seeds_planned",
            "mean_across_seeds",
            "sd_across_seeds",
            "minimum",
            "maximum",
            "status",
            "scope",
        ],
    ).to_csv(out_dir / "transformer_seed_summary.csv", index=False)
    (out_dir / "transformer_seed_inventory.json").write_text(
        json.dumps(
            dict(
                planned_seeds_including_reference=planned,
                status="not_run" if not rows else "summarized",
                completed_summary_rows=len(rows),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def write_reporting(out_dir, dataset, config, stage):
    out_dir = Path(out_dir)
    cv, m, neural = (config.get(k, {}) for k in ["cv", "modeling", "neural"])
    records = [
        ("cohort", "original supplied labels; sensitivity outcomes are separate"),
        ("pcr_targets", json.dumps(dataset.pcr_columns)),
        (
            "total_definition",
            "PCR_panel_total = sum of 16 targeted concentrations; independent Excel total excluded",
        ),
        ("gas_definition", "H2S + CH3SH; DMS not measured in this input"),
        ("normalization", m.get("pcr_transform", "none")),
        (
            "CLR_pseudocount_copies_per_mL",
            config.get("sensitivity", {}).get("clr_pseudocount_copies_ml", 1.0),
        ),
        (
            "panel_indices",
            "detected targets >0; Shannon -sum(p*ln(p)); Pielou H/ln(S) if S>1; no Chao1",
        ),
        (
            "species_tests",
            "Welch and two-sided asymptotic Mann-Whitney; Bonferroni/BH within each 16-taxon representation/test family",
        ),
        (
            "diversity_tests",
            "Welch and Mann-Whitney; Bonferroni within the three panel indices for each test",
        ),
        (
            "species_effect_bootstrap_iterations",
            config.get("statistics", {}).get("effect_bootstrap_repeats", 1000),
        ),
        (
            "cluster_stability_bootstrap_iterations",
            config.get("sensitivity", {}).get("cluster_bootstrap_repeats", 100),
        ),
        (
            "cluster_sensitivity_thresholds",
            json.dumps(
                config.get("sensitivity", {}).get("correlation_thresholds", [0.3, 0.4, 0.5, 0.6])
            ),
        ),
        (
            "partial_correlation_control",
            "16-target panel sum, with part-whole coupling; not independent universal 16S load",
        ),
        ("categorical_variables", "Sex, Smoking, Oral_hygiene; train-fitted encoding"),
        ("missing_policy", m.get("missing_policy", "error")),
        ("sampling", m.get("sampling", "none")),
        ("class_weight", m.get("class_weight")),
        ("outer_folds", cv.get("outer_folds")),
        ("outer_repeats", cv.get("outer_repeats", 1)),
        ("inner_folds", cv.get("inner_folds")),
        ("cv_seed", cv.get("seed")),
        ("neural_seed", neural.get("seed", 42)),
        ("cluster_discovery", m.get("panels")),
        (
            "cluster_graph",
            f"{m.get('correlation_method','pearson')} {'abs(r)' if m.get('absolute_correlation') else 'r'} {'>=' if m.get('correlation_inclusive') else '>'} {m.get('correlation_threshold')} connected components, not cliques",
        ),
        ("target_transform", m.get("target_transform", "none")),
        ("early_stopping", neural.get("early_stopping", False)),
        ("training_grid", json.dumps(config.get("grids", {}), sort_keys=True)),
        ("neural_settings", json.dumps(neural, sort_keys=True)),
        ("model_OOF_bootstrap_iterations", config.get("bootstrap", {}).get("iterations")),
        (
            "neural_seed_sensitivity",
            json.dumps(config.get("sensitivity", {}).get("neural_seeds", [])),
        ),
        (
            "bootstrap_scope",
            "conditional on fixed OOF fits; participant blocks preserve repeats/model pairs; does not refit/select",
        ),
        (
            "primary_metric",
            "mean of repeat-specific pooled OOF AUROC/R2; averaged-prediction metric separately labeled",
        ),
        (
            "increment_comparators",
            "primary: matched-algorithm Clinical_only; secondary: Dummy reference, not a conditional PCR increment",
        ),
        (
            "regression_skill",
            "1-SSE_model/SSE_train_mean_dummy, separately labeled from conventional R2",
        ),
        (
            "regression_comparisons",
            "participant-wise mean absolute OOF errors across repeats; exploratory Friedman/Wilcoxon, Holm within target",
        ),
        ("permutation_settings", json.dumps(config.get("permutation", {}), sort_keys=True)),
        (
            "permutation_scope",
            "recomputed observed/null same CV; label-permutation distinct from feature importance; global association, not conditional PCR increment",
        ),
        ("current_execution_stage", stage),
    ]
    pd.DataFrame(records, columns=["item", "definition_or_setting"]).to_csv(
        out_dir / "methods_inventory.csv", index=False, encoding="utf-8-sig"
    )
    _seed_stability(out_dir, config)
    inventory = []
    for task in ["classification", "regression"]:
        oof = _read(out_dir / f"{task}_oof.csv")
        if not oof.empty:
            pred = "y_prob" if task == "classification" else "y_pred"
            keys = ["participant_id", "target", "model", "algorithm"]
            if "target_scale" in oof:
                keys += ["target_scale"]
            if oof.groupby(keys, dropna=False).y_true.nunique().max() != 1:
                raise ValueError("Inconsistent outcomes across repeated OOF predictions.")
            spec = {
                "y_true": ("y_true", "first"),
                pred: (pred, "mean"),
                "repeated_predictions": (pred, "size"),
            }
            if task == "regression":
                spec["mean_absolute_error_across_repeats"] = ("absolute_error", "mean")
            average = oof.groupby(keys, as_index=False, dropna=False).agg(**spec)
            average.to_csv(out_dir / f"participant_average_{task}_oof.csv", index=False)
            if task == "classification":
                probability = []
                calibration = []
                for (target, model, algorithm), sub in average.groupby(
                    ["target", "model", "algorithm"]
                ):
                    for group, values in sub.groupby("y_true"):
                        probability.append(
                            dict(
                                target=target,
                                model=model,
                                algorithm=algorithm,
                                group=int(group),
                                n=len(values),
                                minimum=values.y_prob.min(),
                                maximum=values.y_prob.max(),
                                median=values.y_prob.median(),
                                mean=values.y_prob.mean(),
                                n_above_threshold=int(
                                    (values.y_prob >= m.get("threshold", 0.5)).sum()
                                ),
                            )
                        )
                    bins = pd.cut(sub.y_prob, np.linspace(0, 1, 11), include_lowest=True)
                    for interval, values in sub.groupby(bins, observed=True):
                        calibration.append(
                            dict(
                                target=target,
                                model=model,
                                algorithm=algorithm,
                                probability_bin=str(interval),
                                n=len(values),
                                mean_predicted=values.y_prob.mean(),
                                fraction_positive=values.y_true.mean(),
                                scope="descriptive participant-average fixed OOF; no calibration fitted",
                            )
                        )
                pd.DataFrame(probability).to_csv(out_dir / "probability_summary.csv", index=False)
                pd.DataFrame(calibration).to_csv(out_dir / "calibration_bins.csv", index=False)
        table = _read(out_dir / f"{task}_summary.csv")
        if table.empty:
            continue
        keys = ["target", "model", "algorithm"]
        inventory.extend(
            dict(task=task, **row) for row in table[keys].drop_duplicates().to_dict("records")
        )
        display = table.copy()
        for column in display:
            if not pd.api.types.is_numeric_dtype(display[column]):
                continue
            if column in {
                "n",
                "n_participants",
                "n_models",
                "repeats",
                "n_repeats",
                "outer_repeats",
            }:
                continue
            if column.startswith("p_") or column.endswith("_p") or column == "p":
                display[column] = display[column].map(
                    lambda v: "" if pd.isna(v) else "<0.001" if v < 0.001 else f"{v:.3f}"
                )
            else:
                display[column] = display[column].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        display.to_csv(out_dir / f"{task}_summary_Word.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inventory, columns=["task", "target", "model", "algorithm"]).to_csv(
        out_dir / "reported_model_inventory.csv", index=False
    )
    fitted = _read(out_dir / "selected_features.csv")
    complexity = []
    if not fitted.empty:
        for record in fitted.to_dict("records"):
            meta = json.loads(record.get("metadata", "{}"))
            keep = {
                k: record[k]
                for k in ["task", "target", "target_scale", "model", "algorithm", "repeat", "fold"]
                if k in record
            }
            rank = meta.get("design_rank_with_intercept", 0)
            counts = meta.get("class_counts_before", [])
            positive = counts[1] if len(counts) > 1 else None
            conventional = record["algorithm"] in {"logistic", "logistic_sklearn"}
            complexity.append(
                dict(
                    **keep,
                    transformed_features=meta.get("transformed_features"),
                    design_rank_with_intercept=rank,
                    identifiable_linear_predictor_df=max(0, rank - 1),
                    neural_or_estimator_parameter_count=meta.get("n_parameters"),
                    original_train_events=positive,
                    events_per_identifiable_predictor_df=(
                        positive / (rank - 1)
                        if conventional and positive is not None and rank > 1
                        else np.nan
                    ),
                    epv_scope=(
                        "linear logistic predictor only; not a Transformer capacity metric"
                        if conventional
                        else "not_applicable"
                    ),
                    fallback_reason=meta.get("fallback_reason", ""),
                    early_stopping=meta.get("early_stopping"),
                    initialization_seed=meta.get("initialization_seed"),
                    loss_last=meta.get("loss_last"),
                )
            )
    pd.DataFrame(complexity).to_csv(out_dir / "model_complexity.csv", index=False)
    audit = _read(out_dir / "fit_audit.csv")
    if not audit.empty:
        audit.to_csv(out_dir / "fold_scope_audit.csv", index=False)
    choice = _read(out_dir / "selector_choices.csv")
    if not choice.empty:
        keys = [
            k
            for k in [
                "task",
                "target",
                "target_scale",
                "model",
                "selected_model",
                "selected_algorithm",
            ]
            if k in choice
        ]
        choice.groupby(keys, as_index=False, dropna=False).size().rename(
            columns={"size": "outer_folds_selected"}
        ).to_csv(out_dir / "selection_frequency.csv", index=False)
    folds = _read(out_dir / "fold_metrics.csv")
    if not folds.empty:
        folds.loc[folds.model.isin(["Clinical_only", "Dummy_prior", "Dummy_train_mean"])].to_csv(
            out_dir / "baseline_diagnostics.csv", index=False
        )
    for filename in [
        "clinical_summary.csv",
        "clinical_categories.csv",
        "taxon_summary.csv",
        "PCR_panel_total_summary.csv",
        "PCR_panel_diversity_summary.csv",
        "species_comparisons.csv",
        "paired_performance_deltas.csv",
        "dummy_reference_deltas.csv",
        "regression_pairwise.csv",
        "regression_friedman.csv",
        "outcome_sensitivity_summary.csv",
        "outcome_sensitivity_enodatum_effects.csv",
        "repeat_vsc_reliability.csv",
    ]:
        table = _read(out_dir / filename)
        if table.empty:
            continue
        for col in table.select_dtypes(include=np.number):
            p_col = (
                col.startswith("p_")
                or col.endswith("_p")
                or "_p_" in col
                or col == "p"
                or "bonferroni" in col
                or "_bh_" in col
                or col.startswith("mannwhitney_bonf")
            )
            if p_col:
                table[col] = table[col].map(
                    lambda v: "" if pd.isna(v) else "<0.001" if v < 0.001 else f"{v:.3f}"
                )
            elif col in {
                "low",
                "high",
                "icc",
                "prevalence",
                "descriptive_max_auc",
                "nested_best_single_auc",
                "nested_best_candidate_auc",
            } or any(
                token in col
                for token in [
                    "mean",
                    "sd",
                    "median",
                    "q25",
                    "q75",
                    "delta",
                    "ci_low",
                    "ci_high",
                    "difference",
                    "estimate",
                ]
            ):
                table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        table.to_csv(
            out_dir / (Path(filename).stem + "_Word.csv"), index=False, encoding="utf-8-sig"
        )
    class_count = sum(r["task"] == "classification" for r in inventory)
    reg_counts = (
        pd.Series([r["target"] for r in inventory if r["task"] == "regression"])
        .value_counts()
        .to_dict()
    )
    legends = f"""Supplementary Methods. Reproducible analysis settings for the 16-target qPCR study.
Settings and actual model identities are saved in methods_inventory.csv and reported_model_inventory.csv. The supplied labels are retained. Total targeted bacterial abundance denotes the sum across the 16 qPCR targets; the separate Excel total is excluded. PCR panel diversity uses detected-target count, natural-log Shannon entropy and Pielou evenness; Chao1 is excluded. These are targeted-panel indices.

Model performance tables. All saved candidate models and nested selection procedures.
Actual rows in this run: {class_count} classification identities; regression identities per outcome: {reg_counts}. Primary performance is the mean of repeat-specific pooled OOF metrics. Fold SD and performance after averaging each participant's repeated OOF predictions are separate quantities. Confidence limits resample participants with all their repeated predictions and are conditional on the fitted OOF results. They do not include model refitting or selection uncertainty. Raw-ppb and log-target R2 are different quantities. MAE and RMSE in ppb use inverse-transformed predictions; inverse log predictions are not automatically conditional-mean concentrations. Classification uses probability predictions at the configured threshold.

Appendix regression comparisons. Exploratory paired comparisons of participant-level mean absolute OOF prediction errors.
Friedman and Wilcoxon compare absolute errors, not R2. Repeats are averaged within a participant before testing. Holm adjustment applies within the reported outcome family. Model counts and H2S/CH3SH/VSC headings must be taken from the saved tables rather than copied from the previous manuscript. Overlapping training folds limit inferential interpretation.

Abbreviations: qPCR, quantitative polymerase chain reaction; OOF, out-of-fold; CV, cross-validation; AUROC, area under the receiver operating characteristic curve; CI, confidence interval; SD, standard deviation; MAE, mean absolute error; RMSE, root mean squared error; CLR, centered log ratio; VSC, volatile sulfur compounds (here H2S + CH3SH); FWER, family-wise error rate. Performance values are displayed to two decimals; calculations retain full precision. P values must not be reported as 0.00.
"""
    (out_dir / "table_titles_legends_abbreviations.txt").write_text(legends, encoding="utf-8")
    return {
        "classification_model_identities": class_count,
        "regression_model_identities_by_target": reg_counts,
        "word_files_modified": False,
        "definitions_file": "methods_inventory.csv",
    }
