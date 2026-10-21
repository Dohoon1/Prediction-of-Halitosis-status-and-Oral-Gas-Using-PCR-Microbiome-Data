"""Prespecified PCR sensitivity definitions, optional model hooks and repeat data.

No alternative label overwrites the original outcome. No models are trained
unless the caller explicitly supplies a runner. Independent total-bacteria
measurements are excluded. Absent clinical measurements are never reconstructed.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import is_dataclass, replace
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd


def _merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Variant names must use letters, digits, underscores or hyphens.")
    return name


def build_model_variants(config):
    """Return a finite prespecified [(name, full_config)] list, not a grid winner.

    Explicit sensitivity.model_variants=[{name, overrides}] replaces the defaults.
    Transformations are performed by modeling.py from the original raw frame.
    """
    cfg = config.get("sensitivity", {})
    specs = cfg.get("model_variants")
    if specs is None:
        specs = [{"name": "reference", "overrides": {}}]
        for transform in ["none", "log10", "relative", "clr"]:
            specs.append(dict(name="representation_" + transform, overrides={"modeling": {"pcr_transform": transform}}))
        for cutoff in cfg.get("correlation_thresholds", [.3, .4, .5, .6]):
            specs.append(dict(name="cluster_r_" + str(cutoff).replace(".", "p"), overrides={"modeling": {"correlation_threshold": float(cutoff)}}))
        specs.extend([
            dict(name="spearman_clusters", overrides={"modeling": {"correlation_method": "spearman"}}),
            dict(name="no_sampling", overrides={"modeling": {"sampling": "none", "class_weight": None}}),
            dict(name="balanced_no_sampling", overrides={"modeling": {"sampling": "none", "class_weight": "balanced"}}),
            dict(name="gas_log1p", overrides={"modeling": {"tasks": ["regression"], "target_transform": "log1p"}}),
            dict(name="train_derived_clusters", overrides={"modeling": {"panels": "nested"}}),
            dict(name="fixed_exploratory_clusters", overrides={"modeling": {"panels": "fixed_exploratory"}}),
        ])
        for seed in cfg.get("neural_seeds", [11, 42, 2026]):
            specs.append(dict(name=f"neural_seed_{int(seed)}", overrides={"neural": {"seed": int(seed)}}))
    if not isinstance(specs, list):
        raise ValueError("sensitivity.model_variants must be a list of name/overrides definitions.")
    variants, names, fingerprints = [], set(), set()
    for spec in specs:
        name = _safe_name(spec["name"])
        if name in names:
            raise ValueError(f"Duplicate sensitivity variant name: {name}")
        names.add(name)
        overrides = spec.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("Variant overrides must be a nested configuration object.")
        if "data" in overrides:
            raise ValueError("Model variants must not change the data source or supplied-label policy.")
        variant = _merge(config, overrides)
        variant.setdefault("sensitivity", {})["active_model_variant"] = name
        # Same settings under two names are one experiment, not two replications.
        fingerprint_config = copy.deepcopy(variant)
        fingerprint_config["sensitivity"].pop("active_model_variant", None)
        fingerprint = json.dumps(fingerprint_config, sort_keys=True, default=str)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        variants.append((name, variant))
    return variants


def _dataset_copy(dataset, frame):
    if is_dataclass(dataset):
        return replace(dataset, frame=frame.copy(deep=True), audit=copy.deepcopy(dataset.audit))
    return SimpleNamespace(frame=frame.copy(deep=True), pcr_columns=list(dataset.pcr_columns),
        clinical_columns=list(dataset.clinical_columns), source_path=dataset.source_path,
        audit=copy.deepcopy(getattr(dataset, "audit", {})))


def _outcomes(dataset, config, out_dir):
    frame = dataset.frame
    cfg = config.get("sensitivity", {})
    custom = cfg.get("alternative_outcomes") or []
    if not isinstance(custom, list):
        raise ValueError("sensitivity.alternative_outcomes must be a list of explicit definitions.")
    rules = [dict(name="supplied_sex_cutoff_audit", kind="sex_threshold", target="VSCs_ppb",
                      cutoffs=config.get("data", {}).get("sex_vsc_thresholds", {"F": 65.79, "M": 79.94}),
                      reference=config.get("data", {}).get("sex_vsc_threshold_reference", ""), evaluate=False),
                 dict(name="sex_upper_quartile_audit", kind="sex_quantile", target="VSCs_ppb", quantile=.75,
                      reference="Cohort-relative high-gas group; not a validated diagnostic cutoff.", evaluate=False)] + custom
    definitions, participant_rows, contexts = [], [], []
    names = set()
    aliases = {"total_vsc": "VSCs_ppb", "H2S": "H2S_ppb", "CH3SH": "CH3SH_ppb"}
    for rule in rules:
        name = _safe_name(rule["name"])
        if name in names:
            raise ValueError("Alternative outcome names must be unique.")
        names.add(name)
        kind, reference = rule.get("kind", "threshold"), str(rule.get("reference") or "").strip()
        target = aliases.get(rule.get("target"), rule.get("target", "VSCs_ppb"))
        if target not in {"H2S_ppb", "CH3SH_ppb", "VSCs_ppb"}:
            raise ValueError("Alternative outcomes may use only supplied H2S, CH3SH or their total VSC.")
        if target not in frame:
            definitions.append(dict(name=name, target=target, kind=kind, status="not_run", reason="Target measurement absent."))
            continue
        values = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Alternative outcomes require finite nonnegative gas measurements for the full cohort.")
        thresholds = np.full(len(frame), np.nan)
        definition = {}
        if kind in {"sex_threshold", "sex_quantile", "sex_upper_quartile"}:
            if "Sex" not in frame or not frame.Sex.isin(["F", "M"]).all():
                definitions.append(dict(name=name, target=target, kind=kind, status="not_run", reason="Valid supplied sex categories are required."))
                continue
            for sex in ["F", "M"]:
                mask = frame.Sex.to_numpy() == sex
                if not mask.any():
                    continue
                if kind == "sex_threshold":
                    cutoff = rule.get("cutoffs", {}).get(sex)
                else:
                    q = float(rule.get("quantile", .75))
                    if not 0 < q < 1:
                        raise ValueError("Alternative quantiles must lie strictly between zero and one.")
                    cutoff = float(np.quantile(values[mask], q, method="linear"))
                    definition["quantile"] = q
                    definition["quantile_method"] = "linear"
                    definition["interpretation"] = "sample-derived relative high-gas group, not clinical diagnosis"
                if cutoff is None or not np.isfinite(float(cutoff)) or float(cutoff) < 0:
                    raise ValueError(f"Explicit nonnegative cutoff missing for {name}, sex {sex}.")
                thresholds[mask] = float(cutoff)
                definition[sex] = float(cutoff)
        elif kind == "threshold":
            cutoff = rule.get("cutoff_ppb")
            if cutoff is None:
                definitions.append(dict(name=name, target=target, kind=kind, status="not_run", reason="No explicit cutoff_ppb supplied; no cutoff invented."))
                continue
            if not np.isfinite(float(cutoff)) or float(cutoff) < 0:
                raise ValueError("Gas cutoffs must be finite and nonnegative.")
            thresholds[:] = float(cutoff)
            definition["cutoff_ppb"] = float(cutoff)
        else:
            raise ValueError(f"Unknown alternative outcome kind: {kind}")
        labels = (values >= thresholds).astype(int)
        source_ok = bool(reference)
        row = dict(name=name, target=target, kind=kind,
            definition=json.dumps(definition, ensure_ascii=False, sort_keys=True), reference=reference,
            positive_n=int(labels.sum()), negative_n=int((labels == 0).sum()),
            changed_from_original_n=int((labels != frame.Halitosis.to_numpy()).sum()),
            comparison="greater_than_or_equal", original_labels_preserved=True,
            status="defined" if source_ok else "audit_only_reference_missing",
            model_evaluation_requested=bool(rule.get("evaluate", False)))
        definitions.append(row)
        for participant_id, old, label, cutoff in zip(frame.participant_id, frame.Halitosis, labels, thresholds):
            participant_rows.append(dict(definition=name, participant_id=participant_id,
                original_label=int(old), alternative_label=int(label), cutoff_ppb=float(cutoff),
                changed=bool(old != label)))
        contexts.append((name, labels, rule, source_ok))
    pd.DataFrame(definitions).to_csv(out_dir / "outcome_sensitivity_definitions.csv", index=False)
    pd.DataFrame(participant_rows, columns=["definition", "participant_id", "original_label", "alternative_label", "cutoff_ppb", "changed"]).to_csv(out_dir / "outcome_sensitivity_participants.csv", index=False)
    return definitions, contexts


def _outcome_effects(dataset, config, definitions, contexts, summaries, out_dir):
    """A single review-requested taxon across criteria, separate from 16-taxon tests.

    Repeated definitions yielding identical labels share one multiplicity entry;
    pointwise bootstrap intervals are not simultaneous confidence intervals.
    """
    from pcr_statistics import _tests, _effect_interval, _cliff, _adjust
    frame=dataset.frame
    taxon=next((c for c in dataset.pcr_columns if re.sub(r'[^a-z]','',c.lower()).endswith('eubacteriumnodatum')),None)
    if any(d['name']=='original_supplied' for d in definitions):
        raise ValueError('original_supplied is reserved for the original-label reference.')
    refs=[dict(name='original_supplied',positive_n=int(frame.Halitosis.sum()),negative_n=int((frame.Halitosis==0).sum()),status='original_label_reference',reference='Supplied labels retained')]+definitions
    plans=[('original_supplied',frame.Halitosis.to_numpy(),{},True)]+contexts
    effects=[]
    repeats=int(config.get('statistics',{}).get('effect_bootstrap_repeats',1000))
    if repeats<0:
        raise ValueError('Effect bootstrap repetitions must be nonnegative.')
    for name,labels,rule,source_ok in plans:
        labels=np.asarray(labels,dtype=int)
        digest=hashlib.sha256(labels.astype(np.uint8).tobytes()).hexdigest()
        row=dict(name=name,taxon=taxon,label_definition_hash=digest,
            interpretation='Exploratory prespecified E. nodatum association, not diagnostic validation or an adjusted odds ratio',
            CI_scope='Pointwise participant bootstrap within outcome groups; no refitting, no simultaneous coverage',
            effect_status='not_run',reason='E. nodatum PCR target absent')
        if taxon is not None:
            v=np.log10(pd.to_numeric(frame[taxon],errors='raise').to_numpy(dtype=float)+1)
            a,b=v[(labels==0)&np.isfinite(v)],v[(labels==1)&np.isfinite(v)]
            row.update(n0=len(a),n1=len(b))
            if min(len(a),len(b))<2:
                row['reason']='At least two nonmissing measurements per group are required.'
            else:
                lo,hi,dlo,dhi=_effect_interval(a,b,repeats,int(digest[:8],16))
                pw,pm=_tests(a,b)
                row.update(effect_status='computed',reason='',nonhalitosis_mean_log10_plus1=float(a.mean()),halitosis_mean_log10_plus1=float(b.mean()),
                    mean_log10_copies_plus1_difference=float(b.mean()-a.mean()),mean_difference_ci_low=lo,mean_difference_ci_high=hi,
                    cliffs_delta=_cliff(a,b),cliffs_delta_ci_low=dlo,cliffs_delta_ci_high=dhi,p_welch=pw,p_mannwhitney=pm)
        effects.append(row)
    effect_table=pd.DataFrame(effects)
    valid=effect_table.loc[effect_table.effect_status=='computed'].drop_duplicates('label_definition_hash')
    for col in ['p_welch','p_mannwhitney']:
        if col not in effect_table:
            continue
        for method in ['bonferroni','bh']:
            correction=dict(zip(valid.label_definition_hash,_adjust(valid[col].to_numpy(),method)))
            effect_table[col+'_'+method+'_unique_criteria']=effect_table.label_definition_hash.map(correction)
    effect_table['unique_label_definitions_in_effect_family']=len(valid)
    effect_table['effect_test_family']='one prespecified taxon across unique outcome definitions; distinct from main 16-taxon family'
    effect_table.to_csv(out_dir/'outcome_sensitivity_enodatum_effects.csv',index=False)
    combined=pd.concat(summaries,ignore_index=True) if summaries else pd.DataFrame()
    baseline=out_dir/'classification_summary.csv'
    base_table=pd.read_csv(baseline) if baseline.is_file() and baseline.stat().st_size>1 else pd.DataFrame()
    rows=[]
    for definition in refs:
        name=definition['name'];row=dict(definition)
        effect=effect_table.loc[effect_table.name==name]
        if not effect.empty:
            row.update({k:v for k,v in effect.iloc[0].to_dict().items() if k!='name'})
        sub=base_table if name=='original_supplied' else (combined.loc[(combined.sensitivity_variant=='outcome_'+name)&(combined.task=='classification')]
            if not combined.empty else pd.DataFrame())
        metric=next((c for c in ['repeat_mean_auc','auc','pooled_oof_auc'] if c in sub),None)
        row.update(model_evaluation_status='not_run',descriptive_max_auc=np.nan,descriptive_best_model='',descriptive_best_algorithm='',nested_best_single_auc=np.nan,nested_best_candidate_auc=np.nan,
            maximum_scope='Descriptive maximum among evaluated candidates; not independent validation of a selected winner')
        if metric is not None and {'model','algorithm'}.issubset(sub):
            sub=sub.copy();sub[metric]=pd.to_numeric(sub[metric],errors='raise')
            sub=sub.loc[np.isfinite(sub[metric])]
            for selector,col in [('Nested_best_single','nested_best_single_auc'),('Nested_best_candidate','nested_best_candidate_auc')]:
                selected=sub.loc[sub.model==selector,metric]
                if len(selected)==1:row[col]=float(selected.iloc[0])
            ordinary=sub.loc[(sub.algorithm!='dummy')&(sub.algorithm!='selector')]
            if not ordinary.empty:
                best=ordinary.sort_values([metric,'model','algorithm'],ascending=[False,True,True]).iloc[0]
                row.update(model_evaluation_status='completed_saved_scores',descriptive_max_auc=float(best[metric]),descriptive_best_model=best.model,descriptive_best_algorithm=best.algorithm)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir/'outcome_sensitivity_summary.csv',index=False)
    return dict(outcome_effect_rows=len(effect_table),outcome_summary_rows=len(rows))


def icc2_1(values):
    """Two-way random, absolute-agreement, single-measure ICC for balanced data."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("ICC(2,1) needs >=3 participants and >=2 common occasions, with complete finite cells.")
    n, k = values.shape
    grand = values.mean()
    row, col = values.mean(axis=1), values.mean(axis=0)
    msr = k * ((row - grand) ** 2).sum() / (n - 1)
    msc = n * ((col - grand) ** 2).sum() / (k - 1)
    residual = values - row[:, None] - col[None, :] + grand
    mse = (residual ** 2).sum() / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denominator) if denominator > 0 else np.nan


def _repeats(dataset, config, out_dir):
    cfg = config.get("sensitivity", {})
    repeat_cfg = cfg.get("repeated_vsc", {}) or {}
    filename = repeat_cfg.get("file") or cfg.get("repeated_vsc_file")
    inventory, summary, cv_rows = [], [], []
    if not filename:
        inventory.append(dict(status="not_run", reason="Repeated VSC measurements were not supplied; ICC and CV cannot be reconstructed."))
    else:
        repeat_path = Path(filename)
        if not repeat_path.is_absolute():
            repeat_path = Path(__file__).resolve().parents[1] / repeat_path
        data = pd.read_csv(repeat_path, dtype={"participant_id": str})
        required = {"participant_id", "analyte", "visit", "replicate", "value_ppb"}
        if not required <= set(data):
            raise ValueError("Repeated VSC file must contain participant_id,analyte,visit,replicate,value_ppb.")
        if data.participant_id.isna().any() or not set(data.participant_id) <= set(dataset.frame.participant_id.astype(str)):
            raise ValueError("Repeat data contain missing IDs or participants outside the PCR cohort.")
        data["value_ppb"] = pd.to_numeric(data.value_ppb, errors="raise")
        if not np.isfinite(data.value_ppb).all() or (data.value_ppb < 0).any():
            raise ValueError("Repeated VSC values must be finite nonnegative measurements.")
        design = repeat_cfg.get("design") or cfg.get("repeated_vsc_design")
        if not design or design.get("icc_model") != "ICC2_1" or not design.get("exchangeable_occasions", False):
            inventory.append(dict(status="not_run", reason="Explicit ICC2_1 design and exchangeable_occasions=true are required.", observed_rows=len(data)))
        else:
            occasion_columns = design.get("occasion_columns")
            if not occasion_columns or not set(occasion_columns) <= {"visit", "replicate"}:
                raise ValueError("Repeated-measure design must name visit and/or replicate as occasion_columns.")
            if data[occasion_columns].isna().any().any():
                raise ValueError("Repeat occasion identifiers cannot be missing.")
            if repeat_cfg.get("visit_filter") is not None:
                data = data.loc[data.visit.astype(str) == str(repeat_cfg["visit_filter"])].copy()
            data["occasion"] = data[occasion_columns].astype(str).agg("|".join, axis=1)
            bootstrap = int(repeat_cfg.get("bootstrap_repeats", cfg.get("icc_bootstrap_repeats", 1000)))
            if bootstrap < 0:
                raise ValueError("ICC bootstrap_repeats must be nonnegative.")
            rng = np.random.default_rng(int(cfg.get("seed", 42)))
            for analyte, rows in data.groupby("analyte", sort=True):
                if rows.duplicated(["participant_id", "occasion"]).any():
                    inventory.append(dict(analyte=analyte, status="not_run", reason="Duplicate subject/occasion cells; no automatic averaging."))
                    continue
                wide = rows.pivot(index="participant_id", columns="occasion", values="value_ppb")
                n_incomplete = int(wide.isna().any(axis=1).sum())
                if n_incomplete and design.get("incomplete_policy", "error") != "complete_cases":
                    inventory.append(dict(analyte=analyte, status="not_run", reason="Unbalanced repeat data; explicit complete_cases policy required.", incomplete_participants=n_incomplete))
                    continue
                wide = wide.dropna()
                if len(wide) < 3 or wide.shape[1] < 2:
                    inventory.append(dict(analyte=analyte, status="not_run", reason="Insufficient actual repeated measurements.", n=len(wide), n_occasions=wide.shape[1]))
                    continue
                values = wide.to_numpy(dtype=float)
                observed = icc2_1(values)
                draws = []
                for _ in range(bootstrap):
                    value = icc2_1(values[rng.integers(len(values), size=len(values))])
                    if np.isfinite(value):
                        draws.append(value)
                low, high = np.percentile(draws, [2.5, 97.5]) if len(draws) >= 2 else [np.nan, np.nan]
                summary.append(dict(analyte=analyte, icc_model="ICC2_1", n=len(wide), n_occasions=wide.shape[1],
                    icc=observed, ci_low=low, ci_high=high, bootstrap_requested=bootstrap,
                    bootstrap_valid=len(draws), incomplete_participants=n_incomplete,
                    status="ok" if np.isfinite(observed) else "undefined_zero_variation", design=json.dumps(design, sort_keys=True)))
                inventory.append(dict(analyte=analyte, status="computed" if np.isfinite(observed) else "undefined", observed_rows=len(rows), n=len(wide), n_occasions=wide.shape[1]))
                for participant_id, row in wide.iterrows():
                    mean = float(row.mean())
                    cv_rows.append(dict(participant_id=participant_id, analyte=analyte, n_measurements=len(row),
                        mean_ppb=mean, cv_percent=float(row.std(ddof=1) / mean * 100) if mean > 0 else np.nan,
                        status="ok" if mean > 0 else "undefined_mean_zero"))
    if not inventory:
        inventory.append(dict(status="not_run", reason="No actual repeat observations match the supplied design/filter."))
    pd.DataFrame(inventory).to_csv(out_dir / "repeat_vsc_inventory.csv", index=False)
    pd.DataFrame(summary, columns=["analyte", "icc_model", "n", "n_occasions", "icc", "ci_low", "ci_high", "bootstrap_requested", "bootstrap_valid", "incomplete_participants", "status", "design"]).to_csv(out_dir / "repeat_vsc_reliability.csv", index=False)
    pd.DataFrame(cv_rows, columns=["participant_id", "analyte", "n_measurements", "mean_ppb", "cv_percent", "status"]).to_csv(out_dir / "repeat_vsc_cv.csv", index=False)
    return dict(repeat_measurement_supplied=bool(filename), reliability_estimates=len(summary))


def run_sensitivity(dataset, config, out_dir, runner=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, original = dataset.frame, dataset.frame.Halitosis.copy(deep=True)
    if not frame.Halitosis.isin([0, 1]).all():
        raise ValueError("Original labels must remain supplied binary labels.")
    definitions, contexts = _outcomes(dataset, config, out_dir)
    variants = build_model_variants(config)
    inventory, summaries = [], []

    def evaluate(name, variant_dataset, variant_config, family):
        row = dict(name=name, family=family, status="not_run", reason="No model runner supplied; definitions/plan only.")
        if runner is not None:
            destination = out_dir / "model_runs" / name
            destination.mkdir(parents=True, exist_ok=True)
            try:
                result = runner(variant_dataset, variant_config, destination)
                row.update(status="completed", reason="Explicit supplied runner returned.", result_metadata=json.dumps(result, default=str), results_path=str(destination))
                for task in ["classification", "regression"]:
                    path = destination / f"{task}_summary.csv"
                    if path.is_file() and path.stat().st_size > 1:
                        try:
                            table = pd.read_csv(path)
                        except pd.errors.EmptyDataError:
                            continue
                        table.insert(0, "sensitivity_variant", name)
                        table.insert(1, "sensitivity_family", family)
                        table.insert(2, "task", task)
                        summaries.append(table)
            except Exception as exc:
                row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                inventory.append(row)
                pd.DataFrame(inventory).to_csv(out_dir / "sensitivity_model_execution.csv", index=False)
                if config.get("sensitivity", {}).get("fail_fast", True):
                    raise
                return
        inventory.append(row)

    registry = []
    for name, variant in variants:
        registry.append(dict(name=name, config=json.dumps(variant, sort_keys=True, default=str),
            data_transform="Raw frame retained; modeling.pcr_transform is applied by modeling.py.",
            selection="All prespecified panels/algorithms are reported; no post hoc winning variant is selected."))
        evaluate(name, _dataset_copy(dataset, frame), variant, "model_settings")
    for name, labels, rule, source_ok in contexts:
        if not rule.get("evaluate", False) or not source_ok:
            inventory.append(dict(name="outcome_" + name, family="alternative_outcome", status="not_run",
                reason="Definition audit only: evaluate=true and a stated reference are both required."))
            continue
        copied = frame.copy(deep=True)
        copied["Halitosis"] = labels
        variant = _merge(config, rule.get("model_overrides", {}))
        variant.setdefault("modeling", {})["tasks"] = ["classification"]
        variant.setdefault("sensitivity", {})["active_outcome_definition"] = name
        if np.unique(labels).size < 2:
            inventory.append(dict(name="outcome_" + name, family="alternative_outcome", status="not_run", reason="Alternative definition yields only one class."))
            continue
        evaluate("outcome_" + name, _dataset_copy(dataset, copied), variant, "alternative_outcome")
    pd.DataFrame(registry, columns=["name", "config", "data_transform", "selection"]).to_csv(out_dir / "sensitivity_model_variants.csv", index=False)
    pd.DataFrame(inventory).to_csv(out_dir / "sensitivity_model_execution.csv", index=False)
    (pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(columns=["sensitivity_variant", "sensitivity_family", "task"])).to_csv(out_dir / "sensitivity_model_summary.csv", index=False)
    outcome_info = _outcome_effects(dataset,config,definitions,contexts,summaries,out_dir)
    repeat_info = _repeats(dataset, config, out_dir)
    metadata = config.get("sensitivity", {}).get("metadata_file") or config.get("statistics", {}).get("metadata_file")
    available = set(frame.columns)
    if metadata:
        metadata_path = Path(metadata)
        if not metadata_path.is_absolute():
            metadata_path = Path(__file__).resolve().parents[1] / metadata_path
        available.update(pd.read_csv(metadata_path, nrows=0).columns)
    pd.DataFrame([dict(measurement=field, supplied=field in available,
        status="available_for_design_review" if field in available else "not_supplied_cannot_reconstruct")
        for field in ["DMS_ppb", "organoleptic_score", "tongue_coating", "periodontal_status", "PD", "BOP", "salivary_flow"]]).to_csv(out_dir / "measurement_inventory.csv", index=False)
    if not dataset.frame.Halitosis.equals(original):
        raise AssertionError("Original outcome was overwritten by a sensitivity analysis.")
    result = dict(status="complete", n=len(frame), original_class_counts={str(g): int((original == g).sum()) for g in [0, 1]},
        original_labels_preserved=True, independent_total_used=False,
        alternative_definitions=len(definitions), prespecified_model_variants=len(variants),
        model_runner_supplied=runner is not None, model_runs_completed=sum(r["status"] == "completed" for r in inventory),
        model_runs_failed=sum(r["status"] == "failed" for r in inventory),
        selection="No winner selected across variants; summaries retain all prespecified model rows.", **repeat_info, **outcome_info)
    (out_dir / "sensitivity_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return result
