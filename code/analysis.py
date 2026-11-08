"""Run the PCR study explicitly; figures are rendered by diagnostic_figures.py.

Use --preflight for an input audit without fitting models. No scientific result
in the manuscript is hard-coded as an expected output of this implementation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import pandas as pd

from common import (
    ROOT,
    GAS_COLUMNS,
    Dataset,
    correlation_components,
    file_sha256,
    load_dataset,
    read_config,
    write_json,
)


def _summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan, q25=np.nan, q75=np.nan)
    return dict(
        n=len(values),
        mean=float(values.mean()),
        sd=float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        median=float(np.median(values)),
        q25=float(np.percentile(values, 25)),
        q75=float(np.percentile(values, 75)),
    )


def descriptive_analysis(dataset: Dataset, config: dict, out_dir: Path) -> dict:
    """Calculate all values used by figures, including descriptive tests."""
    from scipy import stats

    frame, pcr = dataset.frame, dataset.pcr_columns
    numeric = [c for c in ["Age", *GAS_COLUMNS, "PCR_panel_total"] if c in frame]
    clinical_rows, category_rows, taxon_rows = [], [], []
    for variable in numeric:
        for group in (0, 1):
            clinical_rows.append(
                dict(
                    variable=variable,
                    group=group,
                    **_summary(frame.loc[frame.Halitosis == group, variable]),
                )
            )
    for variable in [c for c in dataset.clinical_columns if c != "Age"]:
        table = pd.crosstab(frame[variable], frame.Halitosis).reindex(columns=[0, 1], fill_value=0)
        if table.shape == (2, 2):
            p = float(stats.fisher_exact(table.to_numpy()).pvalue)
            method = "Fisher_exact"
        elif table.shape[0] >= 2 and (table.sum(axis=0) > 0).all():
            p = float(stats.chi2_contingency(table.to_numpy()).pvalue)
            method = "Pearson_chi_squared"
        else:
            p, method = 1.0, "constant_category_no_comparison"
        for category in table.index:
            for group in (0, 1):
                category_rows.append(
                    dict(
                        variable=variable,
                        category=category,
                        group=group,
                        n=int(table.loc[category, group]),
                        p=p,
                        method=method,
                    )
                )
    long_rows = []
    for taxon in pcr:
        a = frame.loc[frame.Halitosis == 0, taxon].dropna().to_numpy(dtype=float)
        b = frame.loc[frame.Halitosis == 1, taxon].dropna().to_numpy(dtype=float)
        if min(len(a), len(b)) < 2:
            p = np.nan
        elif np.var(a) == 0 and np.var(b) == 0:
            p = 1.0 if a[0] == b[0] else 0.0
        else:
            p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        for group in (0, 1):
            values = frame.loc[frame.Halitosis == group, taxon].dropna().to_numpy(dtype=float)
            display = np.log10(values + 1)
            summary = _summary(values)
            q25, median, q75 = (
                np.percentile(display, [25, 50, 75]) if len(display) else [np.nan] * 3
            )
            inside = display[
                (display >= q25 - 1.5 * (q75 - q25)) & (display <= q75 + 1.5 * (q75 - q25))
            ]
            taxon_rows.append(
                dict(
                    taxon=taxon,
                    group=group,
                    **summary,
                    prevalence=float((values > 0).mean()) if len(values) else np.nan,
                    p=p,
                    q_bonferroni=min(p * len(pcr), 1) if np.isfinite(p) else np.nan,
                    method="Welch_t_raw_copies",
                    plot_q25=q25,
                    plot_median=median,
                    plot_q75=q75,
                    plot_whisker_low=float(inside.min()) if len(inside) else np.nan,
                    plot_whisker_high=float(inside.max()) if len(inside) else np.nan,
                )
            )
            subset = frame.loc[frame.Halitosis == group, ["participant_id", taxon]].dropna()
            long_rows.extend(
                dict(
                    participant_id=row.participant_id,
                    group=group,
                    taxon=taxon,
                    log10_copies_plus1=float(np.log10(getattr(row, taxon) + 1)),
                )
                for row in subset.itertuples(index=False)
            )
    pd.DataFrame(clinical_rows).to_csv(out_dir / "clinical_summary.csv", index=False)
    pd.DataFrame(category_rows).to_csv(out_dir / "clinical_categories.csv", index=False)
    pd.DataFrame(taxon_rows).to_csv(out_dir / "taxon_summary.csv", index=False)
    pd.DataFrame(long_rows).to_csv(out_dir / "abundance_long.csv", index=False)
    corr = frame[pcr].corr(method="pearson")
    corr.to_csv(out_dir / "correlation.csv", index_label="taxon_index")
    settings = config["modeling"]
    clusters, edges = correlation_components(
        corr,
        settings["correlation_threshold"],
        settings["absolute_correlation"],
        settings.get("correlation_inclusive", True),
    )
    pd.DataFrame(edges, columns=["source", "target", "r", "cluster"]).to_csv(
        out_dir / "correlation_edges.csv", index=False
    )
    write_json(out_dir / "clusters.json", clusters)
    pd.DataFrame(
        [
            {
                "stage_id": "input",
                "parent_stage_id": "",
                "stage": "Input participants",
                "n": dataset.audit["input_rows"],
            },
            {
                "stage_id": "analytic",
                "parent_stage_id": "input",
                "stage": "PCR analytic cohort",
                "n": len(frame),
            },
            {
                "stage_id": "nonhal",
                "parent_stage_id": "analytic",
                "stage": "Non-halitosis",
                "n": int((frame.Halitosis == 0).sum()),
            },
            {
                "stage_id": "hal",
                "parent_stage_id": "analytic",
                "stage": "Halitosis",
                "n": int((frame.Halitosis == 1).sum()),
            },
        ]
    ).to_csv(out_dir / "sample_flow.csv", index=False)
    from pcr_statistics import run_statistics

    revision_statistics = run_statistics(dataset, config, out_dir)
    return {
        "descriptive_clusters": {name: len(values) for name, values in clusters.items()},
        "descriptive_graph_scope": "full cohort, descriptive only; predictive nested membership is train-derived separately",
        "revision_statistics": revision_statistics,
        "diversity": "16-target PCR detected-target count, Shannon natural log, Pielou. Chao1 excluded.",
    }


def _versions():
    versions = {"python": platform.python_version()}
    for package in [
        "numpy",
        "pandas",
        "openpyxl",
        "scipy",
        "scikit-learn",
        "imbalanced-learn",
        "torch",
        "matplotlib",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "revision_main.json")
    parser.add_argument(
        "--input", type=Path, help="Original workbook/CSV; no source file is modified."
    )
    parser.add_argument(
        "--results", type=Path, help="New empty result directory; existing outputs are never mixed."
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate the input and write an audit without fitting.",
    )
    parser.add_argument(
        "--descriptive-only",
        action="store_true",
        help="Calculate descriptive outputs without model fitting.",
    )
    parser.add_argument(
        "--stage",
        choices=["preflight", "descriptive", "analysis", "sensitivity", "permutation"],
        default="analysis",
    )
    parser.add_argument(
        "--sensitivity-models",
        action="store_true",
        help="Fit all explicitly listed sensitivity variants; expensive.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume a matching permutation checkpoint only."
    )
    parser.add_argument(
        "--permutation-batch-size",
        type=int,
        help="Maximum new permutations this invocation; does not change final requested B.",
    )
    args = parser.parse_args(argv)
    config = read_config(args.config)
    profile = config.get("profile", args.config.stem)
    stage = (
        "preflight" if args.preflight else "descriptive" if args.descriptive_only else args.stage
    )
    suffix = profile + "_" + stage
    output = (args.results or ROOT / "results" / suffix).resolve()
    if args.resume and stage != "permutation":
        parser.error("--resume is supported only for --stage permutation.")
    if args.permutation_batch_size is not None and args.permutation_batch_size < 1:
        parser.error("--permutation-batch-size must be positive.")
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error(f"Result directory is not empty: {output}. Choose a new --results directory.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "profile": profile,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "versions": _versions(),
        "evaluation_scope": config["modeling"].get("panels", "fixed_exploratory"),
        "mode": stage,
        "code_sha256": {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "publication_run": not config.get("development_run", False),
        "permutation_executed": False,
    }
    write_json(output / "manifest.json", manifest)
    try:
        dataset = load_dataset(config, args.input)
        write_json(output / "data_validation.json", dataset.audit)
        manifest.update(
            n=len(dataset.frame),
            input_sha256=file_sha256(dataset.source_path),
            input_path=str(dataset.source_path),
            class_counts=dataset.audit["class_counts"],
        )
        if stage not in {"preflight", "permutation"}:
            manifest.update(descriptive_analysis(dataset, config, output))
            from sensitivity import run_sensitivity

            if stage == "analysis":
                from modeling import run_modeling

                manifest["modeling_results"] = run_modeling(dataset, config, output)
                from validation import finalize_results

                manifest["validation_results"] = finalize_results(output, config)

            def sensitivity_runner(variant_dataset, variant_config, variant_dir):
                from modeling import run_modeling
                from validation import finalize_results

                result = run_modeling(variant_dataset, variant_config, variant_dir)
                finalize_results(variant_dir, variant_config)
                return result

            manifest["sensitivity_results"] = run_sensitivity(
                dataset,
                config,
                output,
                runner=(
                    sensitivity_runner
                    if stage == "sensitivity" and args.sensitivity_models
                    else None
                ),
            )
        if stage == "permutation":
            from validation import run_permutation

            manifest["permutation_results"] = run_permutation(
                dataset, config, output, max_new_permutations=args.permutation_batch_size
            )
            manifest["permutation_executed"] = True
        from reporting import write_reporting

        manifest["reporting"] = write_reporting(output, dataset, config, stage)
        pending = (
            stage == "permutation"
            and manifest.get("permutation_results", {}).get("status") != "complete"
        )
        manifest.update(
            status="incomplete" if pending else "complete",
            completed_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(output / "manifest.json", manifest)
        if pending:
            print(
                "Permutation checkpoint saved; requested permutations remain incomplete. Final p values are not available."
            )
        else:
            print(
                f"Completed {manifest['mode']}: n={len(dataset.frame)}, PCR targets={len(dataset.pcr_columns)}"
            )
        print(f"Results: {output}")
        for warning in dataset.audit["warnings"]:
            print(f"Input audit: {warning}")
        return 0
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "manifest.json", manifest)
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Analysis failed: {error}\nDetails: {output / 'failure.txt'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
