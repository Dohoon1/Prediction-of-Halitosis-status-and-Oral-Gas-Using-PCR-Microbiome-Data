"""Compare a new descriptive run with independent, frozen R2 checkpoints."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from common import ROOT, write_json


def verify(results, reference):
    current = pd.read_csv(results / "species_comparisons.csv")
    frozen = pd.read_csv(reference / "verified_species_tests.csv")
    joined = current.merge(
        frozen,
        left_on=["representation", "taxon"],
        right_on=["scale", "column"],
        validate="one_to_one",
    )
    if len(joined) != 64:
        raise ValueError("Expected all 16 taxa across all four scales.")
    columns = {
        "mean0": "mean0",
        "mean1": "mean1",
        "mean_difference": "difference",
        "p_welch": "welch_p",
        "p_mannwhitney": "mw_p",
        "welch_bonferroni_16": "welch_bonferroni",
        "welch_bh_16": "welch_bh",
        "mannwhitney_bonferroni_16": "mw_bonferroni",
        "mannwhitney_bh_16": "mw_bh",
    }
    deviations = {}
    for left, right in columns.items():
        a = joined[left + "_x" if left == right else left].to_numpy()
        b = joined[right + "_y" if left == right else right].to_numpy()
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
        deviations[left] = float(np.max(np.abs(a - b)))
    diversity = pd.read_csv(results / "PCR_panel_diversity_summary.csv")
    old = pd.read_csv(reference / "verified_panel_diversity.csv")
    mapping = {
        "Richness": "PCR_detected_targets",
        "Shannon": "PCR_panel_Shannon_ln",
        "Evenness": "PCR_panel_Pielou_evenness",
    }
    # Names are explicit; row ordering is never used to align scientific values.
    if set(mapping.values()) != set(diversity.metric):
        raise ValueError("Unexpected diversity metric names.")
    old["metric"] = old["index"].map(mapping)
    merged = diversity.merge(old, on="metric", validate="one_to_one")
    for a, b in [
        ("nonhalitosis_mean", "mean0"),
        ("nonhalitosis_sd", "sd0"),
        ("halitosis_mean", "mean1"),
        ("halitosis_sd", "sd1"),
        ("welch_p_two_sided", "welch_p"),
        ("mannwhitney_p_two_sided", "mw_p"),
    ]:
        np.testing.assert_allclose(merged[a], merged[b], rtol=1e-10, atol=1e-12)
    associations = pd.read_csv(results / "adjusted_associations.csv")
    old_associations = pd.read_csv(reference / "adjusted_associations.csv")
    pair = associations.merge(
        old_associations,
        on=["taxon", "target"],
        suffixes=("_new", "_reference"),
        validate="one_to_one",
    )
    if len(pair) != 12 or not (associations.status == "ok").all():
        raise ValueError("All 12 adjusted associations must be available and fitted.")
    for metric in ["coefficient", "ci_low", "ci_high", "p", "p_bh_within_target"]:
        np.testing.assert_allclose(
            pair[metric + "_new"], pair[metric + "_reference"], rtol=1e-8, atol=1e-10
        )
    return {
        "status": "passed",
        "species_scale_rows": 64,
        "species_numeric_comparisons": 64 * len(columns),
        "diversity_indices": 3,
        "adjusted_associations": 12,
        "maximum_species_absolute_deviations": deviations,
        "scope": "Data-derived descriptive values and adjusted associations. Reported predictive summaries are not independently refitted by this check.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "reference")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = verify(args.results, args.reference)
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
