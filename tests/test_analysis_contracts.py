"""Scientific data and validation contracts, independent of fitted rankings."""

from pathlib import Path
import copy
import importlib.util
import json
import tempfile
import unittest
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import load_dataset, read_config
from modeling import FoldPreprocessor, discover_panels
from pcr_statistics import panel_indices
from sensitivity import _outcomes
from validation import _load_oof, _regression_comparisons


class AnalysisContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_config(ROOT / "config/revision_main.json")
        cls.dataset = load_dataset(cls.config)

    def test_original_cohort_and_vsc_only_are_distinct(self):
        data = self.dataset
        self.assertEqual(data.audit["class_counts"], {"0": 72, "1": 30})
        self.assertEqual(data.audit["sex_threshold_label_mismatches"], 6)
        self.assertEqual(data.audit["vsc_sum_mismatches"], 0)
        self.assertEqual(data.audit["missing_by_column"], {})
        self.assertEqual(len(data.pcr_columns), 16)
        self.assertNotIn("Total_bacteria_copies_mL", data.frame)
        with tempfile.TemporaryDirectory() as tmp:
            before = data.frame.Halitosis.copy(deep=True)
            definitions, _ = _outcomes(data, self.config, Path(tmp))
            counts = {
                r["name"]: r["positive_n"] for r in definitions if r.get("positive_n") is not None
            }
            self.assertEqual(counts["supplied_sex_cutoff_audit"], 36)
            self.assertEqual(counts["sex_upper_quartile_audit"], 26)
            pd.testing.assert_series_equal(before, data.frame.Halitosis)

    def test_label_mapping_lock_catches_swaps_even_with_same_counts(self):
        frame = self.dataset.frame.copy(deep=True)
        first = frame.index[frame.Halitosis.eq(0)][0]
        second = frame.index[frame.Halitosis.eq(1)][0]
        frame.loc[[first, second], "Halitosis"] = [1, 0]
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "swapped.csv"
            frame.drop(columns=["PCR_panel_total"]).to_csv(file, index=False)
            with self.assertRaisesRegex(ValueError, "participant-label mapping changed"):
                load_dataset(self.config, file)

    def test_final_model_family_matches_multiplicity_scope(self):
        panels = discover_panels(self.dataset.frame, self.dataset.pcr_columns, self.config)
        self.assertEqual(len(panels), 20)
        self.assertNotIn("All_PCR", panels)
        self.assertNotIn("Cluster1_PC1", panels)
        selectors = self.config["modeling"]["selector_names"]
        self.assertEqual(selectors, ["Nested_best_single"])
        non_dummy = len(panels) + len(selectors)
        self.assertEqual(non_dummy, 21)
        self.assertEqual(3 * non_dummy, 63)
        self.assertEqual((non_dummy + 1) * non_dummy // 2, 231)
        nn = read_config(ROOT / "config/revision_transformer_42.json")
        self.assertEqual(len(discover_panels(self.dataset.frame, self.dataset.pcr_columns, nn)), 19)
        self.assertFalse(nn["modeling"]["include_dummy"])
        self.assertFalse(nn["modeling"]["include_selectors"])

    def test_heldout_values_do_not_change_fitted_scaler_or_clusters(self):
        data = self.dataset
        train = data.frame.iloc[:80].copy()
        test = data.frame.iloc[80:].copy()
        config = copy.deepcopy(self.config)
        config["modeling"]["sampling"] = "none"
        processor = FoldPreprocessor(
            data.pcr_columns, data.clinical_columns, {"strategy": "largest"}, config
        )
        processor.fit_resample(train, train.Halitosis.to_numpy(), "classification", 42)
        selected = processor.selected_pcr_.copy()
        mean = processor.scaler_.mean_.copy()
        transformed = processor.transform(test)
        test.loc[:, data.pcr_columns] = 1e15
        test.loc[:, "Age"] = 119
        extreme = processor.transform(test)
        self.assertTrue(np.isfinite(extreme).all())
        self.assertEqual(len(transformed), 22)
        self.assertEqual(processor.selected_pcr_, selected)
        np.testing.assert_array_equal(processor.scaler_.mean_, mean)
        self.assertFalse(set(processor.fit_ids_) & set(test.participant_id.astype(str)))
        numeric_mean = (
            processor._features(train, False)[processor.numeric_].to_numpy(dtype=float).mean(axis=0)
        )
        np.testing.assert_allclose(mean, numeric_mean)

    def test_cluster_log_score_matches_manuscript_definition(self):
        data = self.dataset
        config = copy.deepcopy(self.config)
        config["modeling"]["sampling"] = "none"
        processor = FoldPreprocessor(
            data.pcr_columns,
            data.clinical_columns,
            {"columns": data.pcr_columns[:3], "reducer": "logmean"},
            config,
        )
        train = data.frame.iloc[:80]
        processor.fit_resample(train, train.Halitosis.to_numpy(), "classification", 42)
        observed = processor._features(data.frame, False).PCR_cluster_score.to_numpy()
        expected = np.log10(data.frame[data.pcr_columns[:3]].to_numpy() + 1).mean(axis=1)
        np.testing.assert_allclose(observed, expected)

    def test_fixed_cluster_variant_uses_matching_panel_identifiers(self):
        config = copy.deepcopy(self.config)
        config["modeling"]["panels"] = "fixed_exploratory"
        panels = discover_panels(self.dataset.frame, self.dataset.pcr_columns, config)
        self.assertEqual(set(panels), set(self.config["modeling"]["panel_names"]))
        self.assertEqual(len(panels["Cluster_Largest"]["columns"]), 13)

    def test_panel_diversity_uses_target_counts_and_natural_log(self):
        values = np.zeros((2, 16))
        values[0, :2] = 1
        values[1, 0] = 2
        richness, shannon, evenness = panel_indices(values)
        np.testing.assert_array_equal(richness, [2, 1])
        np.testing.assert_allclose(shannon, [np.log(2), 0])
        self.assertAlmostEqual(evenness[0], 1)

    def test_duplicate_oof_rows_are_rejected(self):
        row = dict(
            target="Halitosis",
            model="Clinical_only",
            algorithm="logistic_sklearn",
            participant_id="a",
            repeat=1,
            fold=1,
            y_true=1,
            y_prob=0.6,
        )
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame([row, row]).to_csv(Path(tmp) / "classification_oof.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate participant OOF"):
                _load_oof(tmp, "classification", 1)

    def test_error_comparison_averages_errors_before_repeats(self):
        rows = []
        for participant, truth in enumerate([10.0, 20.0, 30.0, 40.0]):
            for repeat in [1, 2]:
                for model, error in [
                    ("oscillating", 2 if repeat == 1 else -2),
                    ("small", 1),
                    ("large", 4),
                ]:
                    rows.append(
                        dict(
                            target="H2S_ppb",
                            target_scale="raw",
                            participant_id=str(participant),
                            repeat=repeat,
                            model=model,
                            algorithm="ridge",
                            y_true=truth,
                            y_pred=truth + error,
                        )
                    )
        with tempfile.TemporaryDirectory() as tmp:
            _regression_comparisons(pd.DataFrame(rows), tmp)
            pairs = pd.read_csv(Path(tmp) / "regression_pairwise.csv")
            pair = pairs[(pairs.model_a == "oscillating") & (pairs.model_b == "small")].iloc[0]
            self.assertAlmostEqual(pair.mean_absolute_error_difference, 1.0)
            self.assertTrue((pairs.family_comparisons == 3).all())


if __name__ == "__main__":
    unittest.main()
