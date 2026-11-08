# Source and changes

Upstream repository: <https://github.com/Dohoon1/Prediction-of-Halitosis-status-and-Oral-Gas-Using-PCR-Microbiome-Data>

Source commit: `51a1e0ad7f8506cd2e537644bf8ae6535a2d5b7c` (22 October 2026). The source checkout was retained. No remote push was performed. Upstream module hashes and the public workbook hash are recorded in `reference/provenance.json`.

## Consolidated modules

| Upstream | Release | Purpose |
| --- | --- | --- |
| `code/01_analysis.py` | `code/analysis.py` | Staged input audit, descriptive analysis, fitting and reporting |
| `code/common.py` | `code/common.py` | Original-label lock, 16-target whitelist, explicit parsing |
| `code/modeling.py` | `code/modeling.py` | Repeated nested CV, train-fitted preprocessing and selection |
| `code/neural.py` | `code/neural.py` | Local feature Transformer and deterministic training |
| `code/pcr_statistics.py` | `code/pcr_statistics.py` | Species tests, diversity, robustness, adjusted associations |
| `code/sensitivity.py` | `code/sensitivity.py` | Prespecified model variants and separate outcome definitions |
| `code/validation.py` | `code/validation.py` | Participant-block CIs, paired inference, full-pipeline permutations |
| `code/reporting.py` | `code/reporting.py` | CSV and reporting manifests for new runs |
| `code/02_figures.py` | `code/diagnostic_figures.py` | Optional generic plots from saved new runs; not the final manuscript renderer |
| `code/03_permutation_figures.py` | `code/permutation_figures.py` | Optional plots of completed permutation tests |
| Final revision figure scripts | `code/final_figures.py`, `code/workflow_icons.py` | Final R2 design and frozen reported-performance panels |
| Added | `run.py`, `config/`, `code/verify_reference.py`, `tests/` | Portable entry point, complete profiles and checks |

The independent scripts under upstream `Final_Prediction_Codes/Halitosis_Prediction/` and `Oral_Gas_Prediction/` are historical prototypes. Their fitting, ROC, gas summary, cluster and importance functions are covered by the modules above. They used older settings and are not alternate R2 entry points. Duplicate workbooks, Windows-specific launchers, interpreter caches and document-editing utilities are excluded from this release. The original scripts remain available at the pinned source commit.

## Functional corrections in this release

- Restored explicit profiles consistent with the final R2 Methods. The source checkout lacked its referenced configuration directory.
- Restricted the Linear family to the single inner-CV species selector, excluding the extra all-candidate selector. Final totals are 22 Linear and 19 Transformer per outcome.
- Changed the cluster-score expression to the stated `mean(log10(x+1))`. The source used natural log before z-score standardization. Positive constant log-base scaling cancels in the standardized score; the explicit expression now matches the manuscript.
- Kept stable panel names for fixed-cohort and train-derived cluster comparisons. Fixed-cohort clusters remain explicitly exploratory and are rejected for full-pipeline outcome permutation inference unless independently prespecified.
- Corrected the input-audit explanation: the six VSC-only/primary-label differences reflect distinct outcome definitions. Original labels are retained and their participant mapping is hash-locked.
- Made the final figure renderer import-safe and portable. It reads frozen JSON with inventory checks, replacing stale positional indices into an older supplementary Word document.
- Marked development profiles explicitly, separated their outputs, and prevented accidental overwriting. Formatting was standardized across Python files.

## Provenance limits

This is a documented rerun implementation built from the public engine and final R2 protocol, not a recovered archive of the original execution environment. Original seed-specific Transformer OOF predictions, the historical configuration files and original model checkpoints are not deposited. Rerun seeds 11, 42 and 2026 follow public source defaults and are not asserted to be the original three seeds. New cross-seed summaries use sample SD (`ddof=1`); the rounding in frozen reported summaries is retained without reverse engineering seed values.

All manuscript performance tables and their reported CIs/seed SDs remain reference data. New runs save actual participant-level OOF predictions and configuration/code/data hashes. No new predictive OOF data or model checkpoints were invented from summary tables. The validation report states which stages were actually executed.

No license grant was added on behalf of the authors. Attribution and source provenance are retained; the authors can choose a repository license separately.
