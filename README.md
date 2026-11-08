# PCR major-revision analysis code (2026-10-21)

## Process

python -m pip install -r requirements.txt

# 1.
python code/01_analysis.py --stage preflight --results results/preflight_01

# 2. Main Analysis
python code/01_analysis.py --stage analysis --results results/main_01

# 3. Sensitivity Analysis
python code/01_analysis.py --stage sensitivity --sensitivity-models --results results/sensitivity_01
-------------------------------------------------------------
# 4. Permutation validation
python code/01_analysis.py --config config/revision_permutation.json --stage permutation --results results/permutation_01

Linear Model
!python -u permutation_split/code/01_analysis.py --config permutation_split/config/revision_permutation.json --input data/PCR_NGS_Data.xlsx --stage permutation --model-family linear --results runs/revision_01/permutation_linear

Transformer-based Model
!python -u permutation_split/code/01_analysis.py --config permutation_split/config/revision_permutation.json --input data/PCR_NGS_Data.xlsx --stage permutation --model-family transformer --results runs/revision_01/permutation_transformer

Merge results
!python -u permutation_split/code/05_merge_permutation_results.py --linear-results runs/revision_01/permutation_linear --transformer-results runs/revision_01/permutation_transformer --results runs/revision_01/permutation_combined
-------------------------------------------------------------
# 5. Figures
python code/02_figures.py --results results/main_01 --output figures/main_01
python code/03_permutation_figures.py --results results/permutation_01 --output figures/permutation_01
