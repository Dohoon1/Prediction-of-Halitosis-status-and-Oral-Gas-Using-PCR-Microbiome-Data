# PCR major-revision analysis code (2026-10-21)

## Process

# 0. requirements
python -m pip install -r requirements-neural.txt

-------------------------------------------------------------
# Main analysis
python run.py check --output results/input_check

python run.py test

python run.py statistics --output results/statistics

python code/verify_reference.py --results results/statistics --report results/statistics/reference_check.json

python run.py figures --output results/final_figures

python run.py smoke --neural --output results/development_check

```text
python run.py linear --output results/linear_main
python run.py transformer --output results/transformer_main
python run.py sensitivity --output results/sensitivity_main
```
