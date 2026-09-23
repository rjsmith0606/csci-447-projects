#!/bin/bash
# Run this from inside your cloned repo root (where .git/ lives)

# Data directories (same 6 datasets as Project 1)
mkdir -p data/raw/breast_cancer
mkdir -p data/raw/car_evaluation
mkdir -p data/raw/congressional_vote
mkdir -p data/raw/abalone
mkdir -p data/raw/computer_hardware
mkdir -p data/raw/forest_fires
mkdir -p data/processed

# Source code
mkdir -p src
touch src/__init__.py
touch src/preprocessing.py       # reuse/adapt from Project 1 -- no one-hot needed here!
touch src/tree_node.py           # shared node class for both tree types
touch src/classification_tree.py # gain ratio splitting
touch src/regression_tree.py     # MSE splitting
touch src/pruning.py             # reduced-error pruning (shared logic, both tree types)
touch src/evaluation.py          # 5x2 CV loop, error/MSE metrics, stats tests
touch src/splitting.py           # entropy, gain ratio, IV, MSE-split helper functions

# Notebooks
mkdir -p notebooks
touch notebooks/01_explore_data.ipynb
touch notebooks/02_preprocessing_tests.ipynb
touch notebooks/03_tree_sanity_checks.ipynb   # small synthetic data, verify splits make sense
touch notebooks/04_run_experiments.ipynb      # full 5x2 CV, pruned vs unpruned
touch notebooks/05_results_analysis.ipynb

# Results
mkdir -p results/raw_scores
mkdir -p results/figures
mkdir -p results/tree_diagrams   # optional: visualize a few learned trees for the report

# Report
mkdir -p report/figures
touch report/report.tex

# Keep empty dirs tracked by git
find data results -type d -empty -exec touch {}/.gitkeep \;

# README
cat > README.md << 'README'
# Decision Tree Project (CSCI 447 - Project 2)

## Team
- Partner A: [name]
- Partner B: [name]

## Structure
- `data/raw/` — original UCI datasets (same 6 as Project 1)
- `data/processed/` — cleaned datasets (NOTE: no one-hot encoding needed —
  trees split natively on categorical values)
- `src/` — all reusable code
  - `preprocessing.py` — missing values, ID column removal, keep numeric
    vs categorical columns clearly labeled (not one-hot encoded)
  - `splitting.py` — entropy, gain ratio, information value, MSE-split criteria
  - `classification_tree.py` — tree built with gain ratio splits
  - `regression_tree.py` — tree built with MSE splits
  - `pruning.py` — reduced-error (post-)pruning for both tree types
  - `evaluation.py` — 5x2 CV, held-out pruning split logic, metrics
- `notebooks/` — driver notebooks, import from src/
- `results/` — saved CV scores, figures, optional tree diagrams
- `report/` — JMLR-format LaTeX report

## Task split
- Partner A: preprocessing.py, splitting.py, classification_tree.py
- Partner B: regression_tree.py, pruning.py, evaluation.py

## Key experimental design note
20% of each dataset is held out ONLY for pruning evaluation (not part of
cross-validation). The remaining 80% is used for 5x2 CV: train tree fully
on the training fold, prune using the fixed held-out 20%, then test on
the CV test fold.

## Setup (Colab / local)
\`\`\`bash
git clone git@github.com:yourusername/dt-project.git
cd dt-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`
README

# .gitignore
cat > .gitignore << 'GITIGNORE'
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/*.zip
.DS_Store
*.pkl
venv/
GITIGNORE

echo "Structure created."
