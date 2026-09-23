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
