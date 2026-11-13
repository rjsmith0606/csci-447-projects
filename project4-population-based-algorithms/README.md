# Project 3 - Neural Networks (CSCI 447)

Logistic/linear regression, a two-hidden-layer feedforward network trained with
backpropagation, and a feedforward network whose first hidden layer is
pre-trained as an autoencoder. Evaluated with 5x2 cross-validation.

Assigned: October 19, 2026 -- Due: November 9, 2026

## Layout

- `data/raw`, `data/names` - the six UCI data sets (copied from project 2)
- `data/processed` - normalized / encoded data for the networks
- `src/` - preprocessing, activations, losses, networks, autoencoder, evaluation, tuning
- `notebooks/` - numbered notebooks, run in order
- `results/` - figures, per-fold raw scores, tuning grids, saved models
- `report/` - JMLR-format paper (`P3_Paper.tex`, `references.bib`). Needs `jmlr2e.sty`
  (available in Overleaf or at http://www.jmlr.org/format/format.html)

## Setup

```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
