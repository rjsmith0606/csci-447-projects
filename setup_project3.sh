#!/usr/bin/env bash
# =============================================================================
# setup_project3.sh
#
# Creates the project3-neural-networks folder for CSCI 447 Project 3
# (logistic/linear regression, backprop feedforward nets, autoencoder nets),
# mirroring the layout used in project2-decision-trees:
#
#   project3-neural-networks/
#   ├── .gitignore
#   ├── requirements.txt
#   ├── data/        raw/  names/  processed/
#   ├── notebooks/   numbered 01..05, same flow as project 2
#   ├── report/      P3_Paper.tex, references.bib, figures/
#   ├── results/     figures/  raw_scores/  tuning/  models/
#   └── src/         preprocessing, networks, autoencoder, evaluation, tuning
#
# Usage (run from the repo root, next to project1-KNN / project2-decision-trees):
#   bash setup_project3.sh
#
# Safe to re-run: existing files are never overwritten.
# =============================================================================
set -euo pipefail

PROJECT_DIR="project3-neural-networks"
PREV_PROJECT="project2-decision-trees"

# write FILE  (content on stdin) -- skips if FILE already exists
write() {
    if [ -e "$1" ]; then
        echo "  skip    $1 (already exists)"
        cat > /dev/null
    else
        cat > "$1"
        echo "  create  $1"
    fi
}

# notebook FILE "Title" "Description" -- writes a minimal valid .ipynb
notebook() {
    local file="$1" title="$2" desc="$3"
    write "$file" <<EOF
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# $title\n",
    "\n",
    "$desc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append(\"..\")  # so \`from src...\` works from the notebooks folder\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
EOF
}

# --- sanity check: don't create the project inside another project -----------
if [ ! -d "$PREV_PROJECT" ] && [ ! -d "project1-KNN" ]; then
    echo "Note: $PREV_PROJECT not found in $(pwd)."
    echo "      Run this from the repo root to have the raw data copied over."
    echo "      Continuing anyway (data/raw will be left empty)."
    echo
fi

echo "Creating directories..."
mkdir -p "$PROJECT_DIR"/data/{raw,names,processed}
mkdir -p "$PROJECT_DIR"/notebooks
mkdir -p "$PROJECT_DIR"/report/figures
mkdir -p "$PROJECT_DIR"/results/{figures,raw_scores,tuning,models}
mkdir -p "$PROJECT_DIR"/src

# keep empty folders in git
for d in data/processed results/figures results/raw_scores results/tuning results/models report/figures; do
    touch "$PROJECT_DIR/$d/.gitkeep"
done

# --- copy the six UCI data sets from project 2 -------------------------------
echo
echo "Copying raw data and names files from $PREV_PROJECT..."
if [ -d "$PREV_PROJECT/data/raw" ]; then
    cp -n "$PREV_PROJECT"/data/raw/*   "$PROJECT_DIR/data/raw/"   2>/dev/null || true
    cp -n "$PREV_PROJECT"/data/names/* "$PROJECT_DIR/data/names/" 2>/dev/null || true
    echo "  copied  $(ls "$PROJECT_DIR/data/raw" | wc -l | tr -d ' ') raw files, $(ls "$PROJECT_DIR/data/names" | wc -l | tr -d ' ') names files"
else
    echo "  $PREV_PROJECT/data/raw not found -- download the six data sets from"
    echo "  https://archive.ics.uci.edu/ml/ into $PROJECT_DIR/data/raw/ manually."
fi
# NOTE: data/processed is intentionally NOT copied. Neural nets need different
# preprocessing than trees (normalization, one-hot encoding), so regenerate it.

# --- top-level files ---------------------------------------------------------
echo
echo "Writing project files..."

write "$PROJECT_DIR/.gitignore" <<'EOF'
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/*.zip
.DS_Store
*.pkl
*.npz
venv/

# LaTeX build artifacts
report/*.aux
report/*.log
report/*.out
report/*.bbl
report/*.blg
report/*.synctex.gz
EOF

write "$PROJECT_DIR/requirements.txt" <<'EOF'
numpy
pandas
matplotlib
scipy
jupyter
EOF

# --- src ---------------------------------------------------------------------
write "$PROJECT_DIR/src/__init__.py" <<'EOF'
EOF

write "$PROJECT_DIR/src/preprocessing.py" <<'EOF'
"""
Preprocessing for the neural network project.

Key differences from the decision tree preprocessing (project 2):
- Normalization IS required: gradient descent is sensitive to feature scale.
  Fit the scaler on the TRAINING fold only and apply it to the test fold, to
  avoid leakage.
- Categorical features are one-hot encoded (or are already Boolean).
- Classification labels are one-hot encoded to match the softmax output layer.
- Regression targets may need a transform (e.g. ln(x + 1) for Forest Fires).

Each preprocess_* function should return (X, y) as float numpy arrays plus any
metadata needed by the networks (e.g. number of classes).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame, columns: list[str], strategy: str = "drop_rows",
                   missing_values: list | None = None) -> pd.DataFrame:
    """Replace missing-value markers (e.g. '?') and drop / impute rows."""
    raise NotImplementedError


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot encode categorical feature columns."""
    raise NotImplementedError


def one_hot_labels(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Convert integer class labels to a one-hot matrix for softmax / cross-entropy."""
    raise NotImplementedError


def fit_scaler(X_train: np.ndarray) -> dict:
    """Compute normalization stats (min/max or mean/std) from training data only."""
    raise NotImplementedError


def apply_scaler(X: np.ndarray, scaler: dict) -> np.ndarray:
    """Apply stats from fit_scaler to any split."""
    raise NotImplementedError


def log_transform(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """ln(x + 1) transform, used for the skewed Forest Fires burned-area target."""
    raise NotImplementedError


def save_processed(df: pd.DataFrame, path: str) -> None:
    raise NotImplementedError


def load_processed(path: str) -> pd.DataFrame:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Per-dataset wrappers (pick two classification + two regression for the report,
# but keep all six working)
# ---------------------------------------------------------------------------

def preprocess_breast_cancer(df: pd.DataFrame):      # classification
    raise NotImplementedError


def preprocess_car_evaluation(df: pd.DataFrame):     # classification
    raise NotImplementedError


def preprocess_house_votes(df: pd.DataFrame):        # classification
    raise NotImplementedError


def preprocess_abalone(df: pd.DataFrame):            # regression
    raise NotImplementedError


def preprocess_computer_hardware(df: pd.DataFrame):  # regression
    raise NotImplementedError


def preprocess_forest_fires(df: pd.DataFrame):       # regression
    raise NotImplementedError
EOF

write "$PROJECT_DIR/src/activations.py" <<'EOF'
"""
Activation functions and their derivatives.

Per the assignment, hidden layers may use ONLY logistic (sigmoid) or tanh.
No ReLU or other nonlinearities. Output is softmax (classification) or
linear / identity (regression).
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative expressed in terms of the activation a = sigmoid(z)."""
    raise NotImplementedError


def tanh(z: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def tanh_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative expressed in terms of the activation a = tanh(z)."""
    raise NotImplementedError


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise, numerically stable softmax."""
    raise NotImplementedError


def linear(z: np.ndarray) -> np.ndarray:
    raise NotImplementedError
EOF

write "$PROJECT_DIR/src/losses.py" <<'EOF'
"""
Loss functions.

- Cross-entropy for classification (softmax output)
- Mean squared error for regression (linear output) and for the autoencoder
  reconstruction loss  L(x, x_hat) = ||x - x_hat||^2
"""

import numpy as np


def cross_entropy(y_true_onehot: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    raise NotImplementedError


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError
EOF

write "$PROJECT_DIR/src/linear_network.py" <<'EOF'
"""
Linear networks (Figure 1 in the assignment): inputs feed DIRECTLY into the
output layer, no hidden layers.

- LogisticRegression : inputs -> softmax output (classification, cross-entropy)
- LinearRegression   : inputs -> single linear output node (regression, MSE),
                       trained by gradient descent (equivalent to OLS)
"""

import numpy as np


class LogisticRegression:
    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.01):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y_onehot: np.ndarray, epochs: int = 100, batch_size: int | None = None):
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearRegression:
    def __init__(self, n_features: int, learning_rate: float = 0.01):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int | None = None):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
EOF

write "$PROJECT_DIR/src/neural_network.py" <<'EOF'
"""
Feedforward neural network trained with backpropagation (Figure 2).

Topology: Input -> Hidden 1 -> Hidden 2 -> Output
- Hidden activation: logistic OR tanh (choose one, document it in the report)
- Output: softmax + cross-entropy (classification) or linear + MSE (regression)
- Optional momentum

The class should also support:
- initializing the first-layer weights from a pre-trained encoder
- freezing chosen layers (for the autoencoder variant where encoder weights
  are locked) vs. fine-tuning the whole network
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], mode: str = "classification",
                 hidden_activation: str = "sigmoid", learning_rate: float = 0.01,
                 momentum: float = 0.0, seed: int | None = None):
        """
        layer_sizes: e.g. [n_inputs, h1, h2, n_outputs]
        mode: "classification" (softmax + cross-entropy) or "regression" (linear + MSE)
        """
        raise NotImplementedError

    def forward(self, X: np.ndarray):
        raise NotImplementedError

    def backward(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
            batch_size: int | None = None, frozen_layers: list[int] | None = None):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_layer_weights(self, layer_index: int):
        raise NotImplementedError

    def set_layer_weights(self, layer_index: int, W: np.ndarray, b: np.ndarray):
        raise NotImplementedError
EOF

write "$PROJECT_DIR/src/autoencoder.py" <<'EOF'
"""
Autoencoder and the autoencoder-based prediction network (Figure 3).

Stage 1 - Autoencoder: Input -> Encoding (e, strictly fewer nodes than inputs)
          -> Decoding (same number of nodes as inputs), trained with MSE so the
          output reconstructs the input.
Stage 2 - Prediction network: drop the decoder, attach a hidden layer and the
          prediction layer (softmax / linear) to the encoder output:
          Input -> Encoding -> Hidden 2 -> Prediction
          Train with backprop, either (a) encoder weights locked, or
          (b) whole network fine-tuned. Pick one and justify it in the report.
"""

import numpy as np

from src.neural_network import NeuralNetwork


class Autoencoder:
    def __init__(self, n_inputs: int, n_encoding: int, hidden_activation: str = "sigmoid",
                 learning_rate: float = 0.01, momentum: float = 0.0, seed: int | None = None):
        if n_encoding >= n_inputs:
            raise ValueError("Encoding layer must have strictly fewer nodes than the input layer.")
        raise NotImplementedError

    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int | None = None):
        raise NotImplementedError

    def encode(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reconstruction_error(self, X: np.ndarray) -> float:
        raise NotImplementedError


def build_autoencoder_network(autoencoder: Autoencoder, n_hidden2: int, n_outputs: int,
                              mode: str = "classification", freeze_encoder: bool = True,
                              **kwargs) -> NeuralNetwork:
    """Build Input -> Encoding -> Hidden 2 -> Prediction from a trained autoencoder."""
    raise NotImplementedError
EOF

write "$PROJECT_DIR/src/evaluation.py" <<'EOF'
"""
5x2 cross-validation and metrics for all three network types.

5x2 CV: 5 repetitions of 2-fold CV (10 train/test runs per model per data set).
- Stratified folds for classification
- Same fold splits across all three models so results can be compared
  statistically (paired tests)
- Normalization fit on the training fold only
"""

import numpy as np


def classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def five_by_two_splits(y: np.ndarray, mode: str = "classification", seed: int = 0):
    """Yield (rep, fold, train_idx, test_idx) for the 10 runs."""
    raise NotImplementedError


def cross_validation_loop(X: np.ndarray, y: np.ndarray, model_factory,
                          mode: str = "classification", seed: int = 0):
    """
    Run 5x2 CV for one model on one data set.
    model_factory: callable returning a fresh, untrained model.
    Returns a list of per-fold score dicts.
    """
    raise NotImplementedError


def paired_t_test(scores_a: list[float], scores_b: list[float]):
    """Statistical comparison of two models' per-fold scores."""
    raise NotImplementedError
EOF

write "$PROJECT_DIR/src/tuning.py" <<'EOF'
"""
Hyperparameter tuning.

Tune on a validation split carved out of the TRAINING data (never the test
fold). Things to tune and explain in the report:
- hidden nodes in layer 1 and layer 2 (feedforward network)
- encoding size (strictly < number of inputs), hidden 2 size (autoencoder net)
- learning rate, momentum, number of epochs / batch size

Save tuning grids and chosen values to results/tuning/ so the report can cite them.
"""

import numpy as np


def grid_search(model_factory, param_grid: dict, X: np.ndarray, y: np.ndarray,
                mode: str = "classification", seed: int = 0):
    """Return (best_params, results_table)."""
    raise NotImplementedError
EOF

# --- notebooks ---------------------------------------------------------------
notebook "$PROJECT_DIR/notebooks/01_explore_data.ipynb" \
    "01 - Explore Data" \
    "Look at the six UCI data sets, class balance, missing values, and feature ranges."
notebook "$PROJECT_DIR/notebooks/02_preprocessing_tests.ipynb" \
    "02 - Preprocessing Tests" \
    "Run each preprocess_* function and save the outputs to data/processed/."
notebook "$PROJECT_DIR/notebooks/03_network_sanity_checks.ipynb" \
    "03 - Network Sanity Checks" \
    "Gradient checking, overfit-a-tiny-dataset tests, softmax/cross-entropy checks, autoencoder reconstruction."
notebook "$PROJECT_DIR/notebooks/04_tuning.ipynb" \
    "04 - Hyperparameter Tuning" \
    "Tune hidden sizes, learning rate, momentum, and epochs for each network on each data set."
notebook "$PROJECT_DIR/notebooks/05_run_experiments.ipynb" \
    "05 - Run Experiments" \
    "5x2 cross-validation for the linear, feedforward, and autoencoder networks on each data set."
notebook "$PROJECT_DIR/notebooks/06_results_analysis.ipynb" \
    "06 - Results Analysis" \
    "Tables, figures, and statistical tests for the report."

# --- report ------------------------------------------------------------------
write "$PROJECT_DIR/report/references.bib" <<'EOF'
@book{alpaydin2020,
  author    = {Ethem Alpaydin},
  title     = {Introduction to Machine Learning},
  edition   = {4},
  publisher = {MIT Press},
  year      = {2020}
}

@misc{uci_repository,
  author    = {Dua, Dheeru and Graff, Casey},
  title     = {{UCI} Machine Learning Repository},
  year      = {2019},
  url       = {http://archive.ics.uci.edu/ml},
  publisher = {University of California, Irvine, School of Information and Computer Sciences}
}

@article{rumelhart1986,
  author  = {Rumelhart, David E. and Hinton, Geoffrey E. and Williams, Ronald J.},
  title   = {Learning representations by back-propagating errors},
  journal = {Nature},
  volume  = {323},
  pages   = {533--536},
  year    = {1986}
}

@article{hinton2006,
  author  = {Hinton, Geoffrey E. and Salakhutdinov, Ruslan R.},
  title   = {Reducing the dimensionality of data with neural networks},
  journal = {Science},
  volume  = {313},
  number  = {5786},
  pages   = {504--507},
  year    = {2006}
}

@article{dietterich1998,
  author  = {Dietterich, Thomas G.},
  title   = {Approximate statistical tests for comparing supervised classification learning algorithms},
  journal = {Neural Computation},
  volume  = {10},
  number  = {7},
  pages   = {1895--1923},
  year    = {1998}
}
EOF

write "$PROJECT_DIR/report/P3_Paper.tex" <<'EOF'
\documentclass[twoside,11pt]{article}
\usepackage{multirow}
\usepackage{jmlr2e}
\usepackage{lastpage}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{appendix}
\usepackage{graphicx}

\graphicspath{{figures/}{../results/figures/}}

% jmlrheading{volume}{year}{pages}{date submitted}{date published}{paper id}{authors}
\jmlrheading{26}{2026}{1-\pageref{LastPage}}{11/9/26}{11/9/26}{25-0050}{Isaac Wilkey and Riley Smith}
\ShortHeadings{Backpropagation and Autoencoder Feature Extraction}{Wilkey and Smith}
\firstpageno{1}

\begin{document}

\title{Logistic Regression, Backpropagation Networks, and Autoencoder Pre-training}

\author{\name Isaac Wilkey \email isaac.wilkey@student.montana.edu\\
\addr Department of Computer Science \\
Montana State University \\
Bozeman, MT, USA
\AND
\name Riley Smith \email riley.smith13@student.montana.edu\\
\addr Department of Computer Science \\
Montana State University \\
Bozeman, MT, USA
}
\editor{Isaac Wilkey \& Riley Smith}

\maketitle

\begin{abstract}
TODO: Summarize the problem, the three network types compared (linear, two-hidden-layer feedforward, autoencoder-based), the data sets, the $5 \times 2$ cross-validation protocol, and the main findings.
\end{abstract}

\begin{keywords}
backpropagation, autoencoder, logistic regression, feature extraction, $5 \times 2$ cross validation
\end{keywords}

\section{Problem Statement and Hypotheses}
TODO: State the goal of the project and at least one testable hypothesis, including what result would falsify it.
\begin{enumerate}
    \item \textbf{H1:} TODO
    \item \textbf{H2:} TODO
\end{enumerate}

\section{Experimental Approach and Project Design}
\subsection{Datasets}
TODO: Table of the four chosen data sets (two classification, two regression) with instances, features, and task.

\subsection{Preprocessing}
TODO: Missing values, one-hot encoding, normalization (fit on training folds only), target transforms.

\subsection{Cross Validation and Tuning}
TODO: $5 \times 2$ cross validation, how folds are shared across the three models, and how hyperparameters were tuned (hidden nodes, encoding size, learning rate, momentum, epochs) without touching the test fold.

\subsection{Evaluation Metrics}
TODO: Classification error / cross-entropy, mean squared error, and the statistical test used to compare models.

\section{Algorithms Implemented}
\subsection{Linear Networks}
TODO: Logistic regression (softmax + cross-entropy) and linear regression by gradient descent.

\subsection{Feedforward Network with Backpropagation}
TODO: Topology (Input $\Rightarrow$ Hidden 1 $\Rightarrow$ Hidden 2 $\Rightarrow$ Prediction), hidden activation choice, momentum, weight initialization, loss functions.

\subsection{Autoencoder-Based Network}
TODO: Under-complete autoencoder with loss $L(\mathbf{x}, \hat{\mathbf{x}}) = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$, removal of the decoder, and the prediction layers. Explain and justify whether encoder weights are frozen or fine-tuned.

\section{Experimental Results}
TODO: Tables and figures with explanation.

\section{Analysis and Discussion}
TODO

\section{Conclusion}
TODO

\bibliography{references}

\newpage

\appendix
\section{Main Lessons Learned}
TODO

\section{Delineation of Work}
\begin{itemize}
    \item \textbf{Isaac Wilkey:} TODO
    \item \textbf{Riley Smith:} TODO
\end{itemize}

\end{document}
EOF

# --- repo README note --------------------------------------------------------
write "$PROJECT_DIR/README.md" <<'EOF'
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
EOF

echo
echo "Done. Created $PROJECT_DIR/"
echo "Next: cd $PROJECT_DIR && python -m venv venv && pip install -r requirements.txt"
