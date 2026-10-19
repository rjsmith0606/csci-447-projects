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
