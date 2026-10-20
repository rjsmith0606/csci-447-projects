import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Generic preprocessing functions
# ---------------------------------------------------------------------------

def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop columns that aren't useful (IDs, names, etc.)."""
    return df.drop(columns=columns, errors="ignore")


def handle_missing(
    df: pd.DataFrame,
    columns: list[str],
    strategy: str = "drop_rows",
    missing_values: list | None = None,
) -> pd.DataFrame:
    """
    Handle missing values in the given columns.
    """
    df = df.copy()

    # replace specified value with NaN so that pandas can handle it (e.g. "?" -> NaN)
    if missing_values:
        df[columns] = df[columns].replace(missing_values, np.nan)

    # drop any row with a missing value in `columns`
    if strategy == "drop_rows":
        return df.dropna(subset=columns)

    # replace missing numeric values with column mean (only works for numeric columns)
    if strategy == "mean":
        for col in columns:
            df[col] = df[col].fillna(df[col].mean())
        return df

    # replace missing values with column mode (works for numeric or categorical)
    if strategy == "mode":
        for col in columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        return df

    raise ValueError(f"Unknown strategy: {strategy}")


def z_score_normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Scale columns to zero mean, unit variance (across the whole data set)."""
    df = df.copy()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = 0.0 if std == 0 else (df[col] - mean) / std
    return df


def standardize_target(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, float, float]:
    """
    Z-score a regression target. Returns (df, mean, std) so predictions can be
    converted back with inverse_target.
    """
    df = df.copy()
    mean = float(df[column].mean())
    std = float(df[column].std())
    std = std if std != 0 else 1.0
    df[column] = (df[column] - mean) / std
    return df, mean, std


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot encode categorical columns (0/1 integer indicator columns)."""
    return pd.get_dummies(df, columns=columns, dtype=int)


def label_encode(df: pd.DataFrame, columns: list[str], orderings: dict | None = None) -> pd.DataFrame:
    """
    Encode categorical columns as integers.

    orderings: optional dict {column_name: [ordered_values]} to control
    the integer assignment for ordinal features. If not provided for a
    column, values are encoded in sorted order.
    """
    df = df.copy()
    orderings = orderings or {}
    for col in columns:
        if col in orderings:
            mapping = {val: i for i, val in enumerate(orderings[col])}
            df[col] = df[col].map(mapping)
        else:
            df[col] = df[col].astype("category").cat.codes
    return df


def cyclic_encode(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    """
    Encode a cyclic feature (e.g. month 1-12, day-of-week 0-6) as two
    columns using sine/cosine so that, e.g., December and January end up
    close together instead of far apart.

    Assumes `column` is already numeric (use label_encode first if needed).
    `period` is the number of steps in the full cycle (12 for months).
    """
    df = df.copy()
    radians = 2 * np.pi * df[column] / period
    df[f"{column}_sin"] = np.sin(radians)
    df[f"{column}_cos"] = np.cos(radians)
    return df.drop(columns=[column])


def log_transform(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Apply ln(x + 1) transform to a skewed target (e.g. Forest Fires area)."""
    df = df.copy()
    df[column] = np.log(df[column] + 1)
    return df


def parse_to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert string columns to numbers, if possible (otherwise NaN)."""
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Save / load meta alongside a processed CSV
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, meta: dict, path: str) -> None:
    """
    Save a processed DataFrame to `path` (a .csv path) and its meta to a
    matching .json sidecar file (same base name, same folder).

    e.g. save_processed(df, meta, "../data/processed/breast_cancer_processed.csv")
    writes both breast_cancer_processed.csv and breast_cancer_processed.json
    """
    df.to_csv(path, index=False)
    json_path = path.rsplit(".csv", 1)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_processed(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load a processed DataFrame and its meta back from disk,
    given the .csv path (the matching .json sidecar is inferred).
    """
    df = pd.read_csv(path)
    json_path = path.rsplit(".csv", 1)[0] + ".json"
    with open(json_path) as f:
        meta = json.load(f)
    return df, meta


# ---------------------------------------------------------------------------
# Network-ready arrays
# ---------------------------------------------------------------------------

def one_hot_labels(y_labels: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Integer class labels (0..K-1) -> one-hot matrix of shape (n, K)."""
    y_labels = np.asarray(y_labels, dtype=int)
    if n_classes is None:
        n_classes = int(y_labels.max()) + 1
    onehot = np.zeros((len(y_labels), n_classes))
    onehot[np.arange(len(y_labels)), y_labels] = 1.0
    return onehot


def prepare_xy(df: pd.DataFrame, meta: dict):
    """
    Turn a processed DataFrame + meta into arrays for the networks.

    Returns (X, y, y_labels):
      X        : (n, d) float array of features
      y        : classification -> (n, K) one-hot matrix (softmax targets)
                 regression     -> (n, 1) z-scored target
      y_labels : classification -> (n,) integer class indices (for error rate)
                 regression     -> None
    Class index i corresponds to meta["classes"][i].
    """
    target_col = meta["target"]
    X = df.drop(columns=[target_col]).to_numpy(dtype=float)

    if meta["task"] == "classification":
        class_to_idx = {c: i for i, c in enumerate(meta["classes"])}
        y_labels = np.array([class_to_idx[v] for v in df[target_col]], dtype=int)
        return X, one_hot_labels(y_labels, len(meta["classes"])), y_labels

    y = df[target_col].to_numpy(dtype=float).reshape(-1, 1)
    return X, y, None


def inverse_target(y_scaled: np.ndarray, meta: dict) -> np.ndarray:
    """Convert z-scored regression predictions back to the target's units (log units for Forest Fires)."""
    return np.asarray(y_scaled) * meta["target_std"] + meta["target_mean"]


# ---------------------------------------------------------------------------
# Dataset-specific wrappers
# ---------------------------------------------------------------------------

def preprocess_breast_cancer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_col = "class"
    df = drop_columns(df, ["sample_code_number"])
    feature_cols = [c for c in df.columns if c != target_col]
    df = handle_missing(df, feature_cols, strategy="mode", missing_values=["?"])
    df[feature_cols] = df[feature_cols].astype(float)
    df = z_score_normalize(df, feature_cols)

    meta = {
        "task": "classification",
        "target": target_col,
        "classes": sorted(df[target_col].unique().tolist()),  # 2 = benign, 4 = malignant
    }
    return df, meta


def preprocess_car_evaluation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_col = "class"
    orderings = {
        "buying": ["low", "med", "high", "vhigh"],
        "maint": ["low", "med", "high", "vhigh"],
        "doors": ["2", "3", "4", "5more"],
        "persons": ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety": ["low", "med", "high"],
    }
    # Ordinal features keep their natural order as integers, then get z-scored.
    df = label_encode(df, list(orderings), orderings=orderings)
    feature_cols = [c for c in df.columns if c != target_col]
    df = z_score_normalize(df, feature_cols)

    meta = {
        "task": "classification",
        "target": target_col,
        "classes": sorted(df[target_col].unique().tolist()),
    }
    return df, meta


def preprocess_house_votes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # "?" means "abstain" here, NOT missing -- treated as its own category.
    target_col = "class"
    feature_cols = [c for c in df.columns if c != target_col]
    df = one_hot_encode(df, feature_cols)  # y/n/? -> three 0/1 columns per vote; not z-scored

    meta = {
        "task": "classification",
        "target": target_col,
        "classes": sorted(df[target_col].unique().tolist()),
    }
    return df, meta


def preprocess_abalone(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_col = "rings"
    df = one_hot_encode(df, ["sex"])  # sex_F, sex_I, sex_M
    binary_cols = [c for c in df.columns if c.startswith("sex_")]
    measurement_cols = [c for c in df.columns if c not in binary_cols and c != target_col]
    df = z_score_normalize(df, measurement_cols)    
    df, mean, std = standardize_target(df, target_col)

    meta = {
        "task": "regression",
        "target": target_col,
        "target_mean": mean,
        "target_std": std,
    }
    return df, meta


def preprocess_computer_hardware(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_col = "prp"
    df = drop_columns(df, ["vendor_name", "model_name"])
    # ERP was estimated by the original authors via linear regression --
    # not a usable feature, so it is dropped.
    df = drop_columns(df, ["erp"])
    feature_cols = [c for c in df.columns if c != target_col]
    df = z_score_normalize(df, feature_cols)
    df, mean, std = standardize_target(df, target_col)

    meta = {
        "task": "regression",
        "target": target_col,
        "target_mean": mean,
        "target_std": std,
    }
    return df, meta


def preprocess_forest_fires(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_col = "area"
    month_order = ["jan", "feb", "mar", "apr", "may", "jun",
                   "jul", "aug", "sep", "oct", "nov", "dec"]
    day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    # forestfires.data ships with its own header row; if it was read in as a
    # data row (names=[...] without header=0), drop it.
    df = df[df["month"] != "month"].reset_index(drop=True)

    df = parse_to_numeric(df, ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"])

    df = label_encode(df, ["month"], orderings={"month": month_order})
    df = cyclic_encode(df, "month", period=12)
    df = label_encode(df, ["day"], orderings={"day": day_order})
    df = cyclic_encode(df, "day", period=7)

    df = log_transform(df, target_col)

    feature_cols = [c for c in df.columns if c != target_col]
    df = z_score_normalize(df, feature_cols)
    df, mean, std = standardize_target(df, target_col)  # target_mean/std are in ln(area + 1) units

    meta = {
        "task": "regression",
        "target": target_col,
        "target_mean": mean,
        "target_std": std,
    }
    return df, meta