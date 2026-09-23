"""
Preprocessing functions for cleaning and preparing data for decision trees.

Key differences from a distance-based (kNN) preprocessing pipeline:
- No normalization (min-max / z-score): trees split on threshold
  comparisons, which are invariant to monotonic rescaling.
- No one-hot encoding: trees split natively on categorical values
  (multi-way split, one branch per distinct value), so one-hot would
  only fragment a single meaningful split into several worse ones.
- No cyclic (sin/cos) encoding: that existed to fix distance
  calculations. Trees don't compute distances, so a plain categorical
  split on month/day values is sufficient.
- NEW: every preprocess_* wrapper now also returns a `feature_types`
  dict ({column_name: "numeric" | "categorical"}), since your tree code
  needs this to decide, per feature, whether to search a binary
  threshold split or do a full multi-way categorical split.
"""

import numpy as np
import pandas as pd
import json

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


def label_encode(df: pd.DataFrame, columns: list[str], orderings: dict | None = None) -> pd.DataFrame:
    """
    Encode categorical columns as integers.

    orderings: optional dict {column_name: [ordered_values]} to control
    the integer assignment. If not provided for a column, values are
    encoded in sorted order.

    NOTE: for trees, this is just a convenient integer representation --
    it does NOT make the feature numeric/ordered from the tree's
    perspective. Mark it "categorical" in feature_types so your tree
    code does a multi-way split on distinct values, not a threshold split.
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


def log_transform(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Apply ln(x + 1) transform to a skewed target (e.g. Forest Fires area)."""
    df = df.copy()
    df[column] = np.log(df[column] + 1)
    return df


def parse_to_int(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert string columns to numeric, if possible."""
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_feature_types(df: pd.DataFrame, categorical_cols: list[str], target_col: str) -> dict:
    """
    Build a {column_name: "numeric" | "categorical"} map for every feature
    column (excludes the target). Pass this alongside the cleaned df to
    your tree-building code.
    """
    return {
        col: ("categorical" if col in categorical_cols else "numeric")
        for col in df.columns
        if col != target_col
    }


# ---------------------------------------------------------------------------
# Save / load feature_types alongside a processed CSV
# ---------------------------------------------------------------------------
 
def save_processed(df: pd.DataFrame, feature_types: dict, path: str) -> None:
    """
    Save a processed DataFrame to `path` (a .csv path) and its feature_types
    to a matching .json sidecar file (same base name, same folder).
 
    e.g. save_processed(df, feature_types, "../data/processed/breast_cancer_processed.csv")
    writes both breast_cancer_processed.csv and breast_cancer_processed.json
    """
    df.to_csv(path, index=False)
    json_path = path.rsplit(".csv", 1)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(feature_types, f, indent=2)
 
 
def load_processed(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load a processed DataFrame and its feature_types back from disk,
    given the .csv path (the matching .json sidecar is inferred).
    """
    df = pd.read_csv(path)
    json_path = path.rsplit(".csv", 1)[0] + ".json"
    with open(json_path) as f:
        feature_types = json.load(f)
    return df, feature_types


# ---------------------------------------------------------------------------
# Unused-for-trees utilities kept around in case you need them elsewhere
# (e.g. quick EDA, or if you reuse this file for a future project).
# Not called by any wrapper below.
# ---------------------------------------------------------------------------

def z_score_normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Scale numeric columns to zero mean, unit variance. NOT used for trees."""
    df = df.copy()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = 0.0 if std == 0 else (df[col] - mean) / std
    return df


def min_max_normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Scale numeric columns to the range [0, 1]. NOT used for trees."""
    df = df.copy()
    for col in columns:
        col_min = df[col].min()
        col_max = df[col].max()
        denom = col_max - col_min
        df[col] = 0.0 if denom == 0 else (df[col] - col_min) / denom
    return df


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot encode categorical columns. NOT used for trees -- see module docstring."""
    return pd.get_dummies(df, columns=columns, dtype=int)


def cyclic_encode(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    """Sin/cos encode a cyclic feature. NOT used for trees -- see module docstring."""
    df = df.copy()
    radians = 2 * np.pi * df[column] / period
    df[f"{column}_sin"] = np.sin(radians)
    df[f"{column}_cos"] = np.cos(radians)
    return df.drop(columns=[column])


# ---------------------------------------------------------------------------
# Dataset-specific wrappers
# ---------------------------------------------------------------------------

def preprocess_breast_cancer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_col = "class"
    df = drop_columns(df, ["sample_code_number"])
    numeric_cols = [c for c in df.columns if c != target_col]
    df = handle_missing(df, numeric_cols, strategy="mode", missing_values=["?"])
    df[numeric_cols] = df[numeric_cols].astype(float)
    # Features are all in [1, 10]; treated as numeric per assignment notes.
    feature_types = build_feature_types(df, categorical_cols=[], target_col=target_col)
    return df, feature_types


def preprocess_car_evaluation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_col = "class"
    buying_order = ["low", "med", "high", "vhigh"]
    maint_order = ["low", "med", "high", "vhigh"]
    doors_order = ["2", "3", "4", "5more"]
    persons_order = ["2", "4", "more"]
    lug_boot_order = ["small", "med", "big"]
    safety_order = ["low", "med", "high"]

    df = label_encode(df, ["buying"], orderings={"buying": buying_order})
    df = label_encode(df, ["maint"], orderings={"maint": maint_order})
    df = label_encode(df, ["doors"], orderings={"doors": doors_order})
    df = label_encode(df, ["persons"], orderings={"persons": persons_order})
    df = label_encode(df, ["lug_boot"], orderings={"lug_boot": lug_boot_order})
    df = label_encode(df, ["safety"], orderings={"safety": safety_order})

    # Project 1's dataset notes say to treat these as nominal even though
    # technically ordinal -- so mark ALL of them categorical (multi-way
    # split), not numeric (threshold split). Revisit this if you decide
    # ordinal treatment is more appropriate for tree splits.
    categorical_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    feature_types = build_feature_types(df, categorical_cols=categorical_cols, target_col=target_col)
    return df, feature_types


def preprocess_house_votes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # "?" means "abstain" here, NOT missing -- treated as its own category.
    target_col = "class"
    feature_cols = [c for c in df.columns if c != target_col]
    df = label_encode(df, feature_cols)  # y/n/? each become their own code

    feature_types = build_feature_types(df, categorical_cols=feature_cols, target_col=target_col)
    return df, feature_types


def preprocess_abalone(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_col = "rings"
    # Unlike Project 1, we KEEP sex rather than dropping it -- trees can
    # split on it natively without needing one-hot encoding.
    df = label_encode(df, ["sex"])

    feature_types = build_feature_types(df, categorical_cols=["sex"], target_col=target_col)
    return df, feature_types


def preprocess_computer_hardware(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_col = "prp"
    df = drop_columns(df, ["vendor_name", "model_name"])
    # ERP was estimated by the original authors via linear regression --
    # not a usable feature, but keep it out of the returned df entirely
    # (save it separately from your raw file if you want it for comparison).
    df = drop_columns(df, ["erp"])

    # All remaining features are numeric per assignment notes.
    feature_types = build_feature_types(df, categorical_cols=[], target_col=target_col)
    return df, feature_types


def preprocess_forest_fires(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_col = "area"
    month_order = ["jan", "feb", "mar", "apr", "may", "jun",
                   "jul", "aug", "sep", "oct", "nov", "dec"]
    day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    # Label-encoded for a convenient integer representation, but marked
    # categorical below -- NOT cyclic-encoded, since trees don't need
    # month/day to be "distance-aware."
    df = label_encode(df, ["month"], orderings={"month": month_order})
    df = label_encode(df, ["day"], orderings={"day": day_order})

    df = parse_to_int(df, ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"])

    # Target skew fix -- independent of model type, still worth applying.
    df = log_transform(df, target_col)

    categorical_cols = ["month", "day"]
    feature_types = build_feature_types(df, categorical_cols=categorical_cols, target_col=target_col)
    return df, feature_types


# ---------------------------------------------------------------------------
# Quick manual test / usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "feat_a": [10, 20, "?", 40],
        "feat_b": ["red", "blue", "red", "green"],
        "month": ["dec", "jan", "jun", "jul"],
        "target": [0, 1, 0, 1],
    })

    print("Original:")
    print(sample)

    cleaned = drop_columns(sample, ["id"])
    cleaned = handle_missing(cleaned, ["feat_a"], strategy="mode", missing_values=["?"])
    cleaned["feat_a"] = cleaned["feat_a"].astype(float)
    cleaned = label_encode(cleaned, ["feat_b"])
    cleaned = label_encode(cleaned, ["month"],
                            orderings={"month": ["jan", "feb", "mar", "apr", "may", "jun",
                                                  "jul", "aug", "sep", "oct", "nov", "dec"]})

    feature_types = build_feature_types(cleaned, categorical_cols=["feat_b", "month"], target_col="target")

    print("\nCleaned:")
    print(cleaned)
    print("\nFeature types:")
    print(feature_types)