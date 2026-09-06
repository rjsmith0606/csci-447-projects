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
    missing_values: list = None,
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
    """Scale numeric columns to zero mean, unit variance."""
    df = df.copy()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = 0.0 if std == 0 else (df[col] - mean) / std
    return df


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    return pd.get_dummies(df, columns=columns)


def log_transform(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Apply ln(x + 1) transform to a skewed target (e.g. Forest Fires area)."""
    df = df.copy()
    df[column] = np.log(df[column] + 1)
    return df


# ---------------------------------------------------------------------------
# Dataset-specific wrappers
# ---------------------------------------------------------------------------
# Each function takes a raw DataFrame (already loaded from disk) and
# returns a cleaned DataFrame ready for distance/model code.
# Fill in the exact column names once you've loaded each raw file, since
# UCI files often ship without headers.

def preprocess_breast_cancer(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_columns(df, ["sample_code_number"])
    numeric_cols = [c for c in df.columns if c != "class"] 
    df = handle_missing(df, numeric_cols, strategy="mode", missing_values=["?"])
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = z_score_normalize(df, numeric_cols)
    return df


# def preprocess_car_evaluation(df: pd.DataFrame) -> pd.DataFrame:
#     feature_cols = [c for c in df.columns if c != "class"]
#     # All features treated as nominal per the assignment notes
#     df = one_hot_encode(df, feature_cols)
#     return df


# def preprocess_congressional_vote(df: pd.DataFrame) -> pd.DataFrame:
#     # NOTE: "?" means "abstain" here, NOT missing -- treat it as its own
#     # category rather than dropping/imputing it.
#     feature_cols = [c for c in df.columns if c != "class"]
#     df = one_hot_encode(df, feature_cols)
#     return df


# def preprocess_abalone(df: pd.DataFrame, encode_sex: bool = True) -> pd.DataFrame:
#     numeric_cols = [c for c in df.columns if c not in ("sex", "rings")]
#     df = z_score_normalize(df, numeric_cols)
#     if encode_sex:
#         df = one_hot_encode(df, ["sex"])
#     else:
#         df = drop_columns(df, ["sex"])
#     return df


# def preprocess_computer_hardware(df: pd.DataFrame) -> pd.DataFrame:
#     df = drop_columns(df, ["vendor_name", "model_name"])
#     # Save ERP separately for later comparison -- don't use as a feature
#     erp = df["erp"].copy() if "erp" in df.columns else None
#     df = drop_columns(df, ["erp"])
#     numeric_cols = [c for c in df.columns if c != "prp"]
#     df = z_score_normalize(df, numeric_cols)
#     return df, erp


# def preprocess_forest_fires(df: pd.DataFrame) -> pd.DataFrame:
#     month_order = ["jan", "feb", "mar", "apr", "may", "jun",
#                    "jul", "aug", "sep", "oct", "nov", "dec"]
#     day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

#     df = label_encode(df, ["month"], orderings={"month": month_order})
#     df = cyclic_encode(df, "month", period=12)

#     df = label_encode(df, ["day"], orderings={"day": day_order})
#     df = cyclic_encode(df, "day", period=7)

#     numeric_cols = [c for c in df.columns if c not in ("area",)]
#     df = min_max_normalize(df, numeric_cols)

#     df = log_transform(df, "area")
#     return df


# ---------------------------------------------------------------------------
# Quick manual test / usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic example just to sanity-check the generic functions
    # before wiring up real datasets.
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
    cleaned = handle_missing(cleaned, ["feat_a"], strategy="impute_mode", missing_values=["?"])
    cleaned["feat_a"] = cleaned["feat_a"].astype(float)
    # cleaned = min_max_normalize(cleaned, ["feat_a"])
    cleaned = one_hot_encode(cleaned, ["feat_b"])
    # cleaned = label_encode(cleaned, ["month"],
                            # orderings={"month": ["jan", "feb", "mar", "apr", "may", "jun",
                            #                       "jul", "aug", "sep", "oct", "nov", "dec"]})
    # cleaned = cyclic_encode(cleaned, "month", period=12)

    print("\nCleaned:")
    print(cleaned)
