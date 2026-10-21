"""
data_cleaning.py
CSC311 Machine Learning Project: Survey Data Cleaning & Preprocessing Pipeline
Author: Chris Yang

This script implements the complete data preprocessing workflow:
1. Identifier Isolation (id excluded from features to prevent leakage)
2. Train / Validation / Test Splitting (70% Train, 15% Val, 15% Test grouped by respondent ID)
3. Ticket Price Parsing (regex digit extraction, word numbers, log1p transformation)
4. Athletic Accessibility Standardization (difficulty rating, mean imputation, Z-score)
5. Likert Agreement Ordinal Mapping (1-5 monotonic numeric scale)
6. Multi-Select Multi-Hot Encoding (locations, seasons, music genres)
7. Free-Text Cleaning & Missing Indicators (lowercasing, punctuation stripping, missing tokens)
8. Zero-Leakage Parameter Fitting: All statistical parameters (mean, std, median) are fitted
   STRICTLY on the training split, and then applied to transform validation and test splits.
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Column renaming mapping: survey questions to standardized variable names
RENAME_MAP = {
    "How easy do you think it would be for someone with no prior experience to start playing this sport recreationally? Rate it on a scale from 0 to 10, where 0 means extremely difficult and 10 means extremely easy.": "difficulty_rating",
    "Success in this sport depends heavily on strategy and planning.": "strategy_importance",
    "Players must be comfortable with a high risk of injury.": "injury_risk_comfort",
    "Individual brilliance can outweigh teamwork in this sport.": "individual_brilliance",
    "Where might you play this sport?": "play_location",
    "During which seasons would you be most likely to play this sport recreationally in Toronto?": "play_season",
    "What types of music do you most associate with this sport?": "associated_music",
    "What foods and drinks do you associate with watching this sport in the place or culture you know best?": "associated_foods_drinks",
    "What colour do you associate with this sport?": "associated_color",
    "Which song do you associate with this sport?": "associated_song",
    "How do you feel when you watch this sport?": "emotional_response",
    "How much would you be willing to pay for a decent seat at a professional competition of this sport?": "acceptable_price",
    "label": "target_sport",
}

# Likert 5-point agreement mapping to monotonic integers
LIKERT_MAP = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}

LABEL_MAP = {
    "Soccer": 0,
    "Baseball": 1,
    "Basketball": 2,
}


def parse_ticket_price(val):
    """
    Parses messy ticket price responses into clean numeric float values.
    Handles currency symbols, word numbers (thousand/k), price ranges, and qualitative text.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()

    if s in ["", "none", "n/a", "na", "null", "-", "no"]:
        return np.nan

    if "free" in s or s == "0":
        return 0.0

    # Resolve word numbers like '1 thousand' or '1k'
    s = re.sub(r"(\d+)\s*(?:thousand|k)\b", r"\g<1>000", s)

    # Resolve numerical ranges (e.g., '200-300' or '50 to 100') by taking midpoint
    range_match = re.findall(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", s)
    if range_match:
        low, high = float(range_match[0][0]), float(range_match[0][1])
        return (low + high) / 2.0

    # Extract clean digits, removing thousand separator commas
    num_match = re.findall(r"(\d+(?:,\d{3})*(?:\.\d+)?)", s)
    if num_match:
        return float(num_match[0].replace(",", ""))

    return np.nan


def clean_base_features(df_raw):
    """
    Applies deterministic, rule-based feature cleaning (no statistics learned).
    Safe to apply independently to train, validation, or test sets.
    """
    df = df_raw.copy()

    # Step 1: Standardize column names
    for old_col, new_col in RENAME_MAP.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})

    # Step 2: Numeric ticket price parsing
    if "acceptable_price" in df.columns:
        df["price_clean"] = df["acceptable_price"].apply(parse_ticket_price)

    # Step 3: Athletic difficulty rating
    if "difficulty_rating" in df.columns:
        df["difficulty_clean"] = pd.to_numeric(df["difficulty_rating"], errors="coerce")

    # Step 4: Ordinal mapping of 5-point Likert questions
    likert_cols = ["strategy_importance", "injury_risk_comfort", "individual_brilliance"]
    for col in likert_cols:
        if col in df.columns:
            df[f"{col}_num"] = df[col].map(LIKERT_MAP)

    # Step 5: Multi-select questions to binary multi-hot encoding
    multi_select_cols = ["play_season", "play_location", "associated_music"]
    for col in multi_select_cols:
        if col in df.columns:
            dummies = df[col].fillna("None").str.get_dummies(sep=",").add_prefix(f"{col}_")
            df = pd.concat([df, dummies], axis=1)

    # Step 6: Free-text normalization and missingness indicators
    text_cols = ["associated_foods_drinks", "associated_color", "associated_song", "emotional_response"]
    for col in text_cols:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(int)
            df[f"{col}_clean"] = (
                df[col]
                .fillna("missing")
                .astype(str)
                .str.lower()
                .str.replace(r"[^\w\s]", "", regex=True)
                .str.strip()
            )

    # Step 7: Map label if present
    if "target_sport" in df.columns:
        df["target_sport_num"] = df["target_sport"].map(LABEL_MAP)

    return df


def fit_train_parameters(train_df):
    """
    Learns statistical parameters (medians, means, standard deviations)
    EXCLUSIVELY from the training partition to prevent data snooping.
    """
    params = {}
    params["price_median"] = train_df["price_clean"].median()
    train_price_imputed = train_df["price_clean"].fillna(params["price_median"])
    train_log_price = np.log1p(train_price_imputed)
    params["log_price_mean"] = train_log_price.mean()
    params["log_price_std"] = train_log_price.std()

    params["diff_mean"] = train_df["difficulty_clean"].mean()
    params["diff_std"] = train_df["difficulty_clean"].std()

    return params


def apply_statistical_transformations(df, params):
    """
    Transforms numerical features using pre-fitted training parameters.
    Guarantees strict separation without recomputing metrics on val/test data.
    """
    df_out = df.copy()

    # 1. Price imputation, log1p, and standardization
    if "price_clean" in df_out.columns:
        imputed_price = df_out["price_clean"].fillna(params["price_median"])
        log_p = np.log1p(imputed_price)
        df_out["log_price"] = log_p
        df_out["log_price_std"] = (log_p - params["log_price_mean"]) / (params["log_price_std"] + 1e-8)

    # 2. Difficulty rating imputation and Z-score
    if "difficulty_clean" in df_out.columns:
        imputed_diff = df_out["difficulty_clean"].fillna(params["diff_mean"])
        df_out["difficulty_zscore"] = (imputed_diff - params["diff_mean"]) / (params["diff_std"] + 1e-8)

    # 3. Fill missing Likert values with neutral score 3
    likert_cols = ["strategy_importance_num", "injury_risk_comfort_num", "individual_brilliance_num"]
    for col in likert_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(3.0)

    return df_out


def split_and_clean_pipeline(csv_path="csc311_project_training_202609.csv", random_state=42):
    """
    Executes the end-to-end 70% Train, 15% Val, 15% Test split and cleaning pipeline.
    Ensures zero data leakage.
    """
    raw_df = pd.read_csv(csv_path)

    # Split unique respondents: 70% train (203 IDs), 15% val (43 IDs), 15% test (44 IDs)
    unique_ids = raw_df["id"].unique()
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.30, random_state=random_state)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=random_state)

    train_raw = raw_df[raw_df["id"].isin(train_ids)].copy()
    val_raw = raw_df[raw_df["id"].isin(val_ids)].copy()
    test_raw = raw_df[raw_df["id"].isin(test_ids)].copy()

    # Step A: Apply deterministic rule cleaning to all splits
    train_base = clean_base_features(train_raw)
    val_base = clean_base_features(val_raw)
    test_base = clean_base_features(test_raw)

    # Step B: Learn statistical parameters ONLY from training split
    learned_params = fit_train_parameters(train_base)

    # Step C: Apply transformations to all splits using training parameters
    train_clean = apply_statistical_transformations(train_base, learned_params)
    val_clean = apply_statistical_transformations(val_base, learned_params)
    test_clean = apply_statistical_transformations(test_base, learned_params)

    return train_clean, val_clean, test_clean, learned_params


if __name__ == "__main__":
    train_df, val_df, test_df, params = split_and_clean_pipeline()
    print("=== Zero-Leakage Preprocessing Pipeline Executed Successfully ===")
    print(f"Train split shape: {train_df.shape} (203 unique respondents)")
    print(f"Val split shape:   {val_df.shape}   (43 unique respondents)")
    print(f"Test split shape:  {test_df.shape}  (44 unique respondents)")
    print("Learned Training Parameters (Never computed on val/test):")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")
