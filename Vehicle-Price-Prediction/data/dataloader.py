#!/usr/bin/env python3
"""
Vehicle Price Prediction – Data Loader & Preprocessor

Features:
- Accepts one or many CarDekho-style CSVs (e.g., car_data.csv, Car Details from CarDekho.csv, Car details v3.csv, Car details v4.csv)
- Optionally accepts a directory and will read all *.csv files inside it
- Cleans and unifies schemas; parses numeric fields with units (mileage, engine, max_power, torque)
- Extracts MAKE from the name column (robust multi-word make matching)
- Engineers features (age, mileage_unit)
- Splits into train/val/test with stratification on binned target
- Builds a ColumnTransformer (scale numeric, one-hot categorical)
- Saves: preprocessor.joblib + processed_data.pkl + a quick summary .txt

Usage:
    python data/dataloader.py \
        --csv "data/Car details v3.csv" \
        --csv "data/Car details v4.csv" \
        --out outputs/

    # Or point to a directory with multiple CSVs
    python data/dataloader.py --dataset_dir data/ --out outputs/

Outputs (default):
- outputs/preprocessor.joblib
- outputs/processed_data.pkl (dict with X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
- outputs/data_summary.txt

Notes:
- Target column is auto-detected among ['selling_price', 'price', 'Price']
- You can override with --target price

"""
from __future__ import annotations
import sklearn
from packaging import version

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# Helpers
# -----------------------------

TARGET_CANDIDATES = ["selling_price", "price", "Price"]

# Common Indian market makes (multi-word first for greedy match)
COMMON_MAKES = [
    "Rolls-Royce", "Land Rover", "Mercedes-Benz", "Aston Martin", "Mahindra Renault",
    "Force Motors", "Isuzu", "Volkswagen", "Chevrolet", "Mitsubishi",
    "Mahindra", "Ambassador", "Lamborghini", "Bentley", "Renault", "Hyundai",
    "Maruti", "Toyota", "Jaguar", "Suzuki", "Skoda", "Nissan", "Porsche",
    "Ferrari", "Peugeot", "Citroen", "Kia", "MG", "Volvo", "Mini", "BMW",
    "Audi", "Fiat", "Jeep", "Tata", "Honda", "Lexus", "Datsun", "DC"
]
# Sort by length descending so multi-word/longer makes match before substrings
COMMON_MAKES = sorted(COMMON_MAKES, key=lambda s: len(s), reverse=True)

NUMERIC_COL_ALIASES = {
    "km_driven": ["km_driven", "odometer", "kms_driven", "kms"]
}

CAT_COL_ALIASES = {
    "fuel": ["fuel"],
    "transmission": ["transmission"],
    "owner": ["owner", "owners"],
    "seller_type": ["seller_type", "seller type", "seller"]
}

BASIC_COL_ALIASES = {
    "name": ["name", "car_name", "title"],
    "year": ["year", "model_year"],
}

EXTRA_NUMERIC = ["mileage", "engine", "max_power", "torque", "seats"]


@dataclass
class ParsedColumns:
    target: str
    name: Optional[str]
    year: Optional[str]
    km_driven: Optional[str]
    fuel: Optional[str]
    transmission: Optional[str]
    owner: Optional[str]
    seller_type: Optional[str]
    mileage: Optional[str]
    engine: Optional[str]
    max_power: Optional[str]
    torque: Optional[str]
    seats: Optional[str]


def _pick_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def detect_columns(df: pd.DataFrame) -> ParsedColumns:
    # Target
    target = _pick_column(df.columns.tolist(), TARGET_CANDIDATES)
    if target is None:
        raise ValueError(
            "Could not find target column among: selling_price / price / Price")

    # Basics
    name = _pick_column(df.columns.tolist(), BASIC_COL_ALIASES["name"])
    year = _pick_column(df.columns.tolist(), BASIC_COL_ALIASES["year"])

    # km_driven
    km_driven = _pick_column(
        df.columns.tolist(), NUMERIC_COL_ALIASES["km_driven"])

    # Categoricals
    fuel = _pick_column(df.columns.tolist(), CAT_COL_ALIASES["fuel"])
    transmission = _pick_column(
        df.columns.tolist(), CAT_COL_ALIASES["transmission"])
    owner = _pick_column(df.columns.tolist(), CAT_COL_ALIASES["owner"])
    seller_type = _pick_column(
        df.columns.tolist(), CAT_COL_ALIASES["seller_type"])

    # Extra numeric-ish
    mileage = "mileage" if "mileage" in df.columns else _pick_column(
        df.columns.tolist(), ["mileage"])
    engine = "engine" if "engine" in df.columns else _pick_column(
        df.columns.tolist(), ["engine"])
    max_power = "max_power" if "max_power" in df.columns else _pick_column(
        df.columns.tolist(), ["max_power", "max power", "power"])
    torque = "torque" if "torque" in df.columns else _pick_column(
        df.columns.tolist(), ["torque"])  # string in Cardekho
    seats = "seats" if "seats" in df.columns else _pick_column(
        df.columns.tolist(), ["seats"])  # numeric

    return ParsedColumns(
        target=target,
        name=name,
        year=year,
        km_driven=km_driven,
        fuel=fuel,
        transmission=transmission,
        owner=owner,
        seller_type=seller_type,
        mileage=mileage,
        engine=engine,
        max_power=max_power,
        torque=torque,
        seats=seats,
    )


_float_re = re.compile(r"[-+]?\d*\.?\d+")


def extract_float(val) -> Optional[float]:
    if pd.isna(val):
        return np.nan
    s = str(val)
    m = _float_re.search(s.replace(",", ""))
    return float(m.group()) if m else np.nan


def parse_torque(val) -> Tuple[Optional[float], Optional[float]]:
    """Attempt to parse torque strings like '190Nm@ 2000rpm' → (190, 2000).
    Returns (torque_nm, rpm)."""
    if pd.isna(val):
        return (np.nan, np.nan)
    s = str(val).lower().replace(" ", "")
    # Capture torque number (nm) and rpm if present
    nm = extract_float(s)
    # Find something like '2000rpm'
    rpm_match = re.search(r"(\d{3,5})rpm", s)
    rpm = float(rpm_match.group(1)) if rpm_match else np.nan
    return (nm, rpm)


def infer_make(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name)
    for make in COMMON_MAKES:
        # match at word boundary for robustness, case-insensitive
        if re.search(rf"\b{re.escape(make)}\b", s, flags=re.IGNORECASE):
            return make
    # fallback: first token
    return s.split()[0] if len(s.split()) > 0 else None


def clean_and_engineer(df: pd.DataFrame, cols: ParsedColumns, ref_year: Optional[int] = None) -> pd.DataFrame:
    # Rename target to 'price'
    if cols.target != "price":
        df = df.rename(columns={cols.target: "price"})
    # Standardize basic cols if present
    rename_map = {}
    if cols.km_driven and cols.km_driven != "km_driven":
        rename_map[cols.km_driven] = "km_driven"
    if cols.year and cols.year != "year":
        rename_map[cols.year] = "year"
    if cols.name and cols.name != "name":
        rename_map[cols.name] = "name"
    for k in ("fuel", "transmission", "owner", "seller_type"):
        v = getattr(cols, k)
        if v and v != k:
            rename_map[v] = k
    for k in ("mileage", "engine", "max_power", "torque", "seats"):
        v = getattr(cols, k)
        if v and v != k:
            rename_map[v] = k
    if rename_map:
        df = df.rename(columns=rename_map)

    # Keep only known columns + price; silently ignore extras
    keep_cols = [c for c in [
        "price", "name", "year", "km_driven", "fuel", "transmission", "owner", "seller_type",
        "mileage", "engine", "max_power", "torque", "seats"
    ] if c in df.columns]
    df = df[keep_cols].copy()

    # Basic cleaning
    # to numeric for km_driven (strip commas)
    if "km_driven" in df:
        df["km_driven"] = pd.to_numeric(df["km_driven"].astype(
            str).str.replace(",", ""), errors="coerce")

    # mileage: numeric value + unit feature
    if "mileage" in df:
        df["mileage_value"] = df["mileage"].apply(extract_float)
        df["mileage_unit"] = df["mileage"].astype(
            str).str.extract(r"(kmpl|km/kg)", expand=False)
    else:
        df["mileage_value"], df["mileage_unit"] = (np.nan, np.nan)

    # engine: e.g., '1248 CC'
    if "engine" in df:
        df["engine_cc"] = df["engine"].apply(extract_float)
    else:
        df["engine_cc"] = np.nan

    # max_power: e.g., '74 bhp'
    if "max_power" in df:
        df["max_power_bhp"] = df["max_power"].apply(extract_float)
    else:
        df["max_power_bhp"] = np.nan

    # torque: string -> (nm, rpm)
    if "torque" in df:
        t = df["torque"].apply(parse_torque)
        df["torque_nm"] = t.apply(lambda x: x[0])
        df["torque_rpm"] = t.apply(lambda x: x[1])
    else:
        df["torque_nm"], df["torque_rpm"] = (np.nan, np.nan)

    # seats to numeric
    if "seats" in df:
        df["seats"] = pd.to_numeric(df["seats"], errors="coerce")

    # derive make
    if "name" in df:
        df["make"] = df["name"].apply(infer_make)
    else:
        df["make"] = np.nan

    # derive age
    if "year" in df:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        ref = ref_year if ref_year is not None else pd.Timestamp.now().year
        df["age"] = ref - df["year"]
    else:
        df["age"] = np.nan

    # remove obvious outliers / non-sensical
    # - negative/zero price or km_driven
    df = df[df["price"].apply(lambda x: pd.notna(x))]
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[(df["price"] > 0)]
    if "km_driven" in df:
        df = df[(df["km_driven"] >= 0)]

    # Drop duplicates on a subset of useful keys
    subset_keys = [c for c in ["name", "year",
                               "km_driven", "price"] if c in df.columns]
    if subset_keys:
        df = df.drop_duplicates(subset=subset_keys)

    return df


def load_all_csvs(csv_paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p, keep_default_na=True, na_values=[
                             '', 'NA', 'null', 'NULL', 'nan', 'NaN'])
        except UnicodeDecodeError:
            # some files may be encoded differently
            df = pd.read_csv(p, encoding="latin-1", keep_default_na=True,
                             na_values=['', 'NA', 'null', 'NULL', 'nan', 'NaN'])
        if df.empty:
            continue
        # detect columns and clean
        cols = detect_columns(df)
        df_clean = clean_and_engineer(df, cols)
        df_clean["_source_file"] = os.path.basename(p)
        frames.append(df_clean)
    if not frames:
        raise ValueError("No valid rows loaded from the provided CSVs.")
    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Convert all pd.NA values to np.nan for scikit-learn compatibility
    # This handles the issue where pd.NA causes problems with boolean operations in sklearn
    combined = combined.replace({pd.NA: np.nan})

    # Also ensure object columns with any remaining NA-like values are properly handled
    for col in combined.select_dtypes(include=['object']).columns:
        combined[col] = combined[col].fillna(np.nan)

    # harmonize dtypes (categoricals to string for safety)
    for c in ["fuel", "transmission", "owner", "seller_type", "mileage_unit", "make"]:
        if c in combined.columns:
            combined[c] = combined[c].astype("string")

    return combined


def stratify_bins(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Return bin indices (0..n_bins-1) for approximate stratification on a continuous target.
    Handles duplicates by using rank-based quantiles.
    """
    # add tiny noise to break ties
    y = np.asarray(y).astype(float)
    y = y + np.random.RandomState(42).normal(0, 1e-9, size=y.shape)
    q = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(y, q)
    # ensure uniqueness (np.digitize needs strictly increasing bin edges)
    bins = np.unique(bins)
    # if too few unique edges, fall back to no stratification
    if len(bins) < 3:
        return None
    # rightmost = True to include max
    labels = np.digitize(y, bins[1:-1], right=False)
    return labels


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Handle OneHotEncoder API change: sparse -> sparse_output (>=1.2)
    ohe_kwargs = {"handle_unknown": "ignore"}
    try:
        if version.parse(sklearn.__version__) >= version.parse("1.2"):
            ohe_kwargs["sparse_output"] = False
        else:
            ohe_kwargs["sparse"] = False
    except Exception:
        # Fallback if version parsing fails; prefer new arg
        ohe_kwargs["sparse_output"] = False

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(**ohe_kwargs)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )
    return pre


def get_feature_names(pre: ColumnTransformer) -> List[str]:
    feature_names = []
    # numeric
    num_ct = pre.named_transformers_["num"]
    num_cols = pre.transformers_[0][2]
    feature_names.extend(num_cols)
    # categorical
    cat_ct = pre.named_transformers_["cat"]
    cat_cols = pre.transformers_[1][2]
    ohe: OneHotEncoder = cat_ct.named_steps["ohe"]
    try:
        cat_names = list(ohe.get_feature_names_out(cat_cols))
    except Exception:
        # older sklearn fallback
        cat_names = []
        for col, cats in zip(cat_cols, ohe.categories_):
            for cat in cats:
                cat_names.append(f"{col}_{cat}")
    feature_names.extend(cat_names)
    return feature_names


def main():
    ap = argparse.ArgumentParser(
        description="Load & preprocess vehicle price data")
    ap.add_argument("--csv", action="append",
                    help="Path(s) to CSV file(s). Can repeat --csv")
    ap.add_argument("--dataset_dir",
                    help="Directory containing CSVs to ingest", default=None)
    ap.add_argument("--out", help="Output directory", default="outputs")
    ap.add_argument(
        "--target", help="Target column name (override autodetect)", default=None)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--random_state", type=int, default=42)

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    csv_paths: List[str] = []
    if args.csv:
        csv_paths.extend(args.csv)
    if args.dataset_dir:
        for fname in os.listdir(args.dataset_dir):
            if fname.lower().endswith(".csv"):
                csv_paths.append(os.path.join(args.dataset_dir, fname))
    if not csv_paths:
        raise SystemExit(
            "Please provide at least one CSV via --csv or a folder via --dataset_dir")

    print(f"Loading {len(csv_paths)} CSV(s)...")
    df = load_all_csvs(csv_paths)

    # If user forces target name, rename to 'price'
    if args.target:
        if args.target not in df.columns:
            raise SystemExit(
                f"--target '{args.target}' not found in combined dataframe columns: {list(df.columns)}")
        if args.target != "price":
            df = df.rename(columns={args.target: "price"})

    # Select model features
    numeric_features = [c for c in [
        "km_driven", "mileage_value", "engine_cc", "max_power_bhp", "torque_nm", "torque_rpm", "seats", "age"
    ] if c in df.columns]
    categorical_features = [c for c in [
        "fuel", "transmission", "owner", "seller_type", "mileage_unit", "make"
    ] if c in df.columns]

    # Keep only rows with a finite price
    df = df[pd.to_numeric(df["price"], errors="coerce").notna()]

    # Train/Val/Test split with stratification on binned price
    y_all = df["price"].astype(float).values
    strat = stratify_bins(y_all, n_bins=10)

    test_size = args.test_size
    val_size = args.val_size

    # First split test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        df, y_all, test_size=test_size, random_state=args.random_state, stratify=strat
    ) if strat is not None else train_test_split(
        df, y_all, test_size=test_size, random_state=args.random_state
    )

    # Now split train/val from tmp
    # Adjust val fraction from remaining
    val_fraction = val_size / (1.0 - test_size)
    strat_tmp = stratify_bins(y_tmp, n_bins=10)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction, random_state=args.random_state, stratify=strat_tmp
    ) if strat_tmp is not None else train_test_split(
        X_tmp, y_tmp, test_size=val_fraction, random_state=args.random_state
    )

    # Build preprocessor and fit on train
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Ensure no pd.NA values remain before preprocessing
    # Convert string columns to object dtype with None instead of pd.NA
    train_data = X_train[numeric_features + categorical_features].copy()
    val_data = X_val[numeric_features + categorical_features].copy()
    test_data = X_test[numeric_features + categorical_features].copy()

    # Convert string columns to object and replace pd.NA with None
    for col in categorical_features:
        if col in train_data.columns:
            train_data[col] = train_data[col].astype(
                'object').replace({pd.NA: None})
            val_data[col] = val_data[col].astype(
                'object').replace({pd.NA: None})
            test_data[col] = test_data[col].astype(
                'object').replace({pd.NA: None})

    # Ensure numeric columns have np.nan instead of pd.NA
    for col in numeric_features:
        if col in train_data.columns:
            train_data[col] = train_data[col].replace({pd.NA: np.nan})
            val_data[col] = val_data[col].replace({pd.NA: np.nan})
            test_data[col] = test_data[col].replace({pd.NA: np.nan})

    # Fit on train only
    X_train_proc = preprocessor.fit_transform(train_data)
    X_val_proc = preprocessor.transform(val_data)
    X_test_proc = preprocessor.transform(test_data)

    feature_names = get_feature_names(preprocessor)

    # Persist artifacts
    preproc_path = os.path.join(args.out, "preprocessor.joblib")
    joblib.dump(preprocessor, preproc_path)

    data_blob = {
        "X_train": X_train_proc,
        "y_train": y_train,
        "X_val": X_val_proc,
        "y_val": y_val,
        "X_test": X_test_proc,
        "y_test": y_test,
        "feature_names": feature_names,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "sources": sorted(df.get("_source_file", pd.Series(dtype=str)).dropna().unique().tolist()),
    }

    data_path = os.path.join(args.out, "processed_data.pkl")
    joblib.dump(data_blob, data_path)

    # quick human-readable summary
    summary = []
    summary.append(f"Rows total: {len(df)}")
    summary.append(
        f"Splits: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    summary.append(f"Numeric features: {numeric_features}")
    summary.append(f"Categorical features: {categorical_features}")
    if data_blob["sources"]:
        summary.append(f"Source files: {data_blob['sources']}")
    # basic target stats
    p = df["price"].astype(float)
    summary.append(
        "Price stats: min={:.0f}, p25={:.0f}, median={:.0f}, p75={:.0f}, max={:.0f}".format(
            p.min(), p.quantile(0.25), p.median(), p.quantile(0.75), p.max()
        )
    )

    with open(os.path.join(args.out, "data_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    print("✔ Saved:")
    print(f"  - {preproc_path}")
    print(f"  - {data_path}")
    print(f"  - {os.path.join(args.out, 'data_summary.txt')}")


if __name__ == "__main__":
    main()
