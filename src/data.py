"""Load Adult Income and make the 70/15/15 split."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_adult(spec, path=None):
    """Load Adult as a DataFrame with hyphenated column names and string categoricals.

    Uses `path` (CSV with header) if it exists, otherwise downloads from OpenML.
    Missing values and '?' become the category "missing".
    """
    if path and Path(path).exists():
        df = pd.read_csv(path, skipinitialspace=True)
    else:
        from sklearn.datasets import fetch_openml
        df = fetch_openml("adult", version=2, as_frame=True, parser="auto").frame
    df.columns = [c.strip().replace("_", "-") for c in df.columns]
    if "class" in df.columns:
        df = df.rename(columns={"class": spec["target"]})

    num_cols = {c for c, v in spec["columns"].items() if v.get("type") == "numeric"}
    for c in df.columns:
        if c in num_cols:
            df[c] = pd.to_numeric(df[c])
        else:
            s = df[c].astype(object)
            s = s.where(s.notna(), "missing").astype(str).str.strip()
            df[c] = s.replace("?", "missing")

    t = spec["target"]
    df[t] = np.where(df[t].str.contains(">50K"), ">50K", "<=50K")

    keep = [c for c, v in spec["columns"].items() if not v.get("drop")]
    return df[keep].reset_index(drop=True)


def split_data(df, target, seed=0):
    """70/15/15 stratified split with a fixed seed."""
    train, rest = train_test_split(df, test_size=0.30, random_state=seed, stratify=df[target])
    val, test = train_test_split(rest, test_size=0.50, random_state=seed, stratify=rest[target])
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))
