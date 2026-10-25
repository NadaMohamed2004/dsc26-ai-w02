"""Data audit for the Rawaj Pay transfers extract."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MIN_LABELLED = 20
AUC_HIGH = 0.98
AUC_LOW = 0.02


def _is_blank(value) -> bool:
    return isinstance(value, str) and value.strip() == ""


def _missing_mask(s: pd.Series) -> np.ndarray:
    """True where NaN/None/NA, or a string that is empty or only whitespace."""
    mask = s.isna().to_numpy(dtype=bool).copy()
    dtype = s.dtype
    if not (
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
    ):
        blank = np.fromiter((_is_blank(v) for v in s.astype(object)), dtype=bool, count=len(s))
        mask |= blank
    return mask


def _sorted_list(values) -> list:
    values = list(values)
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def _parse_times(values: pd.Series) -> pd.Series:
    """Parse non-missing time values; unparseable ones become NaT."""
    if len(values) == 0:
        return pd.Series([], dtype="datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return values
    strings = values.astype(str).str.strip()
    try:
        parsed = pd.to_datetime(strings, format="ISO8601", errors="coerce")
        if not pd.api.types.is_datetime64_any_dtype(parsed.dtype):
            raise ValueError("mixed time zones")
    except (ValueError, TypeError):
        parsed = pd.to_datetime(strings, format="ISO8601", errors="coerce", utc=True)
    return parsed


def _numeric_view(s: pd.Series, miss: np.ndarray) -> np.ndarray | None:
    """Non-missing values as floats if the column is numeric or boolean, else None."""
    dtype = s.dtype
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        pass
    elif pd.api.types.is_object_dtype(dtype):
        numeric_types = (bool, int, float, np.bool_, np.integer, np.floating)
        if not all(isinstance(v, numeric_types) for v in s[~miss]):
            return None
    else:
        return None
    try:
        return s[~miss].astype(float).to_numpy()
    except (TypeError, ValueError):
        return None


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    if not np.isfinite(x).all():  # AUC only depends on ranks
        x = pd.Series(x).rank().to_numpy()
    return float(roc_auc_score(y, x))


def _extreme(auc: float) -> bool:
    return auc >= AUC_HIGH or auc <= AUC_LOW


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    for name in (target, id_col, time_col):
        if name not in df.columns:
            raise KeyError(name)

    n_rows = int(len(df))
    masks = {col: _missing_mask(df[col]) for col in df.columns}

    # missing fractions
    missing = {col: (float(m.mean()) if n_rows else 0.0) for col, m in masks.items()}

    # duplicate ids (missing ids ignored)
    ids = df[id_col][~masks[id_col]]
    counts = ids.value_counts()
    duplicate_ids = _sorted_list(counts[counts > 1].index.tolist())

    # duplicate rows
    try:
        duplicate_rows = int(df.duplicated().sum())
    except TypeError:  # unhashable cells
        duplicate_rows = int(df.astype(str).duplicated().sum())

    # constant columns (target excluded, missing ignored)
    constant_columns = _sorted_list(
        col
        for col in df.columns
        if col != target and df[col][~masks[col]].nunique() <= 1
    )

    # labels
    labelled = ~masks[target]
    n_labelled = int(labelled.sum())
    unlabelled_rows = n_rows - n_labelled
    y_num = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    y_bin = (y_num == 1).astype(int)
    positive_rate = float(y_bin[labelled].sum() / n_labelled) if n_labelled else 0.0

    # timestamps
    time_missing = masks[time_col]
    parsed = _parse_times(df[time_col][~time_missing])
    bad_timestamps = int(parsed.isna().sum())
    valid = parsed.dropna()
    if len(valid):
        time_range = [valid.min().isoformat(), valid.max().isoformat()]
    else:
        time_range = [None, None]

    # leakage suspects
    leakage_suspects: list = []
    y_lab = y_bin[labelled]
    if n_labelled >= MIN_LABELLED and y_lab.any() and not y_lab.all():
        for col in df.columns:
            if col in (target, id_col, time_col):
                continue
            s_lab = df[col][labelled]
            miss_lab = masks[col][labelled]
            n_distinct = s_lab[~miss_lab].nunique() + int(miss_lab.any())
            if n_distinct <= 1:  # same value in every row (missing counts as a value)
                continue

            suspect = False
            # (a) value leak: numeric/boolean column that almost separates the classes
            x = _numeric_view(s_lab, miss_lab)
            if x is not None and len(x) >= MIN_LABELLED:
                y_x = y_lab[~miss_lab]
                if y_x.min() != y_x.max():
                    suspect = _extreme(_auc(y_x, x))
            # (b) missingness leak: blank-ness almost separates the classes
            if not suspect and miss_lab.any() and not miss_lab.all():
                suspect = _extreme(_auc(y_lab, miss_lab.astype(int)))

            if suspect:
                leakage_suspects.append(col)

    return {
        "n_rows": n_rows,
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": int(unlabelled_rows),
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": _sorted_list(leakage_suspects),
    }
