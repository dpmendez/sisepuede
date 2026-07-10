"""
sisepuede/calibration/_training_set.py

Build (X, Y) training matrices for v3 surrogate training out of a
SensitivityResult.

The v3 surrogate is trained on **raw per-technology SSP outputs** (e.g.
`nemomod_entc_annual_production_by_technology_pp_coal`), not on IEA-
aggregated values. Aggregation to IEA cells happens downstream, at
inference time, via a sparse crosswalk aggregation matrix `A`:

    y_ssp  = surrogate.predict(f[None, :])[0]     # shape (T_ssp,)
    y_iea  = A @ y_ssp                             # shape (T_iea,)  in TJ

This module builds both pieces:
- `build_training_matrices(result, year_target, ssp_columns) -> (X, Y_ssp)`
  extracts raw per-tech outputs at a single year from
  `result.model_outputs`.
- `crosswalk_ssp_columns_for_iea_targets(iea_target_rows, crosswalk) ->
  (ssp_columns, A)` returns the ordered union of raw SSP fields the
  crosswalk maps to the requested IEA rows, plus the aggregation matrix
  that folds per-tech predictions back into IEA-comparable TJ values
  (unit conversion baked in).

Other helpers here are for reporting / optimizer boundary:
- `baseline_targets(...)` returns IEA-aggregated SSP baselines from the
  crosswalk comparison (useful for diagnostics; not for training).
- `iea_targets(...)` returns the IEA-side observed values (the `b_IEA`
  vector the optimizer matches against).
- `split_train_dev_test(...)` produces the three-way ML split.

This module is intentionally pure-pandas: no SISEPUEDE imports, no model
runs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Avoid importing the heavy sensitivity module here -- a string forward ref
# keeps this file standalone for callers that only need the (X, Y) pivot
# (e.g. unit tests, notebook exploration).
SensitivityResult = "sisepuede.calibration.sensitivity.SensitivityResult"
IEACrosswalk      = "sisepuede.calibration.iea_crosswalk.IEACrosswalk"

# The SSP-side value column emitted by IEACrosswalk.build_comparison().
# Used by `baseline_targets` and `iea_targets` (IEA-aggregated diagnostics
# only; the raw training data goes through `result.model_outputs`).
DEFAULT_VALUE_COL = "value_sisepuede_tj"


def build_training_matrices(
    result,
    year_target:  int,
    ssp_columns:  List[str],
    fill_missing: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract (X, Y_ssp) from a SensitivityResult.

    Y_ssp columns are **raw per-technology SSP output fields** (e.g.
    `nemomod_entc_annual_production_by_technology_pp_coal`), pulled from
    `result.model_outputs` at `year_target`. Callers typically obtain
    `ssp_columns` from `crosswalk_ssp_columns_for_iea_targets()` so the
    downstream aggregation matrix aligns.

    Parameters
    ----------
    result : SensitivityResult
        Output of SensitivityRunner.run_lhs. Must carry `input_samples`
        (X-shaped, scale factors) and `model_outputs` (long, stacked).
    year_target : int
        Calendar year to extract Y at. Exactly one row per run_index at
        this year is expected.
    ssp_columns : List[str]
        Ordered list of raw SSP output column names to use as training
        targets. Order is preserved as Y's column order; downstream
        stacking with the crosswalk aggregation matrix depends on it.
    fill_missing : float
        Value to substitute when a (run_index, ssp_column) cell has no
        data. Defaults to 0.0 (consistent with NemoMod returning zero
        for techs the country doesn't operate). Pass np.nan to propagate.

    Returns
    -------
    X : pd.DataFrame, shape (N_runs, N_knobs)
        Input scale factors per LHS run. Copy of `result.input_samples`
        (safe for callers to mutate).
    Y : pd.DataFrame, shape (N_runs, len(ssp_columns))
        Raw SSP outputs at `year_target` for each column in
        `ssp_columns`. Index is `run_index`, aligned to X. Column labels
        are plain SSP field names (strings, not tuples).

    Raises
    ------
    KeyError
        If `result.model_outputs` lacks the required keys.
    ValueError
        If `ssp_columns` is empty.
    """
    if not ssp_columns:
        raise ValueError(
            "build_training_matrices: `ssp_columns` must be a non-empty "
            "list of SSP output field names."
        )

    df_out = result.model_outputs
    required = {"run_index", "year"}
    missing = required - set(df_out.columns)
    if missing:
        raise KeyError(
            f"build_training_matrices: SensitivityResult.model_outputs is "
            f"missing required columns: {sorted(missing)}."
        )

    X = result.input_samples.copy()

    # Filter to the requested year and set run_index as the index so we
    # can reindex against X.
    df_y = (
        df_out[df_out["year"] == year_target]
        .set_index("run_index")
    )

    # Assemble Y, handling absent SSP columns and missing runs both with
    # `fill_missing`. Column order matches `ssp_columns` verbatim.
    y_data: Dict[str, np.ndarray] = {}
    n_runs = len(X.index)
    for col in ssp_columns:
        if col in df_y.columns:
            y_data[col] = (
                df_y[col].reindex(X.index).fillna(fill_missing).to_numpy(dtype=float)
            )
        else:
            y_data[col] = np.full(n_runs, fill_missing, dtype=float)

    Y = pd.DataFrame(y_data, index=X.index, columns=ssp_columns)
    return X, Y


def crosswalk_ssp_columns_for_iea_targets(
    iea_target_rows: List[Tuple[str, str]],
    crosswalk,
) -> Tuple[List[str], np.ndarray]:
    """Build the raw SSP column list + aggregation matrix for a set of
    IEA target rows.

    The surrogate is trained on raw per-tech outputs; the optimizer
    compares aggregated IEA values against IEA observations. The
    aggregation happens via a sparse matrix `A`:

        y_ssp = surrogate.predict(f[None, :])[0]   # (T_ssp,)  native units
        y_iea = A @ y_ssp                          # (T_iea,)  in TJ

    Each row of `A` corresponds to one IEA target row and contains the
    per-tech `unit_conversion_to_tj` factor for the ssp columns it
    aggregates; other entries are zero.

    Parameters
    ----------
    iea_target_rows : List[Tuple[str, str]]
        Ordered list of (iea_balance_code, iea_product_code) pairs. The
        matrix's row order matches this list; downstream stacking with
        the b_IEA vector must use the same order.
    crosswalk : IEACrosswalk
        Provides `get_crosswalk_entry(balance, product) -> dict` where
        the dict carries `ssp_fields` (List[str]) and
        `unit_conversion_to_tj` (float).

    Returns
    -------
    ssp_columns : List[str]
        Deterministic union (sorted) of all SSP fields any of the
        `iea_target_rows` needs. This becomes the surrogate's target set.
    A : np.ndarray, shape (len(iea_target_rows), len(ssp_columns))
        Aggregation matrix. `A[i, j] = unit_conversion_to_tj_i` if
        `ssp_columns[j]` is in `iea_target_rows[i]`'s ssp_fields, else 0.
    """
    if not iea_target_rows:
        raise ValueError(
            "crosswalk_ssp_columns_for_iea_targets: `iea_target_rows` "
            "must be a non-empty list."
        )

    per_row_fields: List[List[str]] = []
    per_row_conv:   List[float]     = []
    all_fields: set = set()

    for bal, prod in iea_target_rows:
        entry = crosswalk.get_crosswalk_entry(bal, prod)
        if entry is None:
            per_row_fields.append([])
            per_row_conv.append(1.0)
            continue
        fields = list(entry.get("ssp_fields") or [])
        conv   = float(entry.get("unit_conversion_to_tj", 1.0))
        per_row_fields.append(fields)
        per_row_conv.append(conv)
        all_fields.update(fields)

    ssp_columns = sorted(all_fields)
    col_idx     = {c: j for j, c in enumerate(ssp_columns)}

    n_iea = len(iea_target_rows)
    n_ssp = len(ssp_columns)
    A = np.zeros((n_iea, n_ssp), dtype=float)
    for i, (fields, conv) in enumerate(zip(per_row_fields, per_row_conv)):
        for f in fields:
            if f in col_idx:
                A[i, col_idx[f]] = conv

    return ssp_columns, A


def baseline_targets(
    result,
    year_target: int,
    target_rows: List[Tuple[str, str]],
    value_col: str = DEFAULT_VALUE_COL,
    fill_missing: float = 0.0,
) -> pd.Series:
    """Extract baseline SSP-side values at `year_target` for the same target
    rows, as a Series indexed by (balance, product).

    Useful for the surrogate's envelope/baseline-comparison logic and as the
    Phase-1-fixed reference `g(f_default)` in the SQP outer loop.

    Parameters mirror `build_training_matrices`. Returns a Series of length
    len(target_rows), preserving the requested target order.
    """
    df_base = result.baseline_iea_comparison
    target_keys = pd.MultiIndex.from_tuples(target_rows,
                                            names=["iea_balance_code",
                                                   "iea_product_code"])

    df_b = df_base[df_base["year"] == year_target].copy()
    df_b_idx = pd.MultiIndex.from_frame(
        df_b[["iea_balance_code", "iea_product_code"]],
    )
    df_b = df_b[df_b_idx.isin(target_keys)]

    s = (
        df_b.groupby(["iea_balance_code", "iea_product_code"])[value_col]
        .sum()
        .reindex(target_keys)
        .fillna(fill_missing)
    )
    s.name = value_col
    return s


def split_train_dev_test(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    fractions: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Split aligned (X, Y) DataFrames into train / dev / test partitions.

    Parameters
    ----------
    X : DataFrame, shape (N, N_vars)
        Input scale factors (typically from
        `build_training_matrices`).
    Y : DataFrame, shape (N, N_targets)
        Aligned targets, index must equal X.index.
    fractions : (train, dev, test)
        Non-negative fractions that must sum to <= 1.0. If they sum to
        less than 1, the leftover rows are discarded (useful for
        subsampling / learning-curve experiments).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    Dict[str, (X_split, Y_split)]
        Keys "train", "dev", "test". Preserves the columns of X and Y
        for each split; index is a fresh integer range (permuted).
    """
    if list(X.index) != list(Y.index):
        raise ValueError(
            "split_train_dev_test: X and Y must have identical indices."
        )
    if any(f < 0 for f in fractions):
        raise ValueError("split_train_dev_test: fractions must be non-negative.")
    total = sum(fractions)
    if total > 1.0 + 1e-9:
        raise ValueError(
            f"split_train_dev_test: fractions must sum to <= 1.0 (got {total})."
        )

    N = X.shape[0]
    if N == 0:
        raise ValueError("split_train_dev_test: empty (X, Y).")

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(N)

    n_train = int(round(N * fractions[0]))
    n_dev   = int(round(N * fractions[1]))
    n_test  = int(round(N * fractions[2]))

    idx_train = perm[:n_train]
    idx_dev   = perm[n_train : n_train + n_dev]
    idx_test  = perm[n_train + n_dev : n_train + n_dev + n_test]

    def _slice(idx: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (
            X.iloc[idx].reset_index(drop=True),
            Y.iloc[idx].reset_index(drop=True),
        )

    return {
        "train": _slice(idx_train),
        "dev":   _slice(idx_dev),
        "test":  _slice(idx_test),
    }


def iea_targets(
    result,
    year_target: int,
    target_rows: List[Tuple[str, str]],
    iea_value_col: str = "value_iea_tj",
    fill_missing: float = np.nan,
) -> pd.Series:
    """Extract the IEA-side observed values at `year_target` for the target
    rows, as a Series indexed by (balance, product).

    This is the `b_surr` vector the v3 optimizer compares against:
        residual = ssp_predicted - iea_observed
    Cell values that are missing from the IEA data (countries that don't
    report a particular fuel) come back as NaN by default so the optimizer
    can skip them.
    """
    df_comp = result.iea_comparison
    target_keys = pd.MultiIndex.from_tuples(target_rows,
                                            names=["iea_balance_code",
                                                   "iea_product_code"])

    df_i = df_comp[df_comp["year"] == year_target].copy()
    df_i_idx = pd.MultiIndex.from_frame(
        df_i[["iea_balance_code", "iea_product_code"]],
    )
    df_i = df_i[df_i_idx.isin(target_keys)]

    # The IEA value is constant across run_index for a fixed (year, target),
    # so first() picks the representative value.
    s = (
        df_i.groupby(["iea_balance_code", "iea_product_code"])[iea_value_col]
        .first()
        .reindex(target_keys)
    )
    if not np.isnan(fill_missing):
        s = s.fillna(fill_missing)
    s.name = iea_value_col
    return s
