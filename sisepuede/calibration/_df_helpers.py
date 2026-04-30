"""
sisepuede/calibration/_df_helpers.py

Tiny shared helpers for reading SISEPUEDE input/output DataFrames at a
specific time_period. Kept minimal on purpose -- one function per
distinct read pattern.
"""

from __future__ import annotations

import pandas as pd
from typing import List


def sum_fields_at_time_period(
    df: pd.DataFrame,
    time_period: int,
    fields: List[str],
) -> float:
    """Sum the values of ``fields`` in ``df`` at the row matching ``time_period``.

    Returns 0.0 if ``time_period`` is not present, if ``fields`` is empty,
    or if none of the named fields exist as columns in ``df``. Missing
    fields are silently dropped (not an error) so callers can pass the
    full set returned by the crosswalk without pre-filtering.
    """
    available = [f for f in fields if f in df.columns]
    if not available:
        return 0.0

    mask = df["time_period"] == time_period
    if not mask.any():
        return 0.0

    return float(df.loc[mask, available].sum(axis=1).iloc[0])


def attach_year_column(
    df_out: pd.DataFrame,
    df_with_year: pd.DataFrame,
    *,
    raise_if_missing_source: bool = False,
) -> pd.DataFrame:
    """Ensure ``df_out`` has a ``year`` column, joining from ``df_with_year``.

    SISEPUEDE model output carries ``time_period`` but not ``year``; downstream
    consumers (e.g. IEACrosswalk.aggregate_sisepuede) need the calendar year.
    This helper takes the ``time_period -> year`` map from ``df_with_year``
    and left-merges it into ``df_out``.

    Parameters
    ----------
    df_out : pd.DataFrame
        Frame to extend. Returned unchanged if it already has ``year``.
    df_with_year : pd.DataFrame
        Source of the ``time_period -> year`` mapping (e.g. the calibration
        input frame or a cached baseline). Must contain ``time_period`` and,
        if not already supplied on ``df_out``, ``year``.
    raise_if_missing_source : bool, default False
        When True, raises ``ValueError`` if ``df_with_year`` lacks ``year``.
        When False, returns ``df_out`` unchanged in that case (caller is
        expected to tolerate the missing column).
    """
    if "year" in df_out.columns:
        return df_out

    if "year" not in df_with_year.columns:
        if raise_if_missing_source:
            raise ValueError(
                "Cannot attach 'year' to model output: source DataFrame does "
                "not have a 'year' column. Add it (mapping time_period -> "
                "calendar year) before calling."
            )
        return df_out

    year_map = df_with_year[["time_period", "year"]].drop_duplicates()
    return df_out.merge(year_map, on="time_period", how="left")
