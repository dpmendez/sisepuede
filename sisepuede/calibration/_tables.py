"""
sisepuede/calibration/_tables.py

LaTeX table builders for calibration summary reports.

Three formatters operate on the comparison frames produced by
``Calibrator.evaluate()`` and the long-form knob frame produced by
``Calibrator.summarize_knobs()``:

    build_error_table        — proportional deviation per (balance, product)
                               at a target year, plus an overall unweighted
                               mean.
    build_improvement_table  — before/after errors with a colour-graded ratio
                               column (green = improvement, red = regression).
    build_knob_tables        — one table per (balance, product); rows
                               sub-grouped by simplex_group_id and the
                               ``% change`` column shaded by magnitude
                               (yellow > 25%, orange > 50%, red > 75% by default).

The emitted LaTeX assumes the consuming document loads:

    \\usepackage{booktabs}
    \\usepackage[table]{xcolor}
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#   Constants / colour scheme
# ---------------------------------------------------------------------------

# Improvement table — ratio = err_after / err_before
# (lower is better; > 1 means calibration made the pair worse).
_RATIO_BUCKETS = [
    (0.5,        "green!60"),
    (0.9,        "green!25"),
    (1.1,        None),       # neutral band — no fill
    (1.5,        "red!25"),
    (float("inf"), "red!60"),
]

# Knob table — colour scale on |pct_change|.
# Default thresholds (yellow, orange, red).
_KNOB_DEFAULT_COLOURS = ("yellow!40", "orange!40", "red!50")


# ---------------------------------------------------------------------------
#   Small LaTeX helpers
# ---------------------------------------------------------------------------

def _tex_escape(s: str) -> str:
    """Minimal LaTeX escape for column names / labels seen in this project."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _fmt_pct(x: float, decimals: int = 1) -> str:
    """Format a fraction-of-1 number as a percentage with one decimal."""
    if pd.isna(x) or not np.isfinite(x):
        return "--"
    return f"{x * 100:.{decimals}f}\\%"


def _fmt_signed_pp(x: float, decimals: int = 1) -> str:
    """Format a fraction-of-1 number as signed percentage points (no unit)."""
    if pd.isna(x) or not np.isfinite(x):
        return "--"
    return f"{x * 100:+.{decimals}f}"


def _fmt_signed_pct_already_pct(x: float, decimals: int = 1) -> str:
    """Format a number already in percent units as a signed percentage."""
    if pd.isna(x) or not np.isfinite(x):
        return "--"
    return f"{x:+.{decimals}f}\\%"


def _fmt_num(x: float, decimals: int = 4) -> str:
    if pd.isna(x) or not np.isfinite(x):
        return "--"
    return f"{x:.{decimals}g}"


def _ratio_colour(r: float) -> Optional[str]:
    if pd.isna(r) or not np.isfinite(r):
        return None
    for upper, colour in _RATIO_BUCKETS:
        if r <= upper:
            return colour
    return None


def _knob_colour(pct: float, thresholds, colours) -> Optional[str]:
    if pd.isna(pct) or not np.isfinite(pct):
        return None
    a = abs(pct)
    t1, t2, t3 = thresholds
    c1, c2, c3 = colours
    if a >= t3:
        return c3
    if a >= t2:
        return c2
    if a >= t1:
        return c1
    return None


def _wrap_cell(value: str, colour: Optional[str]) -> str:
    if colour is None:
        return value
    return f"\\cellcolor{{{colour}}}{value}"


def _write_if_requested(tex: str, out_path: Optional[str]) -> None:
    if out_path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(tex)


# ---------------------------------------------------------------------------
#   Shared comparison-frame plumbing
# ---------------------------------------------------------------------------

def _pair_errors(df_comp: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return |rel_err| per (balance, product) at a given year.

    Equivalent to ``proportional_deviation`` with ``signed=False`` applied to
    each row of the comparison frame.
    """
    d = df_comp.loc[df_comp["year"] == year].copy()
    d = d.dropna(subset=["value_iea_tj", "value_sisepuede_tj"])
    d = d.loc[d["value_iea_tj"] != 0].copy()
    d["err"] = (d["value_sisepuede_tj"] - d["value_iea_tj"]).abs() / d["value_iea_tj"]
    return d[["iea_balance_code", "iea_product_code", "err"]]


def _merged_errors(
    df_comp_baseline: pd.DataFrame,
    df_comp_calibrated: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    eb = _pair_errors(df_comp_baseline, year).rename(columns={"err": "err_before"})
    ea = _pair_errors(df_comp_calibrated, year).rename(columns={"err": "err_after"})
    df = eb.merge(
        ea,
        on=["iea_balance_code", "iea_product_code"],
        how="outer",
    )
    return df.sort_values(["iea_balance_code", "iea_product_code"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
#   1) Error table
# ---------------------------------------------------------------------------

def build_error_table(
    df_comp_baseline: pd.DataFrame,
    df_comp_calibrated: pd.DataFrame,
    country: str,
    target_year: int,
    out_path: Optional[str] = None,
) -> str:
    """LaTeX table of per-(balance, product) absolute proportional deviation
    before/after calibration, with an unweighted-mean footer row.
    """
    df = _merged_errors(df_comp_baseline, df_comp_calibrated, target_year)

    overall_before = df["err_before"].mean(skipna=True)
    overall_after  = df["err_after"].mean(skipna=True)

    country_tex = _tex_escape(country)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{Calibration error per IEA (balance, product) at "
        f"{target_year} for {country_tex}. Error is the absolute proportional "
        f"deviation $|y_{{\\mathrm{{SSP}}}} - y_{{\\mathrm{{IEA}}}}|/y_{{\\mathrm{{IEA}}}}$.}}",
        f"\\label{{tab:cal_error_{_sanitize_label(country)}_{target_year}}}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"IEA balance & IEA product & Error before & Error after \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"{_tex_escape(row['iea_balance_code'])} & "
            f"{_tex_escape(row['iea_product_code'])} & "
            f"{_fmt_pct(row['err_before'])} & "
            f"{_fmt_pct(row['err_after'])} \\\\"
        )

    lines += [
        r"\midrule",
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Overall (unweighted mean)}}}} & "
        f"\\textbf{{{_fmt_pct(overall_before)}}} & "
        f"\\textbf{{{_fmt_pct(overall_after)}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    _write_if_requested(tex, out_path)
    return tex


# ---------------------------------------------------------------------------
#   2) Improvement table
# ---------------------------------------------------------------------------

def build_improvement_table(
    df_comp_baseline: pd.DataFrame,
    df_comp_calibrated: pd.DataFrame,
    country: str,
    target_year: int,
    out_path: Optional[str] = None,
) -> str:
    """LaTeX table of error before/after, the ratio after/before (colour-graded),
    and the signed change in percentage points.
    """
    df = _merged_errors(df_comp_baseline, df_comp_calibrated, target_year)
    df["ratio"] = df["err_after"] / df["err_before"]
    df["delta_pp"] = df["err_after"] - df["err_before"]

    country_tex = _tex_escape(country)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{Calibration improvement at {target_year} for {country_tex}. "
        f"Ratio $= \\mathrm{{err}}_{{\\mathrm{{after}}}}/\\mathrm{{err}}_{{\\mathrm{{before}}}}$ "
        r"(green = improvement, red = regression).}",
        f"\\label{{tab:cal_improvement_{_sanitize_label(country)}_{target_year}}}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"IEA balance & IEA product & Err. before & Err. after & Ratio & $\Delta$ (pp) \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        ratio_cell = _wrap_cell(_fmt_num(row["ratio"], 2), _ratio_colour(row["ratio"]))
        lines.append(
            f"{_tex_escape(row['iea_balance_code'])} & "
            f"{_tex_escape(row['iea_product_code'])} & "
            f"{_fmt_pct(row['err_before'])} & "
            f"{_fmt_pct(row['err_after'])} & "
            f"{ratio_cell} & "
            f"{_fmt_signed_pp(row['delta_pp'])} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    _write_if_requested(tex, out_path)
    return tex


# ---------------------------------------------------------------------------
#   3) Knob tables (one per (balance, product))
# ---------------------------------------------------------------------------

def build_knob_tables(
    df_knobs: pd.DataFrame,
    country: str,
    target_year: int,
    pct_thresholds: Tuple[float, float, float] = (25.0, 50.0, 75.0),
    colours: Tuple[str, str, str] = _KNOB_DEFAULT_COLOURS,
    out_dir: Optional[str] = None,
) -> Dict[Tuple[str, str], str]:
    """One LaTeX table per (iea_balance, iea_product) describing how the
    calibration knobs that target that pair changed.

    Rows are sub-grouped by ``simplex_group_id`` (unconstrained variables —
    phase-1 scalar knobs — get their own block labelled ``Unconstrained``).
    The ``% change`` column is shaded when ``|pct_change|`` crosses
    ``pct_thresholds`` (default 25 / 50 / 75).

    Returns
    -------
    dict
        ``{(balance, product): latex_string}``. If ``out_dir`` is provided
        each table is also written to ``{out_dir}/knobs_{balance}_{product}.tex``.
    """
    out: Dict[Tuple[str, str], str] = {}
    if df_knobs.empty:
        return out

    country_tex = _tex_escape(country)
    country_lbl = _sanitize_label(country)

    for (bal, prod), df_pair in df_knobs.groupby(
        ["iea_balance", "iea_product"], sort=True
    ):
        tex = _build_single_knob_table(
            df_pair=df_pair,
            balance=bal,
            product=prod,
            country_tex=country_tex,
            country_lbl=country_lbl,
            target_year=target_year,
            pct_thresholds=pct_thresholds,
            colours=colours,
        )
        out[(bal, prod)] = tex

        if out_dir is not None:
            fname = f"knobs_{_sanitize_label(bal)}_{_sanitize_label(prod)}.tex"
            _write_if_requested(tex, os.path.join(out_dir, fname))

    return out


def _build_single_knob_table(
    df_pair: pd.DataFrame,
    balance: str,
    product: str,
    country_tex: str,
    country_lbl: str,
    target_year: int,
    pct_thresholds: Tuple[float, float, float],
    colours: Tuple[str, str, str],
) -> str:
    bal_tex  = _tex_escape(balance)
    prod_tex = _tex_escape(product)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{Calibration knob changes for ({bal_tex}, {prod_tex}) "
        f"at {target_year} for {country_tex}. Cells shaded when "
        f"$|\\Delta| \\geq {pct_thresholds[0]:.0f}/{pct_thresholds[1]:.0f}/{pct_thresholds[2]:.0f}\\%$.}}",
        f"\\label{{tab:cal_knobs_{_sanitize_label(balance)}_{_sanitize_label(product)}_{country_lbl}_{target_year}}}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Variable & Simplex group & Initial & Final & \% change \\",
        r"\midrule",
    ]

    # Stable sub-group order: numeric simplex IDs first (ascending), then
    # unconstrained (NaN) last.
    df_pair = df_pair.copy()
    df_pair["_sort_key"] = df_pair["simplex_group_id"].apply(
        lambda v: (1, -1) if pd.isna(v) else (0, int(v))
    )
    df_pair = df_pair.sort_values(["_sort_key", "variable"], kind="stable")

    last_block = object()
    for _, row in df_pair.iterrows():
        gid = row["simplex_group_id"]
        block = "unconstrained" if pd.isna(gid) else int(gid)
        if block != last_block:
            label = "Unconstrained" if block == "unconstrained" else f"Simplex group {block}"
            lines.append(
                f"\\multicolumn{{5}}{{l}}{{\\textit{{{label}}}}} \\\\"
            )
            last_block = block

        pct_cell = _wrap_cell(
            _fmt_signed_pct_already_pct(row["pct_change"]),
            _knob_colour(row["pct_change"], pct_thresholds, colours),
        )
        gid_str = "--" if pd.isna(gid) else str(int(gid))

        lines.append(
            f"{_tex_escape(row['variable'])} & "
            f"{gid_str} & "
            f"{_fmt_num(row['initial'])} & "
            f"{_fmt_num(row['final'])} & "
            f"{pct_cell} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
#   Label sanitization
# ---------------------------------------------------------------------------

def _sanitize_label(s: str) -> str:
    """Make a LaTeX \\label-safe slug (alnum + underscore only)."""
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in str(s).lower())
