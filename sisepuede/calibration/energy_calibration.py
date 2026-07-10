"""
sisepuede/calibration/energy_calibration.py

End-to-end energy calibration pipeline.

Wraps the steps spelled out in ``energy_calibration.ipynb``:

    1. Load SISEPUEDE inputs, run AFOLU + IPPU, merge their outputs into the
       energy-input frame.
    2. Build / load the IEA-SISEPUEDE crosswalk and load IEA data for the
       requested country.
    3. Run the baseline energy model and build the baseline IEA comparison.
    4. Build the calibration plan and run the two-phase Calibrator.
    5. Re-run the model on calibrated inputs, build the post-calibration
       comparison, save plots, summary tables, and the calibrated input CSV.

The notebook is exploratory; this module is the user-callable equivalent
(no display() calls, no manual inspection cells).
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import List, Optional, Tuple

_REPO_ROOT = "/Users/dianamendez/feature-energy-calibration"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

from sisepuede.manager.sisepuede_file_structure          import SISEPUEDEFileStructure
from sisepuede.models.afolu                              import AFOLU
from sisepuede.models.energy_consumption                 import EnergyConsumption
from sisepuede.models.ippu                               import IPPU

from sisepuede.calibration.build_energy_calibration_plan import build_energy_calibration_plan
from sisepuede.calibration.build_iea_energy_crosswalk    import IEACrosswalkBuilder
from sisepuede.calibration.calibrator                    import Calibrator
from sisepuede.calibration.iea_crosswalk                 import IEACrosswalk
from sisepuede.calibration.iea_data_loader               import IEADataLoader

from sisepuede.calibration._plotting import (
    plot_baseline_discrepancy_bar,
    plot_before_after_bar,
    plot_before_after_discrepancy_bar,
    plot_before_after_time_series,
)
from sisepuede.calibration._tables import (
    build_improvement_table,
    build_knob_tables,
    build_values_table,
)


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
#   Small helpers
# ---------------------------------------------------------------------------

def _vprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _attach_year(df: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Attach a 'year' column if missing, then move it to the front."""
    if "year" not in df.columns:
        df = df.copy()
        df["year"] = range(start_year, start_year + len(df))
    col = df.pop("year")
    df.insert(0, "year", col)
    return df


def _changed_columns(
    df_baseline: pd.DataFrame,
    df_calibrated: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """Return mean scale (calibrated / baseline) for every input column that
    actually changed during calibration."""
    rows = []
    for col in columns:
        if col not in df_baseline.columns or col not in df_calibrated.columns:
            continue
        if not np.issubdtype(df_calibrated[col].dtype, np.number):
            continue
        orig = df_baseline[col].values
        new  = df_calibrated[col].values
        if np.allclose(orig, new, equal_nan=True):
            continue
        ratio = np.nanmean(new / np.where(orig == 0, np.nan, orig))
        rows.append({"column": col, "mean_scale_applied": round(float(ratio), 4)})
    return pd.DataFrame(rows)


def _sectors_with_fuel_mix(df_comp: pd.DataFrame) -> List[str]:
    """Return sectors that have more than one fuel product mapped (i.e. a
    non-trivial fuel mix worth plotting)."""
    grp = (
        df_comp[["iea_balance_code", "iea_product_code"]]
        .drop_duplicates()
        .groupby("iea_balance_code")["iea_product_code"]
        .nunique()
    )
    return sorted(grp[grp > 1].index.tolist())


# ---------------------------------------------------------------------------
#   Pipeline stages
# ---------------------------------------------------------------------------

def _load_inputs(
    path_sisepuede_input: str,
    start_year: int,
    iea_year_limit: int,
    verbose: bool,
) -> pd.DataFrame:
    df_input = pd.read_csv(path_sisepuede_input)
    df_input = _attach_year(df_input, start_year)
    df_input = df_input.loc[df_input["year"] <= iea_year_limit].reset_index(drop=True)

    _vprint(verbose, f"Input SSP dataframe shape: {df_input.shape}")
    _vprint(verbose, f"Years: {df_input['year'].min()}–{df_input['year'].max()}\n")
    return df_input


def _build_energy_input_frame(
    df_input: pd.DataFrame,
    model_attributes,
    verbose: bool,
) -> pd.DataFrame:
    """Run AFOLU + IPPU, merge their outputs, fill any residual NaNs."""
    model_afolu = AFOLU(model_attributes)
    model_ippu  = IPPU(model_attributes)

    df_out_afolu = model_afolu(df_input)
    df_out_ippu  = model_ippu(df_input)

    df_input_energy = (
        df_input
        .merge(df_out_afolu, on="time_period", how="left", suffixes=("", "_afolu"))
        .merge(df_out_ippu,  on="time_period", how="left", suffixes=("", "_ippu"))
    )

    nan_count = int(df_input_energy.isna().sum().sum())
    if nan_count:
        df_input_energy = df_input_energy.ffill().bfill().fillna(0.0)
        _vprint(verbose, f"Filled {nan_count} NaN values in df_input_energy.")

    _vprint(verbose, f"df_input_energy shape: {df_input_energy.shape}\n")
    return df_input_energy


def _run_energy_model(
    model_energycon: EnergyConsumption,
    df_input_energy: pd.DataFrame,
    label: str,
    verbose: bool,
) -> pd.DataFrame:
    _vprint(verbose, f"\nRunning {label} model...", end=" ", flush=True)
    df_out = model_energycon(df_input_energy)
    df_out = df_out.merge(
        df_input_energy[["time_period", "year"]].drop_duplicates(),
        on="time_period", how="left",
    )
    _vprint(verbose, "done")
    return df_out


def _build_comparison(
    xw: IEACrosswalk,
    df_out: pd.DataFrame,
    df_iea_raw: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    df_ssp = xw.aggregate_sisepuede(df_out, col_year="year")
    return xw.build_comparison(
        df_ssp, df_iea_raw,
        year_min=start_year, year_max=end_year,
    )


# ---------------------------------------------------------------------------
#   Output stages: plots, tables, CSVs
# ---------------------------------------------------------------------------

def _tag_suffix(tag: str) -> str:
    """Return ``'_<tag>'`` for a non-empty tag, else ``''``."""
    return f"_{tag}" if tag else ""


def _save_baseline_plots(
    df_comp_baseline: pd.DataFrame,
    iso_country: str,
    target_year: int,
    plots_dir: str,
    tag: str,
    verbose: bool,
) -> None:
    t = _tag_suffix(tag)
    _vprint(verbose, "Saving baseline discrepancy plots...")
    for variable, name in (("ratio", "ratio"), ("rel_error", "rel_error")):
        plot_baseline_discrepancy_bar(
            df_comp_baseline,
            year_target=target_year,
            variable=variable,
            country=iso_country,
            savepath=os.path.join(plots_dir, f"{iso_country}_baseline_discrepancy_{name}{t}.png"),
        )


def _save_before_after_plots(
    df_comp_baseline: pd.DataFrame,
    df_comp_calibrated: pd.DataFrame,
    iso_country: str,
    target_year: int,
    plots_dir: str,
    tag: str,
    verbose: bool,
) -> None:
    t = _tag_suffix(tag)
    _vprint(verbose, "Saving before/after discrepancy plots...")
    for variable, name in (("ratio", "ratio"), ("rel_error", "rel_error")):
        plot_before_after_discrepancy_bar(
            df_comp_baseline, df_comp_calibrated,
            year_target=target_year,
            variable=variable,
            country=iso_country,
            savepath=os.path.join(plots_dir, f"{iso_country}_before_after_discrepancy_{name}{t}.png"),
        )

    _vprint(verbose, "Saving primary-sector time series and bar plots...")
    plot_before_after_time_series(
        df_comp_baseline, df_comp_calibrated,
        year_target=target_year, mode="primary", country=iso_country,
        savepath=os.path.join(plots_dir, f"{iso_country}_primary_timeseries{t}.png"),
    )
    plot_before_after_bar(
        df_comp_baseline, df_comp_calibrated,
        year_target=target_year, mode="primary", country=iso_country,
        savepath=os.path.join(plots_dir, f"{iso_country}_primary_bar{t}.png"),
    )

    _vprint(verbose, "Saving fuel-mix plots per sector...")
    for sector in _sectors_with_fuel_mix(df_comp_baseline):
        try:
            plot_before_after_time_series(
                df_comp_baseline, df_comp_calibrated,
                year_target=target_year, mode="fuel_mix",
                sector=sector, country=iso_country, with_diagnostics=True,
                savepath=os.path.join(plots_dir, f"{iso_country}_fuelmix_{sector}_timeseries{t}.png"),
            )
            plot_before_after_bar(
                df_comp_baseline, df_comp_calibrated,
                year_target=target_year, mode="fuel_mix",
                sector=sector, country=iso_country,
                savepath=os.path.join(plots_dir, f"{iso_country}_fuelmix_{sector}_bar{t}.png"),
            )
        except Exception as e:
            if verbose:
                warnings.warn(f"Fuel-mix plot failed for sector={sector}: {e}")


def _save_summary_tables(
    df_comp_baseline: pd.DataFrame,
    df_comp_calibrated: pd.DataFrame,
    df_knobs: pd.DataFrame,
    iso_country: str,
    target_year: int,
    tables_dir: str,
    tag: str,
    verbose: bool,
) -> str:
    """Write the LaTeX summary tables. Returns the path to the knob-tables
    subdirectory (tag-aware so multiple runs do not collide)."""
    t = _tag_suffix(tag)
    knobs_dir = os.path.join(tables_dir, f"knobs{t}")

    _vprint(verbose, "Building LaTeX summary tables...")
    build_values_table(
        df_comp_baseline, df_comp_calibrated,
        country=iso_country, target_year=target_year,
        out_path=os.path.join(tables_dir, f"{iso_country}_values_table{t}.tex"),
    )
    build_improvement_table(
        df_comp_baseline, df_comp_calibrated,
        country=iso_country, target_year=target_year,
        out_path=os.path.join(tables_dir, f"{iso_country}_improvement_table{t}.tex"),
    )
    build_knob_tables(
        df_knobs,
        country=iso_country, target_year=target_year,
        out_dir=knobs_dir,
    )
    return knobs_dir


def _tagged(stem: str, iso_country: str, target_year: int, tag: str) -> str:
    """Build a filename of the form  '{stem}_{iso}_{year}[_{tag}].csv'."""
    return f"{stem}_{iso_country.lower()}_{target_year}{_tag_suffix(tag)}.csv"


def _save_calibration_dataframes(
    df_comp_baseline:   pd.DataFrame,
    df_comp_calibrated: pd.DataFrame,
    df_knobs:           pd.DataFrame,
    df_log:             pd.DataFrame,
    df_coverage:        pd.DataFrame,
    iso_country:        str,
    target_year:        int,
    data_dir:           str,
    tag:                str,
    verbose:            bool,
) -> dict:
    """Persist the dataframes most useful for downstream analysis / plotting
    (comparisons before/after, knob changes, calibration log, plan coverage).
    Returns a dict of {label: path}.
    """
    payload = {
        "comparison_baseline":   df_comp_baseline,
        "comparison_calibrated": df_comp_calibrated,
        "knobs":                 df_knobs,
        "calibration_log":       df_log,
        "coverage":              df_coverage,
    }

    paths: dict = {}
    for label, df in payload.items():
        path = os.path.join(data_dir, _tagged(label, iso_country, target_year, tag))
        df.to_csv(path, index=False)
        paths[label] = path
        _vprint(verbose, f"Saved {label}: {path}  ({len(df)} rows)")

    return paths


def _save_calibrated_inputs(
    df_input: pd.DataFrame,
    df_input_energy: pd.DataFrame,
    df_calibrated: pd.DataFrame,
    iso_country: str,
    target_year: int,
    output_dir: str,
    tag: str,
    verbose: bool,
) -> Tuple[str, str]:
    """Save the calibrated input frame (original columns only) plus a record
    of which columns were modified. Returns the two output paths."""
    original_cols      = df_input.columns.tolist()
    cols_in_calibrated = [c for c in original_cols if c in df_calibrated.columns]
    df_calibrated_out  = df_calibrated[cols_in_calibrated].copy()

    t = _tag_suffix(tag)
    path_inputs  = os.path.join(
        output_dir, f"input_data_{iso_country.lower()}_calibrated_{target_year}{t}.csv",
    )
    path_changed = os.path.join(
        output_dir, f"calibrated_columns_{iso_country.lower()}_{target_year}{t}.csv",
    )

    df_calibrated_out.to_csv(path_inputs, index=False)
    _vprint(verbose, f"Saved calibrated inputs: {path_inputs}  shape={df_calibrated_out.shape}")

    df_changed = _changed_columns(df_input_energy, df_calibrated, cols_in_calibrated)
    df_changed.to_csv(path_changed, index=False)
    _vprint(verbose, f"Saved changed-columns log: {path_changed}  ({len(df_changed)} columns adjusted)")

    return path_inputs, path_changed


# ---------------------------------------------------------------------------
#   Main entry point
# ---------------------------------------------------------------------------

def energy_calibration(
    iso_country:            str,
    target_year:            int,
    num_iterations:         int,
    cal_option:             int,
    start_year:             int,
    end_year:               int,
    iea_year_limit:         int,
    path_sisepuede_input:   str,
    path_iea_data_dir:      str,
    path_crosswalk_file:    str,
    path_output_dir:        str,
    tag:                    str = "",
    gamma:                  float = 100.0,
    enforce_varspec_bounds: bool = False,
    simplex_mode:           str = "full_simplex",
    verbose:                bool = True,
) -> dict:
    """Run the full energy calibration pipeline and persist all artefacts.

    Parameters
    ----------
    iso_country : str
        ISO-3 country code used to pull IEA data and label outputs (e.g. "PER").
    target_year : int
        Calendar year the calibration targets.
    num_iterations : int
        Number of Phase 1 + Phase 2 iterations passed to ``Calibrator.calibrate``.
    cal_option : int
        Calibration option (0..4). See ``Calibrator.calibrate``.
    start_year, end_year : int
        Inclusive year range for the IEA-vs-SISEPUEDE comparison table.
    iea_year_limit : int
        Last year of actual IEA data; rows beyond this in the input CSV are dropped.
    path_sisepuede_input, path_iea_data_dir, path_crosswalk_file : str
        Inputs.
    path_output_dir : str
        Root directory under which plots/, tables/, and calibrated CSVs are written.
    tag : str
        Optional suffix appended to output filenames.
    gamma : float
        QP regularisation weight (cal_option in {3, 4}).
    enforce_varspec_bounds : bool
        Enforce VariableSpec bounds inside the QP (cal_option in {3, 4}).
    verbose : bool
        When False, suppress progress prints and skip non-essential plots.

    Returns
    -------
    dict
        Bundle of outputs:
            df_calibrated, df_comp_baseline, df_comp_calibrated,
            df_log, df_knobs, plan, calibrator, paths
    """

    # ── Output layout ───────────────────────────────────────────────────────
    plots_dir  = _ensure_dir(os.path.join(path_output_dir, "plots"))
    tables_dir = _ensure_dir(os.path.join(path_output_dir, "tables"))
    data_dir   = _ensure_dir(os.path.join(path_output_dir, "data"))
    _ensure_dir(path_output_dir)

    # ── 1. Setup ────────────────────────────────────────────────────────────
    file_structure   = SISEPUEDEFileStructure()
    model_attributes = file_structure.model_attributes

    df_input        = _load_inputs(path_sisepuede_input, start_year, iea_year_limit, verbose)
    df_input_energy = _build_energy_input_frame(df_input, model_attributes, verbose)
    model_energycon = EnergyConsumption(model_attributes)

    # ── 2. Crosswalk + IEA data ─────────────────────────────────────────────
    IEACrosswalkBuilder(model_attributes, path_crosswalk_file).build(write_csv=True)
    xw = IEACrosswalk(model_attributes, path_crosswalk=path_crosswalk_file)
    _vprint(verbose, f"Crosswalk loaded: {len(xw.df_crosswalk)} (balance, product) pairs")

    loader     = IEADataLoader(path_iea_data_dir, model_attributes)
    df_iea_raw = loader.load_country(iso_country)
    _vprint(
        verbose,
        f"IEA rows loaded: {len(df_iea_raw)}  "
        f"years: {df_iea_raw['year'].min()}–{df_iea_raw['year'].max()}\n",
    )

    # ── 3. Baseline comparison ──────────────────────────────────────────────
    df_out_baseline  = _run_energy_model(model_energycon, df_input_energy, "baseline", verbose)
    df_comp_baseline = _build_comparison(xw, df_out_baseline, df_iea_raw, start_year, end_year)

    n_pairs = df_comp_baseline[["iea_balance_code", "iea_product_code"]].drop_duplicates().shape[0]
    _vprint(verbose, f"Baseline comparison: {len(df_comp_baseline)} rows, {n_pairs} unique pairs\n")

    _save_baseline_plots(df_comp_baseline, iso_country, target_year, plots_dir, tag, verbose)

    # ── 4. Calibration plan ─────────────────────────────────────────────────
    plan = build_energy_calibration_plan(model_attributes)
    _vprint(verbose,
        f"Calibration plan: {len(plan)} groups  "
        f"(scalar={len(plan.scalar_groups())}, simplex={len(plan.simplex_groups())})"
    )

    df_coverage = plan.coverage_report(df_comp_baseline)
    n_both  = int(df_coverage["has_both"].sum())
    n_iea   = int(df_coverage["has_iea_data"].sum())
    n_ssp   = int(df_coverage["has_ssp_data"].sum())
    n_total = len(df_coverage)
    _vprint(verbose,
        f"Coverage ({n_total} group-target pairs): "
        f"both={n_both}  iea_only={n_iea - n_both}  "
        f"ssp_only={n_ssp - n_both}  neither={n_total - n_iea - n_ssp + n_both}\n"
    )

    # ── 5. Run calibration ──────────────────────────────────────────────────
    calibrator = Calibrator(
        models                    = model_energycon,
        crosswalk                 = xw,
        df_iea_long               = df_iea_raw,
        year_target               = target_year,
        include_energy_production = False,
    )

    df_calibrated, log = calibrator.calibrate(
        df_in                  = df_input_energy,
        plan                   = plan,
        n_iter                 = num_iterations,
        option                 = cal_option,
        gamma                  = gamma,
        enforce_varspec_bounds = enforce_varspec_bounds,
        simplex_mode           = simplex_mode,
    )
    df_log = calibrator.log_summary(log)

    # ── 6. Post-calibration evaluation ──────────────────────────────────────
    df_out_calibrated  = _run_energy_model(model_energycon, df_calibrated, "calibrated", verbose)
    df_comp_calibrated = _build_comparison(xw, df_out_calibrated, df_iea_raw, start_year, end_year)

    df_knobs = calibrator.summarize_knobs(df_input_energy, df_calibrated, plan)

    _save_before_after_plots(
        df_comp_baseline, df_comp_calibrated,
        iso_country, target_year, plots_dir, tag, verbose,
    )
    knobs_dir = _save_summary_tables(
        df_comp_baseline, df_comp_calibrated, df_knobs,
        iso_country, target_year, tables_dir, tag, verbose,
    )

    # ── 7. Save calibrated inputs ───────────────────────────────────────────
    path_inputs, path_changed = _save_calibrated_inputs(
        df_input, df_input_energy, df_calibrated,
        iso_country, target_year, path_output_dir, tag, verbose,
    )

    # ── 8. Save analysis-ready dataframes (comparisons, knobs, log, coverage)
    data_paths = _save_calibration_dataframes(
        df_comp_baseline   = df_comp_baseline,
        df_comp_calibrated = df_comp_calibrated,
        df_knobs           = df_knobs,
        df_log             = df_log,
        df_coverage        = df_coverage,
        iso_country        = iso_country,
        target_year        = target_year,
        data_dir           = data_dir,
        tag                = tag,
        verbose            = verbose,
    )

    # ── Final summary of where outputs landed ───────────────────────────────
    _vprint(verbose, "\n" + "─" * 70)
    _vprint(verbose, "Calibration complete. Outputs written to:")
    _vprint(verbose, f"  plots             : {plots_dir}")
    _vprint(verbose, f"  tables            : {tables_dir}")
    _vprint(verbose, f"  knob tables       : {knobs_dir}")
    _vprint(verbose, f"  data (CSVs)       : {data_dir}")
    _vprint(verbose, f"  calibrated inputs : {path_inputs}")
    _vprint(verbose, f"  changed columns   : {path_changed}")
    _vprint(verbose, "─" * 70)

    return {
        "df_calibrated":      df_calibrated,
        "df_comp_baseline":   df_comp_baseline,
        "df_comp_calibrated": df_comp_calibrated,
        "df_log":             df_log,
        "df_knobs":           df_knobs,
        "df_coverage":        df_coverage,
        "plan":               plan,
        "calibrator":         calibrator,
        "paths": {
            "calibrated_inputs": path_inputs,
            "changed_columns":   path_changed,
            "plots_dir":         plots_dir,
            "tables_dir":        tables_dir,
            "knobs_dir":         knobs_dir,
            "data_dir":          data_dir,
            **data_paths,
        },
    }
