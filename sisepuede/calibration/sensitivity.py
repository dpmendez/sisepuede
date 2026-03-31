"""
sisepuede/calibration/sensitivity.py

Sensitivity analysis framework for SISEPUEDE energy calibration.

The framework supports two sampling modes:

  OAT (one-at-a-time)
      One variable is perturbed per run; all others stay at their baseline
      values. Good for quick screening of which variables matter at all.
      N_runs = 1 (baseline) + N_vars x N_levels.

  LHS (Latin Hypercube Sampling)
      All variables are perturbed simultaneously in each run, sampled from a
      space-filling LHS design. Better parameter-space coverage and produces
      clean surrogate-model training data. N_runs = n_samples.

Suggested workflow
------------------
1. Run OAT to identify the 10-20 variables that actually move IEA targets.
2. Run LHS on that sensitive subset to build surrogate training data (for potential future calibration ML approach).
3. Use sensitivity_scores() to rank variables by Spearman rank correlation.
4. Use linearity_check() to decide between linear calibration vs optimisation.

Typical usage
-------------
    from sisepuede.manager.sisepuede_file_structure import SISEPUEDEFileStructure
    from sisepuede.calibration.iea_crosswalk import IEACrosswalk
    from sisepuede.calibration.sensitivity import (
        VariableSpec, SensitivityRunner, sensitivity_scores, linearity_check,
    )

    file_struct      = SISEPUEDEFileStructure()
    model_attributes = file_struct.model_attributes

    # Construct runner
    runner = SensitivityRunner(
        models        = models,          # SISEPUEDEModels instance
        df_baseline   = df_input,        # one country, all time periods
        iea_crosswalk = IEACrosswalk(model_attributes),
        df_iea_raw    = df_iea_raw,
        iso           = "ARG",
    )

    # Define which variables to vary and by how much
    specs = [
        VariableSpec("frac_inen_energy_cement_coal",   lb=0.5, ub=1.5),
        VariableSpec("consumpinit_inen_energy_cement", lb=0.8, ub=1.2),
    ]

    # Quick OAT screen
    result_oat = runner.run_oat(specs)

    # Full LHS for surrogate training
    result_lhs = runner.run_lhs(specs, n_samples=50)

    # Analysis
    scores = sensitivity_scores(result_lhs, years=[2021])
    fig    = linearity_check(
        result_lhs,
        var_column       = "consumpinit_inen_energy_cement",
        iea_balance_code = "TFC",
        iea_product_code = "COAL",
        years            = [2021],
    )

Notes on simplex groups
-----------------------
Fuel-fraction variables (e.g. frac_inen_energy_cement_*) belong to simplex
groups: within each industry category the fractions must sum to 1. Free
scalar perturbation breaks this constraint.

This version applies scalar scaling regardless. The VariableSpec fields
`is_simplex_group` and `simplex_group_id` are included as forward-compatible
markers - they are not yet acted on. Simplex-aware perturbation should be added in the future.
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from sisepuede.manager.sisepuede_models import SISEPUEDEModels

try:
    from scipy.stats.qmc import LatinHypercube
    from scipy.stats.qmc import scale as lhs_scale
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


####################
#    DATA CLASSES  #
####################

@dataclass
class VariableSpec:
    """Specification for one input variable to include in sensitivity analysis.

    Parameters
    ----------
    column : str
        Exact column name in the SISEPUEDE input DataFrame.
    lb : float
        Lower bound for the scale factor (multiplier on baseline value).
        E.g. 0.8 -> the variable can be reduced by up to 20 %.
    ub : float
        Upper bound for the scale factor.
        E.g. 1.2 -> the variable can be increased by up to 20 %.
    is_simplex_group : bool
        Informational flag. If True this variable is part of a simplex group
        (fuel fractions that must sum to 1 within a category).
        Simplex-aware perturbation is NOT yet implemented — free scalar
        scaling is applied regardless.
    simplex_group_id : int | None
        Group ID from ModelAttributes.dict_field_to_simplex_group.
        Stored alongside is_simplex_group for future use.
    """
    column: str
    lb: float = 0.8
    ub: float = 1.2
    is_simplex_group: bool = False
    simplex_group_id: Optional[int] = None


@dataclass
class SensitivityResult:
    """Output from one sensitivity analysis run.

    Attributes
    ----------
    variable_specs : List[VariableSpec]
        The specs used to define this run.
    sampling_mode : str
        "oat" or "lhs".
    input_samples : pd.DataFrame
        Shape (N_runs, N_vars).  Each row is the set of scale factors applied
        in one model run.  Index is run_index (0-based integer).
        This is the X matrix for surrogate-model training.
    iea_comparison : pd.DataFrame
        All years, all (balance, product) pairs, all runs — stacked.
        Columns: run_index + all columns from IEACrosswalk.build_comparison().
        Filter to a specific year and pivot on run_index to get the Y matrix
        for surrogate training.
    model_outputs : pd.DataFrame
        Full SSP model output for every run, stacked.
        Columns: run_index, time_period, + all SSP output columns.
        Useful for tracking side-effects on outputs beyond the IEA targets.
    baseline_output : pd.DataFrame
        Unperturbed model output.
    baseline_iea_comparison : pd.DataFrame
        IEA comparison table for the unperturbed baseline (all years).
    """
    variable_specs: List[VariableSpec]
    sampling_mode: str
    input_samples: pd.DataFrame
    iea_comparison: pd.DataFrame
    model_outputs: pd.DataFrame
    baseline_output: pd.DataFrame
    baseline_iea_comparison: pd.DataFrame


##########################
#    PURE FUNCTIONS      #
##########################

def perturb_inputs(
    df: pd.DataFrame,
    variable_scales: Dict[str, float],
) -> pd.DataFrame:
    """Return a copy of df with specified columns multiplied by scale factors.

    All other columns are left unchanged. The original DataFrame is never
    modified.

    Parameters
    ----------
    df : pd.DataFrame
        SISEPUEDE input DataFrame.
    variable_scales : Dict[str, float]
        {column_name: scale_factor}.  A scale_factor of 1.0 is a no-op.

    Raises
    ------
    ValueError
        If any key in variable_scales is not a column in df.

    Examples
    --------
    # Reduce coal fraction by 20 % for a single manual test:
    df_test = perturb_inputs(df_baseline, {"frac_inen_energy_cement_coal": 0.8})
    """
    missing = [c for c in variable_scales if c not in df.columns]
    if missing:
        raise ValueError(
            f"perturb_inputs: columns not found in input DataFrame: {missing}"
        )
    df_out = df.copy()
    for col, scale in variable_scales.items():
        df_out[col] = df_out[col] * scale
    return df_out


def sample_oat(
    specs: List[VariableSpec],
    levels: List[float] = None,
) -> pd.DataFrame:
    """Generate one-at-a-time (OAT) samples as a DataFrame of scale factors.

    Each row perturbs exactly one variable; all others have scale factor 1.0.
    Row 0 is the unperturbed baseline (all 1.0).

    Parameters
    ----------
    specs : List[VariableSpec]
        Variables to include.
    levels : List[float] | None
        Scale-factor levels to test per variable.
        Defaults to [0.8, 0.9, 1.1, 1.2].

    Returns
    -------
    pd.DataFrame
        Shape (1 + N_vars x N_levels, N_vars).
        Index is run_index (0-based).  Columns are variable column names.
    """
    if levels is None:
        levels = [0.8, 0.9, 1.1, 1.2]

    cols = [s.column for s in specs]
    rows = [{c: 1.0 for c in cols}]       # baseline row

    for spec in specs:
        print(spec)
        for level in levels:
            print(level)
            row = {c: 1.0 for c in cols}
            row[spec.column] = level
            rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def sample_lhs(
    specs: List[VariableSpec],
    n_samples: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate Latin Hypercube Samples (LHS) as a DataFrame of scale factors.

    Each row is one sample: a full set of scale factors for all variables,
    drawn from a space-filling design bounded by [spec.lb, spec.ub].

    Parameters
    ----------
    specs : List[VariableSpec]
        Variables to include. Bounds taken from spec.lb / spec.ub.
    n_samples : int
        Number of LHS samples (= number of model runs).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shape (n_samples, N_vars). Index is run_index (0-based).
        Columns are variable column names.

    Raises
    ------
    ImportError
        If scipy is not installed.
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError(
            "scipy >= 1.7 is required for LHS sampling."
            "Install with: pip install scipy"
        )

    cols = [s.column for s in specs]
    lb   = np.array([s.lb for s in specs])
    ub   = np.array([s.ub for s in specs])

    sampler      = LatinHypercube(d=len(specs), seed=seed)
    unit_samples = sampler.random(n=n_samples)     # (n_samples, n_vars) in [0, 1]
    scaled       = lhs_scale(unit_samples, lb, ub) # (n_samples, n_vars) in [lb, ub]

    return pd.DataFrame(scaled, columns=cols).reset_index(drop=True)


###########################
#    SENSITIVITY RUNNER   #
###########################

class SensitivityRunner:
    """Run SISEPUEDE sensitivity analysis by perturbing input variables.

    Wraps a SISEPUEDEModels instance and an IEACrosswalk to produce
    SensitivityResult objects useful for:

      - Identifying which inputs drive IEA calibration targets
      - Linearity checks (-> linear calibration vs optimisation)
      - Surrogate-model training data

    The baseline model run is lazy and cached — it is only executed on the
    first call to run_oat() or run_lhs().

    Parameters
    ----------
    models : SISEPUEDEModels or EnergyConsumption
        Initialised model object exposing a .project(df) method. For integrated
        runs including energy production, use SISEPUEDEModels; for energy
        consumption only, use EnergyConsumption.
    df_baseline : pd.DataFrame
        Baseline SISEPUEDE input DataFrame for one country, all time periods.
        Must include a 'year' column (mapped from time_period) so that
        IEACrosswalk.aggregate_sisepuede() can attach calendar years to
        model output.
    iea_crosswalk : IEACrosswalk
        Initialised crosswalk object.
    df_iea_raw : pd.DataFrame
        Raw IEA World Energy Balances dataframe.
    iso : str
        ISO-3 country code (e.g. "ARG").  Used to filter df_iea_raw.
    include_energy_production : bool
        Passed to SISEPUEDEModels.project() as include_electricity_in_energy.
        Set True only when calibrating supply-side variables that require
        the Julia EnergyProduction back-end. Ignored for EnergyConsumption.
    year_min : int | None
        If given, passed to IEACrosswalk.build_comparison() so that both
        the IEA frame and the SISEPUEDE frame are trimmed to year >= year_min
        before joining.  Prevents IEA-only rows from years outside the
        simulation window from polluting the comparison table.
    year_max : int | None
        If given, passed to IEACrosswalk.build_comparison() so that both
        frames are trimmed to year <= year_max before joining.
    """

    def __init__(
        self,
        models,
        df_baseline: pd.DataFrame,
        iea_crosswalk,
        df_iea_raw: pd.DataFrame,
        iso: str,
        include_energy_production: bool = False,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> None:

        self.models                   = models
        self.df_baseline              = df_baseline.copy()
        self.iea_crosswalk            = iea_crosswalk
        self.df_iea_raw               = df_iea_raw
        self.iso                      = iso
        self.include_energy_production = include_energy_production
        self.year_min                 = year_min
        self.year_max                 = year_max

        # lazily computed and cached
        self._baseline_output         = None
        self._baseline_iea_long       = None
        self._baseline_iea_comparison = None

    # ------------------------------------------------------------------
    #   CACHED BASELINE PROPERTIES
    # ------------------------------------------------------------------

    @property
    def baseline_output(self) -> pd.DataFrame:
        """Model output for the unperturbed baseline. Computed once."""
        if self._baseline_output is None:
            print("Running baseline model...", end=" ", flush=True)
            t0 = time.time()
            if isinstance(self.models, SISEPUEDEModels):
                df_out = self.models.project(
                    self.df_baseline,
                    include_electricity_in_energy=self.include_energy_production,
                )
            else:
                df_out = self.models.project(
                    self.df_baseline,
                )
            self._baseline_output = self._attach_year(df_out)
            print(f"done ({time.time() - t0:.1f}s)")
        return self._baseline_output

    @property
    def baseline_iea_long(self) -> pd.DataFrame:
        """IEA long frame for this country (all years). Computed once."""
        if self._baseline_iea_long is None:
            self._baseline_iea_long = self.df_iea_raw
        return self._baseline_iea_long

    @property
    def baseline_iea_comparison(self) -> pd.DataFrame:
        """Full IEA comparison table for the unperturbed baseline (all years)."""
        if self._baseline_iea_comparison is None:
            df_ssp  = self.iea_crosswalk.aggregate_sisepuede(self.baseline_output)
            self._baseline_iea_comparison = self.iea_crosswalk.build_comparison(
                df_ssp, self.baseline_iea_long,
                year_min=self.year_min,
                year_max=self.year_max,
            )
        return self._baseline_iea_comparison

    # ------------------------------------------------------------------
    #   INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _attach_year(self, df_out: pd.DataFrame) -> pd.DataFrame:
        """Merge the 'year' column from df_baseline into df_out by time_period.

        IEACrosswalk.aggregate_sisepuede() expects a 'year' column.  The model
        outputs 'time_period'; the year mapping is taken from df_baseline.

        Raises
        ------
        ValueError
            If df_baseline does not contain a 'year' column.
        """
        if "year" in df_out.columns:
            return df_out

        if "year" not in self.df_baseline.columns:
            raise ValueError(
                "Cannot attach 'year' to model output: df_baseline does not "
                "have a 'year' column. Add it (mapping time_period -> calendar "
                "year) before constructing SensitivityRunner."
            )

        year_map = (
            self.df_baseline[["time_period", "year"]]
            .drop_duplicates()
        )
        return df_out.merge(year_map, on="time_period", how="left")

    def _run_single(
        self,
        variable_scales: Dict[str, float],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the model once with the given scale factors.

        Returns
        -------
        (df_out, df_comp)
            df_out  : full model output with year column attached
            df_comp : output of IEACrosswalk.build_comparison() (all years)
        """
        df_in   = perturb_inputs(self.df_baseline, variable_scales)
        if isinstance(self.models, SISEPUEDEModels):
            df_out  = self.models.project(
                df_in,
                include_electricity_in_energy=self.include_energy_production,
            )
        else:
            df_out  = self.models.project(
                df_in,
            )
        df_out  = self._attach_year(df_out)
        df_ssp  = self.iea_crosswalk.aggregate_sisepuede(df_out)
        df_comp = self.iea_crosswalk.build_comparison(
            df_ssp, self.baseline_iea_long,
            year_min=self.year_min,
            year_max=self.year_max,
        )
        return df_out, df_comp

    def _collect_results(
        self,
        samples_df: pd.DataFrame,
        sampling_mode: str,
        specs: List[VariableSpec],
    ) -> SensitivityResult:
        """Execute all rows of samples_df through the model and collect results.

        Prints a one-line progress update per run including elapsed time.
        """
        # ensure baseline is computed and cached before perturbation loop
        _ = self.baseline_output
        _ = self.baseline_iea_comparison

        all_outputs     = []
        all_comparisons = []
        n_runs          = len(samples_df)

        for run_idx, row in samples_df.iterrows():
            variable_scales = row.to_dict()

            # build a compact label for the progress line
            varying = [
                (c, v) for c, v in variable_scales.items()
                if abs(v - 1.0) > 1e-9
            ]
            if sampling_mode == "oat":
                label = (
                    f"{varying[0][0]} = {varying[0][1]:.3f}"
                    if varying
                    else "baseline"
                )
            else:
                label = f"{len(varying)} vars perturbed"

            print(
                f"  Run {run_idx + 1:>{len(str(n_runs))}}/{n_runs}  {label}",
                end=" ... ",
                flush=True,
            )
            t0 = time.time()

            df_out, df_comp = self._run_single(variable_scales)

            df_out          = df_out.copy()
            df_out.insert(0, "run_index", run_idx)

            df_comp         = df_comp.copy()
            df_comp.insert(0, "run_index", run_idx)

            all_outputs.append(df_out)
            all_comparisons.append(df_comp)

            print(f"done ({time.time() - t0:.1f}s)")

        return SensitivityResult(
            variable_specs          = specs,
            sampling_mode           = sampling_mode,
            input_samples           = samples_df.copy(),
            iea_comparison          = pd.concat(all_comparisons, ignore_index=True),
            model_outputs           = pd.concat(all_outputs,     ignore_index=True),
            baseline_output         = self.baseline_output.copy(),
            baseline_iea_comparison = self.baseline_iea_comparison.copy(),
        )

    # ------------------------------------------------------------------
    #   PUBLIC METHODS
    # ------------------------------------------------------------------

    def run_oat(
        self,
        specs: List[VariableSpec],
        levels: List[float] = None,
    ) -> SensitivityResult:
        """Run one-at-a-time (OAT) sensitivity analysis.

        One variable is perturbed per run; all others stay at baseline values.
        Includes an unperturbed baseline run as run 0.

        Parameters
        ----------
        specs : List[VariableSpec]
            Variables to perturb.
        levels : List[float] | None
            Scale factors to test.  Defaults to [0.8, 0.9, 1.1, 1.2].

        Returns
        -------
        SensitivityResult
        """
        _levels = levels or [0.8, 0.9, 1.1, 1.2]
        samples_df = sample_oat(specs, levels=_levels)
        n_perturb  = len(specs) * len(_levels)
        print(
            f"OAT: {len(specs)} variables x {len(_levels)} levels "
            f"= {n_perturb} perturbations + 1 baseline "
            f"({len(samples_df)} total runs)"
        )
        return self._collect_results(samples_df, "oat", specs)

    def run_lhs(
        self,
        specs: List[VariableSpec],
        n_samples: int = 50,
        seed: int = 42,
    ) -> SensitivityResult:
        """Run Latin Hypercube Sampling (LHS) sensitivity analysis.

        All variables are perturbed simultaneously in every run.  Produces
        space-filling samples suitable as surrogate-model training data.

        Parameters
        ----------
        specs : List[VariableSpec]
            Variables to perturb.  Bounds taken from spec.lb / spec.ub.
        n_samples : int
            Number of LHS samples (= number of model runs).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        SensitivityResult
        """
        samples_df = sample_lhs(specs, n_samples=n_samples, seed=seed)
        print(
            f"LHS: {n_samples} samples across {len(specs)} variables "
            f"(seed={seed})"
        )
        return self._collect_results(samples_df, "lhs", specs)


############################
#    ANALYSIS FUNCTIONS    #
############################

def sensitivity_scores(
    result: SensitivityResult,
    years: List[int],
    output_col: str = "rel_error_iea",
) -> pd.DataFrame:
    """Rank input variables by their influence on IEA calibration targets.

    Computes Spearman rank correlation between each input variable's scale
    factor and each IEA (balance, product) output, averaged over the
    requested years.

    Spearman r ∈ [-1, 1]:
      |r| close to 1 -> strong monotone influence
      |r| close to 0 -> weak or no influence
      r > 0           -> increasing the variable increases the IEA ratio
      r < 0           -> increasing the variable decreases the IEA ratio

    Parameters
    ----------
    result : SensitivityResult
    years : List[int]
        Calendar years to include.  Output values are averaged across years
        before correlation is computed.
    output_col : str
        Column from iea_comparison to use as the output signal.
        Default "ratio_sisepuede_over_iea".
        Alternatives: "value_sisepuede_tj", "diff_sisepuede_iea".

    Returns
    -------
    pd.DataFrame
        Rows   = input variable column names.
        Columns = IEA pair labels  "{balance_code}_{product_code}".
        Values = Spearman r.
        NaN where fewer than 3 matched data points exist.
    """
    from scipy.stats import spearmanr

    df_comp = result.iea_comparison
    df_filtered = df_comp[df_comp["year"].isin(years)].copy()

    # average output_col across requested years per (run, balance, product)
    df_avg = (
        df_filtered
        .groupby(["run_index", "iea_balance_code", "iea_product_code"])[output_col]
        .mean()
        .reset_index()
    )

    # pivot to: rows = run_index, columns = IEA pair label
    df_pivot = df_avg.pivot(
        index   ="run_index",
        columns = ["iea_balance_code", "iea_product_code"],
        values  = output_col,
    )
    df_pivot.columns = [f"{b}_{p}" for b, p in df_pivot.columns]

    # align input samples on the same run indices
    X = result.input_samples.loc[df_pivot.index]

    # compute Spearman r between each (input, output) pair
    scores = {}       # {input_col: {iea_pair: r}}
    for in_col in X.columns:
        row = {}
        for out_col in df_pivot.columns:
            y     = df_pivot[out_col]
            valid = y.notna()
            if valid.sum() < 3:
                row[out_col] = np.nan
            else:
                r, _ = spearmanr(X.loc[valid, in_col], y[valid])
                row[out_col] = r
        scores[in_col] = row

    return pd.DataFrame(scores).T   # rows = input vars, columns = IEA pairs


def linearity_check(
    result: SensitivityResult,
    var_column: str,
    iea_balance_code: str,
    iea_product_code: str,
    years: List[int],
    output_col: str = "ratio_sisepuede_over_iea",
):
    """Scatter plot to check whether the input->output relationship is linear.

    Produces a two-panel figure:
      Left  : scatter of scale factor vs IEA output, with a linear fit and R2.
      Right : residuals from that fit.

    How to interpret:
      R2 =~ 1, random residuals  -> relationship is linear -> linear calibration
      Curved pattern or R2 << 1 -> nonlinear -> use optimisation or a surrogate

    Parameters
    ----------
    result : SensitivityResult
    var_column : str
        Input variable to plot on the x-axis.
    iea_balance_code : str
        IEA balance code to plot (e.g. "TFC", "INDPROD").
    iea_product_code : str
        IEA product code to plot (e.g. "COAL", "ELECTR").
    years : List[int]
        Calendar years to include. Output values are averaged across years.
    output_col : str
        Column from iea_comparison to use on the y-axis.
        Default "rel_error_iea".

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If no data is found for the requested (balance, product, years).
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, linregress

    df_comp     = result.iea_comparison
    df_filtered = df_comp[
        (df_comp["year"].isin(years))
        & (df_comp["iea_balance_code"] == iea_balance_code)
        & (df_comp["iea_product_code"] == iea_product_code)
    ].copy()

    if df_filtered.empty:
        raise ValueError(
            f"No data found for balance='{iea_balance_code}', "
            f"product='{iea_product_code}', years={years}."
        )

    df_avg = (
        df_filtered
        .groupby("run_index")[output_col]
        .mean()
        .reset_index()
        .rename(columns={output_col: "y"})
    )

    x_vals = result.input_samples.loc[df_avg["run_index"], var_column].values
    y_vals = df_avg["y"].values

    valid = ~(np.isnan(x_vals) | np.isnan(y_vals))
    x_v, y_v = x_vals[valid], y_vals[valid]

    if len(x_v) < 3:
        raise ValueError(
            f"Not enough valid data points ({len(x_v)}) to fit a line."
        )

    slope, intercept, r_value, _, _ = linregress(x_v, y_v)
    r2      = r_value ** 2
    spear_r, _ = spearmanr(x_v, y_v)
    x_fit   = np.linspace(x_v.min(), x_v.max(), 200)
    y_fit   = slope * x_fit + intercept
    y_resid = y_v - (slope * x_v + intercept)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ── left: scatter + linear fit ──────────────────────────────────────────
    ax = axes[0]
    ax.scatter(x_v, y_v, alpha=0.7, edgecolors="k", linewidths=0.4, zorder=3)
    ax.plot(x_fit, y_fit, color="C1", linewidth=1.8,
            label=f"Linear fit  R2 = {r2:.3f}")
    ax.axhline(1.0, color="grey", linewidth=0.9, linestyle="--",
               label="ratio = 1  (perfect match)")
    ax.set_xlabel(f"Scale factor:  {var_column}", fontsize=9)
    ax.set_ylabel(output_col, fontsize=9)
    ax.set_title(
        f"{iea_balance_code} / {iea_product_code}\n"
        f"Spearman r = {spear_r:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)

    # ── right: residuals ────────────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(x_v, y_resid, alpha=0.7, edgecolors="k", linewidths=0.4, zorder=3)
    ax.axhline(0.0, color="C1", linewidth=1.8)
    ax.set_xlabel(f"Scale factor:  {var_column}", fontsize=9)
    ax.set_ylabel("Residual", fontsize=9)
    ax.set_title(
        "Residuals\n"
        "(random pattern -> linear approximation is adequate)",
        fontsize=10,
    )

    fig.tight_layout()
    return fig