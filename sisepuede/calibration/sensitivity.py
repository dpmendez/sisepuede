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
groups: within each industry category the fractions must sum to 1.

Which columns are co-constrained is resolved through a SimplexRegistry
(see sisepuede/calibration/_simplex_registry.py), which is a thin facade
over ModelAttributes.dict_field_to_simplex_group / dict_simplex_group_to_fields
— i.e. the same attribute-table CSVs that define the model.  Nothing in this
module infers simplex membership from column names; the registry is the
single source of truth.

Simplex-aware perturbation is implemented via perturb_inputs_simplex():
  - Scale the focal column by scale_factor (clamped to [0, 1]).
  - Redistribute the gained/lost mass proportionally across all other columns
    in the same simplex group (looked up from the registry).

For OAT: only one simplex variable is non-unity per run, so the redistribution
is unambiguous.

For LHS: multiple simplex variables in the same constraint may have non-unity
scale factors simultaneously.  All scale factors are applied at once via
simultaneous renormalization (Aitchison perturbation):

    desired_i  = original_i x scale_i   for every column in the constraint
    final_i    = desired_i  / sum(desired)

Columns not in the spec list act as background (scale = 1.0).  This is
order-independent and gives each variable's Spearman score a clean
single-variable interpretation.
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from sisepuede.manager.sisepuede_models import SISEPUEDEModels
from sisepuede.calibration._simplex_registry import SimplexRegistry

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
        If True, this variable is part of a simplex group (fuel fractions
        that must sum to 1 within a category).  When True,
        perturb_inputs_simplex() is used instead of simple scalar scaling so
        that the constraint is preserved.
    simplex_group_id : int | None
        Group ID from ModelAttributes.dict_field_to_simplex_group.  This is
        the authoritative identifier used to look up the column's simplex
        siblings (via a SimplexRegistry) — not informational.  When
        is_simplex_group is True, simplex_group_id must be populated and
        consistent with the registry used at perturbation time.  Populate
        automatically with CalibrationPlan.from_specs_dict(model_attributes=...).
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


def perturb_inputs_simplex(
    df: pd.DataFrame,
    focal_column: str,
    scale_factor: float,
    simplex_columns: Optional[List[str]] = None,
    simplex_registry: Optional[SimplexRegistry] = None,
) -> pd.DataFrame:
    """Perturb one fuel-fraction column while preserving the simplex sum = 1.

    Scales `focal_column` by `scale_factor`, clamps to [0, 1], then
    proportionally adjusts all other co-constrained columns so that their
    per-row sum remains 1.0.

    Parameters
    ----------
    df : pd.DataFrame
        SISEPUEDE input DataFrame.
    focal_column : str
        The fuel-fraction column to scale (e.g. "frac_inen_energy_cement_coal").
    scale_factor : float
        Multiplier applied to focal_column before clamping.
        Values > 1.0 shift mass toward the focal fuel.
        Values < 1.0 shift mass away from it.
    simplex_columns : List[str] | None
        All columns in this simplex constraint (must include focal_column).
        When None, looked up from `simplex_registry`.  Exactly one of
        `simplex_columns` or `simplex_registry` must be provided.
    simplex_registry : SimplexRegistry | None
        Authoritative registry of simplex-group membership.  If supplied,
        the co-constrained columns for `focal_column` are read from the
        registry rather than inferred from column names.

    Returns
    -------
    pd.DataFrame
        Copy of df with focal_column and co-constrained columns adjusted.

    Raises
    ------
    ValueError
        If focal_column is not in df, or if neither simplex_columns nor
        simplex_registry is provided.
    """
    if focal_column not in df.columns:
        raise ValueError(
            f"perturb_inputs_simplex: column '{focal_column}' not found in DataFrame."
        )

    if simplex_columns is None:
        if simplex_registry is None:
            raise ValueError(
                "perturb_inputs_simplex: must supply either simplex_columns "
                "(explicit list) or simplex_registry (SimplexRegistry built "
                "from ModelAttributes).  Naming-based inference is no longer "
                "supported."
            )
        simplex_columns = simplex_registry.co_constrained_with(focal_column)
        # Restrict to columns actually present in the DataFrame
        simplex_columns = [c for c in simplex_columns if c in df.columns]
        if focal_column not in simplex_columns:
            simplex_columns = [focal_column] + simplex_columns

    other_cols = [c for c in simplex_columns if c != focal_column]

    df_out = df.copy()
    original_focal = df_out[focal_column].values.astype(float)

    # Desired new focal value, clamped to [0, 1]
    new_focal = np.clip(original_focal * scale_factor, 0.0, 1.0)
    delta = new_focal - original_focal   # >0 = take mass from others; <0 = give mass back

    if not other_cols:
        df_out[focal_column] = new_focal
        return df_out

    other_vals = df_out[other_cols].values.astype(float)   # (n_rows, n_other)
    other_sum  = other_vals.sum(axis=1)                     # (n_rows,)

    # Cap delta so no column goes below 0
    # When delta > 0 (focal grows): cap at how much mass is available in others
    # When delta < 0 (focal shrinks): cap at how much focal itself can give
    max_gain = other_sum
    max_loss = original_focal
    delta = np.where(delta > 0,
                     np.minimum(delta,  max_gain),
                     np.maximum(delta, -max_loss))

    # Proportional redistribution weights across other columns
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(
            other_sum[:, None] > 1e-10,
            other_vals / other_sum[:, None],
            np.full((len(other_sum), len(other_cols)), 1.0 / len(other_cols)),
        )

    df_out[focal_column] = original_focal + delta
    for i, col in enumerate(other_cols):
        df_out[col] = np.clip(other_vals[:, i] - delta * weights[:, i], 0.0, None)

    return df_out


def apply_perturbations(
    df: pd.DataFrame,
    specs: List[VariableSpec],
    variable_scales: Dict[str, float],
    simplex_registry: Optional[SimplexRegistry] = None,
) -> pd.DataFrame:
    """Apply perturbations to df, routing scalar and simplex specs correctly.

    Scalar specs (is_simplex_group=False) are handled by perturb_inputs() —
    each column is independently multiplied by its scale factor.

    Simplex specs (is_simplex_group=True) are handled via simultaneous
    renormalization (Aitchison perturbation): for every simplex constraint
    that has at least one non-unity scale factor, all columns in that
    constraint (as defined by the SimplexRegistry) are scaled together and
    then renormalized so the row sum remains 1.0.  Columns not in the spec
    list act as background (scale=1.0).  This is order-independent and
    keeps each variable's Spearman score interpretable as a single-variable
    effect.

    Simplex-group membership is resolved authoritatively from the supplied
    `simplex_registry` (a wrapper over ModelAttributes.dict_field_to_simplex_group).
    There is no name-based fallback: if a spec has is_simplex_group=True but
    no registry is provided, the function raises.

    Parameters
    ----------
    df : pd.DataFrame
        SISEPUEDE input DataFrame (baseline).
    specs : List[VariableSpec]
        Full spec list for this run (used to determine constraint type per column).
    variable_scales : Dict[str, float]
        {column_name: scale_factor} for this particular run.
    simplex_registry : SimplexRegistry | None
        Registry of simplex-group membership.  Required whenever any spec has
        is_simplex_group=True.

    Returns
    -------
    pd.DataFrame
        Perturbed copy of df.

    Raises
    ------
    ValueError
        If any simplex spec is present but no simplex_registry is provided,
        or if a simplex spec references a column that the registry does not
        recognise.
    """
    spec_map: Dict[str, VariableSpec] = {s.column: s for s in specs}

    # ── 1. Scalar perturbations (all at once) ────────────────────────────────
    scalar_scales = {
        col: scale
        for col, scale in variable_scales.items()
        if col in spec_map and not spec_map[col].is_simplex_group
    }
    df_out = perturb_inputs(df, scalar_scales) if scalar_scales else df.copy()

    # ── 2. Simplex perturbations (simultaneous renormalization per constraint) ─
    # Collect only simplex specs whose scale is actually non-unity.
    simplex_specs = [
        s for s in specs
        if s.is_simplex_group and abs(variable_scales.get(s.column, 1.0) - 1.0) > 1e-9
    ]

    if not simplex_specs:
        return df_out

    if simplex_registry is None:
        raise ValueError(
            "apply_perturbations: simplex specs are present but no "
            "simplex_registry was supplied.  Build one with "
            "SimplexRegistry.from_model_attributes(model_attributes) and pass "
            "it as the simplex_registry argument."
        )

    # Group simplex specs by their authoritative simplex group ID.
    group_to_specs: Dict[int, List[VariableSpec]] = {}
    unknown: List[str] = []
    for spec in simplex_specs:
        gid = simplex_registry.group_id(spec.column)
        if gid is None:
            unknown.append(spec.column)
            continue
        # Cross-check spec metadata if populated
        if spec.simplex_group_id is not None and spec.simplex_group_id != gid:
            raise ValueError(
                f"apply_perturbations: spec for '{spec.column}' claims "
                f"simplex_group_id={spec.simplex_group_id} but the registry "
                f"says {gid}.  VariableSpec metadata and ModelAttributes are "
                "out of sync — rebuild the plan with the current ModelAttributes."
            )
        group_to_specs.setdefault(gid, []).append(spec)

    if unknown:
        raise ValueError(
            "apply_perturbations: the following columns are marked "
            f"is_simplex_group=True but are not registered in any simplex "
            f"group (check ModelAttributes): {unknown}"
        )

    for gid, grp_specs in group_to_specs.items():
        # All columns in this simplex constraint, per the registry.
        # Restrict to columns that actually exist in df_out so we never read
        # missing values.
        all_cols = [c for c in simplex_registry.columns_in_group(gid)
                    if c in df_out.columns]
        if not all_cols:
            continue

        # Scale vector: spec columns use their sampled factor; others use 1.0
        scale_for_col = {c: 1.0 for c in all_cols}
        for spec in grp_specs:
            if spec.column in scale_for_col:
                scale_for_col[spec.column] = variable_scales.get(spec.column, 1.0)

        # Desired = original x scale, then renormalize rows to sum = 1
        original  = df_out[all_cols].values.astype(float)
        scale_vec = np.array([scale_for_col[c] for c in all_cols])
        desired   = original * scale_vec          # broadcast over rows
        row_sums  = desired.sum(axis=1, keepdims=True)
        row_sums  = np.where(row_sums > 1e-10, row_sums, 1.0)  # guard /0
        df_out[all_cols] = desired / row_sums

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
        for level in levels:
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


def _resolve_registry(models, iea_crosswalk) -> SimplexRegistry:
    """Best-effort derivation of a SimplexRegistry from whatever context is available.

    Searched in order:
        1. models.model_attributes
        2. iea_crosswalk.model_attributes
        3. empty registry (no simplex groups known)

    The empty fallback is safe for scalar-only workflows: it only becomes a
    problem when a simplex spec is encountered, at which point
    apply_perturbations raises a clear error asking the caller to pass a
    registry explicitly.
    """
    for src in (models, iea_crosswalk):
        matt = getattr(src, "model_attributes", None)
        if matt is not None:
            return SimplexRegistry.from_model_attributes(matt)
    return SimplexRegistry.empty()


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
    simplex_registry : SimplexRegistry | None
        Authoritative lookup for simplex-group membership.  If not provided,
        the runner will try to build one from `models.model_attributes`; if
        that is not available (e.g. when running against a surrogate), the
        runner will fall back to an empty registry and raise at perturbation
        time if any simplex specs are encountered.
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
        simplex_registry: Optional[SimplexRegistry] = None,
    ) -> None:

        self.models                   = models
        self.df_baseline              = df_baseline.copy()
        self.iea_crosswalk            = iea_crosswalk
        self.df_iea_raw               = df_iea_raw
        self.iso                      = iso
        self.include_energy_production = include_energy_production
        self.year_min                 = year_min
        self.year_max                 = year_max
        self.simplex_registry         = (
            simplex_registry
            if simplex_registry is not None
            else _resolve_registry(models, iea_crosswalk)
        )

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
        specs: Optional[List[VariableSpec]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the model once with the given scale factors.

        Parameters
        ----------
        variable_scales : Dict[str, float]
            {column: scale_factor} for this run.
        specs : List[VariableSpec] | None
            Full spec list.  When provided, simplex-constrained variables are
            perturbed via perturb_inputs_simplex() instead of plain scaling.
            When None, all variables are scaled independently.

        Returns
        -------
        (df_out, df_comp)
            df_out  : full model output with year column attached
            df_comp : output of IEACrosswalk.build_comparison() (all years)
        """
        if specs is not None:
            df_in = apply_perturbations(
                self.df_baseline, specs, variable_scales,
                simplex_registry=self.simplex_registry,
            )
        else:
            df_in = perturb_inputs(self.df_baseline, variable_scales)
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

            df_out, df_comp = self._run_single(variable_scales, specs=specs)

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
      r > 0          -> increasing the variable increases the IEA ratio
      r < 0          -> increasing the variable decreases the IEA ratio

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
    output_col: str = "rel_err_iea",
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
    ax.axhline(0.0, color="grey", linewidth=0.9, linestyle="--",
               label="error = 0  (perfect match)")
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