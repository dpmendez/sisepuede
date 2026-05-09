"""
sisepuede/calibration/qp_phase2.py

Phase-2 fuel-mix calibration via Quadratic Programming (v2)
-----------------------------------------------------------
Replaces v1's per-fraction direct ratio with a single QP that:

    - matches the IEA target shares for every (sector, fuel) cell
      simultaneously
    - regularises the solution toward the existing model defaults so
      under-determined many-to-one mappings (e.g. several SSP fuels ->
      one IEA "OIL" category) have a well-posed answer
    - enforces sum-to-1 per simplex group and non-negativity

Mathematical formulation
------------------------
    minimise   ||A f - b||^2  +  gamma * ||f - f_default||^2
    subject to sum_{j in g} f_j = 1   for every simplex group g
               f >= 0

where for every IEA target k = (sector S, fuel F):

    A[k, j] = E_s / E_S            if frac column j is in subsector s subset S
                                   and its SSP fuel suffix maps to IEA fuel F
            = 0                    otherwise
    b[k]    = C^IEA_{S,F} / C^IEA_{S,S}     (dimensionless target share)

and `f_default` is the current value of every frac column at
`year_target` (i.e. just before Phase 2 runs). E_S is the sector total
after Phase 1, E_s is the per-subsector total from the same model run.

Why `b` is normalised by the IEA sector total (and not by E_S as written
in an earlier draft of the technote): both `A @ f` and `b` are then
dimensionless shares in [0, 1] and the QP matches "model share of fuel F
within S" against "IEA share of fuel F within S". Normalising `b` by
`E_S` instead would bake the Phase-1 residual into Phase 2 (the QP would
push fractions to compensate for any Phase-1 over/under-estimate of the
sector total) and would also need an explicit TJ <-> PJ conversion since
C^IEA is in TJ while E_S is in PJ. Iteration already closes the small
Phase-1/Phase-2 coupling.

Wiring back into the model
--------------------------
The QP returns a vector of fractions at the calibration year only. To
write the solution back to df_in we follow the same approach as v1:
compute scale_j = f_solved[j] / f_default[j] and apply it across all
time periods, letting Aitchison renormalisation (apply_perturbations)
restore the simplex constraint. Columns whose default is essentially
zero are skipped with a warning -- scaling cannot recover from zero.

Solver
------
cvxpy + OSQP
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sisepuede.calibration._df_helpers import sum_fields_at_time_period
from sisepuede.calibration._iea_fuel_map import FUEL_SUFFIX_TO_IEA, IEA_FUEL_MAP
from sisepuede.calibration._simplex_registry import SimplexRegistry
from sisepuede.calibration.calibration_group import CalibrationPlan
from sisepuede.calibration.sensitivity import VariableSpec, apply_perturbations


# Order fuel suffixes longest-first so e.g. "hydrocarbon_gas_liquids" is
# matched before "gas". Built once at import time.
_FUEL_SUFFIXES_LONGEST_FIRST: List[str] = sorted(
    FUEL_SUFFIX_TO_IEA.keys(), key=len, reverse=True
)


# Map IEA balance code -> (frac column prefix, SSP subsector total prefix).
# The frac prefix identifies the input columns whose fractions sum to 1
# within each simplex group; the SSP prefix identifies model-output fields
# from which we compute per-subsector total energy (E_s).
#
# AGRICULT is intentionally absent: it is a scalar Phase-1 target with no
# fuel-mix simplex columns of its own. Agriculture's frac columns are named
# frac_inen_energy_agriculture_and_livestock_{fuel} and therefore route to
# INDUSTRY via the `frac_inen_energy_` prefix; the IEA crosswalk has no
# (AGRICULT, FUEL) target to score against in the QP.
_BALANCE_TO_PATTERNS: Dict[str, Dict[str, str]] = {
    "INDUSTRY": {
        "frac_prefix":   "frac_inen_energy_",
        "ssp_total_fmt": "energy_consumption_inen_{sub}",
    },
    "TRANSPORT": {
        "frac_prefix":   "frac_trns_fuelmix_",
        "ssp_total_fmt": "energy_consumption_trns_{sub}_",  # any fuel suffix
    },
    "RESIDENT": {
        "frac_prefix":   "frac_scoe_heat_energy_residential_",
        "ssp_total_fmt": "energy_consumption_scoe_residential",
    },
    "COMMPUB": {
        "frac_prefix":   "frac_scoe_heat_energy_commercial_municipal_",
        "ssp_total_fmt": "energy_consumption_scoe_commercial_municipal",
    },
}


##########################
#    PARSING             #
##########################

def parse_frac_column(col: str) -> Optional[Tuple[str, str, str, str]]:
    """Return (balance_code, subsector, ssp_fuel_suffix, iea_fuel) or None.

    Column-name conventions are documented in build_energy_calibration_plan.
    The fuel suffix can itself contain underscores (e.g.
    "hydrocarbon_gas_liquids", "natural_gas", "solid_biomass"), so we strip
    the longest matching known suffix from the end first, then read off the
    remaining body as the subsector key.
    """
    body = None
    balance = None
    for bal, pat in _BALANCE_TO_PATTERNS.items():
        if col.startswith(pat["frac_prefix"]):
            body = col[len(pat["frac_prefix"]):]
            balance = bal
            break
    if body is None:
        return None

    for suf in _FUEL_SUFFIXES_LONGEST_FIRST:
        if body == suf:
            # No subsector segment (e.g. SCOE residential single-cat columns
            # already have the cat folded into the prefix -> body is just
            # the fuel suffix).
            return (balance, "", suf, FUEL_SUFFIX_TO_IEA[suf])
        tail = "_" + suf
        if body.endswith(tail):
            sub = body[: -len(tail)]
            return (balance, sub, suf, FUEL_SUFFIX_TO_IEA[suf])
    return None


##########################
#    BUILD QP INPUTS     #
##########################

def _subsector_total(
    df_out: pd.DataFrame,
    time_period: int,
    balance: str,
    subsector: str,
) -> float:
    """E_s for one (balance, subsector). Returns 0.0 if no fields match."""
    pat = _BALANCE_TO_PATTERNS.get(balance)
    if pat is None:
        return 0.0

    fmt = pat["ssp_total_fmt"]

    if balance == "TRANSPORT":
        # No single per-mode total in SSP outputs; sum every fuel-specific
        # field for this mode: energy_consumption_trns_{mode}_*
        prefix = fmt.format(sub=subsector)
        fields = [c for c in df_out.columns if c.startswith(prefix)]
        return sum_fields_at_time_period(df_out, time_period, fields)

    if balance == "INDUSTRY":
        field = fmt.format(sub=subsector)
        return sum_fields_at_time_period(df_out, time_period, [field])

    # RESIDENT / COMMPUB: a single SSP total per balance (no per-subsector split).
    # AGRICULT is intentionally not handled here -- see the comment on
    # _BALANCE_TO_PATTERNS for why.
    return sum_fields_at_time_period(df_out, time_period, [fmt])


def build_qp_inputs(
    plan: CalibrationPlan,
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    time_period: int,
    crosswalk,
    iea_tj: Dict[Tuple[str, str], float],
    simplex_registry: SimplexRegistry,
) -> Dict[str, object]:
    """Build the matrices and vectors for the Phase-2 QP.

    Iterates over plan.simplex_groups(): each such group corresponds to one
    IEA target (S, F) and contributes one row of A and one entry of b.
    Columns of A are the union of all frac columns referenced by simplex
    groups whose targets we can score.

    Returns
    -------
    dict with keys:
        columns         : List[str]             order of columns in A and f
        A               : np.ndarray (m, n)
        b               : np.ndarray (m,)
        f_default       : np.ndarray (n,)
        sector_indices  : List[np.ndarray]      column-index arrays per simplex
                                                group id (for sum-to-1 constraints)
        target_keys     : List[Tuple[str,str]]  the IEA targets indexing rows of A
        skipped         : List[dict]            diagnostic records for dropped groups
    """
    # 1. Pick eligible simplex groups (have IEA target + crosswalk + total)
    eligible: List[Tuple[object, str, str, float, float, List[str]]] = []
    skipped: List[dict] = []

    for group in plan.simplex_groups():
        if not group.iea_targets:
            skipped.append({"group": group.name, "phase": 2,
                            "status": "skipped_no_targets"})
            continue

        bal, prod = group.iea_targets[0]

        if bal not in _BALANCE_TO_PATTERNS:
            # QP currently scopes consumption-side balances only.
            skipped.append({"group": group.name, "phase": 2,
                            "status": "skipped_unsupported_balance",
                            "target": (bal, prod)})
            continue

        iea_fuel_tj  = iea_tj.get((bal, prod))
        iea_total_tj = iea_tj.get((bal, bal))
        if iea_fuel_tj is None or iea_total_tj is None or iea_total_tj == 0.0:
            skipped.append({"group": group.name, "phase": 2,
                            "status": "skipped_no_iea", "target": (bal, prod)})
            continue

        # Sector total E_S (in SSP units / PJ) from current model output
        entry = crosswalk.get_crosswalk_entry(bal, bal)
        if entry is None:
            skipped.append({"group": group.name, "phase": 2,
                            "status": "skipped_no_sector_total_xw",
                            "target": (bal, prod)})
            continue
        ssp_total_fields = entry["ssp_fields"]
        E_S = sum_fields_at_time_period(df_out, time_period, ssp_total_fields)
        if E_S <= 0.0:
            skipped.append({"group": group.name, "phase": 2,
                            "status": "skipped_zero_sector_total",
                            "target": (bal, prod)})
            continue

        target_share = iea_fuel_tj / iea_total_tj  # b[k]
        cols_in_group = [c for c in group.columns if c in df_in.columns]
        if not cols_in_group:
            skipped.append({"group": group.name, "phase": 2,
                            "status": "skipped_no_input_cols",
                            "target": (bal, prod)})
            continue

        eligible.append((group, bal, prod, target_share, E_S, cols_in_group))

    if not eligible:
        return {
            "columns": [], "A": np.zeros((0, 0)), "b": np.zeros((0,)),
            "f_default": np.zeros((0,)),
            "lb_per_col": np.zeros((0,)), "ub_per_col": np.zeros((0,)),
            "sector_indices": [], "target_keys": [], "skipped": skipped,
        }

    # 2. Column ordering: union of all eligible columns, deduped ───────────
    seen: set = set()
    columns: List[str] = []
    for _, _, _, _, _, cols in eligible:
        for c in cols:
            if c not in seen:
                seen.add(c)
                columns.append(c)
    col_idx = {c: i for i, c in enumerate(columns)}
    n = len(columns)

    # 3. Cache E_s per (balance, subsector) so we don't recompute ──────────
    es_cache: Dict[Tuple[str, str], float] = {}

    def _e_s(bal: str, sub: str) -> float:
        key = (bal, sub)
        if key not in es_cache:
            es_cache[key] = _subsector_total(df_out, time_period, bal, sub)
        return es_cache[key]

    # 4. Build A and b ──────────────────────────────────────────────────────
    m = len(eligible)
    A = np.zeros((m, n))
    b = np.zeros(m)
    target_keys: List[Tuple[str, str]] = []

    for k, (group, bal, prod, target_share, E_S, cols_in_group) in enumerate(eligible):
        b[k] = target_share
        target_keys.append((bal, prod))

        for c in cols_in_group:
            parsed = parse_frac_column(c)
            if parsed is None:
                continue
            _, sub, _, _ = parsed
            E_s = _e_s(bal, sub)
            if E_s <= 0.0:
                # Subsector not present in the model output: weight is 0,
                # contributing nothing to this row. Regularization will pin
                # the column to its default.
                continue
            A[k, col_idx[c]] = E_s / E_S

    # 5. f_default at year_target ───────────────────────────────────────────
    f_default = np.array(
        [sum_fields_at_time_period(df_in, time_period, [c]) for c in columns],
        dtype=float,
    )

    # 5b. Per-column scale-factor bounds from VariableSpec ──────────────────
    # First spec encountered per column wins (a column may appear in more than
    # one simplex group; bounds are expected to match in practice).
    col_to_spec: Dict[str, VariableSpec] = {}
    for group, _, _, _, _, _ in eligible:
        for s in group.specs:
            if s.column in col_idx and s.column not in col_to_spec:
                col_to_spec[s.column] = s
    lb_per_col = np.array(
        [col_to_spec[c].lb if c in col_to_spec else 0.0 for c in columns],
        dtype=float,
    )
    ub_per_col = np.array(
        [col_to_spec[c].ub if c in col_to_spec else float("inf") for c in columns],
        dtype=float,
    )

    # 6. sector_indices: one column-index array per simplex group id ───────
    # All columns in `columns` belong to some simplex group via the registry.
    # The QP enforces sum=1 over every group whose columns appear here.
    gid_to_idx: Dict[int, List[int]] = {}
    for c, j in col_idx.items():
        gid = simplex_registry.group_id(c)
        if gid is None:
            continue
        gid_to_idx.setdefault(gid, []).append(j)
    sector_indices = [np.array(sorted(v)) for v in gid_to_idx.values()]

    return {
        "columns":        columns,
        "A":              A,
        "b":              b,
        "f_default":      f_default,
        "lb_per_col":     lb_per_col,
        "ub_per_col":     ub_per_col,
        "sector_indices": sector_indices,
        "target_keys":    target_keys,
        "skipped":        skipped,
    }


##########################
#    SOLVER              #
##########################

def solve_phase2_qp(
    A: np.ndarray,
    b: np.ndarray,
    f_default: np.ndarray,
    sector_indices: List[np.ndarray],
    gamma: float = 100.0,
    enforce_varspec_bounds: bool = False,
    lb_per_col: Optional[np.ndarray] = None,
    ub_per_col: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve the Phase-2 QP.

    Parameters
    ----------
    A              : ndarray (m, n)    aggregation matrix
    b              : ndarray (m,)      IEA target shares
    f_default      : ndarray (n,)      regularisation anchor
    sector_indices : list of arrays    column indices per simplex group
    gamma          : float             regularisation weight
    enforce_varspec_bounds : bool
        When True, add per-column scale-factor bounds taken from VariableSpec:
            f[j] >= lb_per_col[j] * f_default[j]
            f[j] <= ub_per_col[j] * f_default[j]
        These are the same bounds LHS uses for sampling, so calibration and
        sensitivity stay coherent. Default False (the v2 plan as written
        relies on `gamma` alone).
    lb_per_col, ub_per_col : ndarray (n,) | None
        Per-column scale factors. Required when `enforce_varspec_bounds` is
        True. Build them with `build_qp_inputs`, which pulls them from the
        VariableSpec attached to each frac column.

    Returns
    -------
    ndarray (n,)  optimised fraction vector

    Raises
    ------
    RuntimeError if the solver does not converge. Hard bounds + tight gamma
    can produce infeasibility; the message points at both knobs.
    ImportError  if cvxpy is not installed.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Phase-2 QP calibration requires cvxpy. "
            "Install it with `pip install cvxpy` or update requirements.txt."
        ) from exc

    n = len(f_default)
    if n == 0:
        return np.zeros(0)

    f = cp.Variable(n)
    objective = cp.Minimize(
        cp.sum_squares(A @ f - b)
        + gamma * cp.sum_squares(f - f_default)
    )
    constraints = [f >= 0]
    for idx in sector_indices:
        if len(idx) == 0:
            continue
        constraints.append(cp.sum(f[idx]) == 1)

    if enforce_varspec_bounds:
        if lb_per_col is None or ub_per_col is None:
            raise ValueError(
                "enforce_varspec_bounds=True but lb_per_col/ub_per_col were "
                "not supplied. Use build_qp_inputs() to populate them."
            )
        if len(lb_per_col) != n or len(ub_per_col) != n:
            raise ValueError(
                f"lb_per_col/ub_per_col length mismatch: expected {n}, got "
                f"{len(lb_per_col)}/{len(ub_per_col)}."
            )
        constraints.append(f >= lb_per_col * f_default)
        constraints.append(f <= ub_per_col * f_default)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    if problem.status == "infeasible" and enforce_varspec_bounds:
        raise RuntimeError(
            "Phase 2 QP is infeasible with enforce_varspec_bounds=True. "
            "IEA targets demand fraction moves that exceed the per-column "
            "lb/ub on the VariableSpecs. Either widen the bounds (pass "
            "larger lb/ub when building the plan), lower `gamma`, or set "
            "enforce_varspec_bounds=False to fall back on soft "
            "regularisation only."
        )
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Phase 2 QP did not converge: {problem.status}")

    return np.asarray(f.value).reshape(-1)


##########################
#    APPLY SOLUTION      #
##########################

def apply_qp_solution(
    df_in: pd.DataFrame,
    columns: List[str],
    f_solved: np.ndarray,
    f_default: np.ndarray,
    plan: CalibrationPlan,
    simplex_registry: SimplexRegistry,
    zero_default_eps: float = 1e-12,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Write the QP solution back to df_in via Aitchison renormalisation.

    For each column j, scale_j = f_solved[j] / f_default[j] is applied across
    all time periods, then per-simplex-group renormalisation restores
    sum-to-1 (this is exactly the v1 mechanism, only the scale comes from
    the QP rather than from a per-fraction ratio).

    Columns whose default is essentially zero are dropped with a warning:
    multiplicative scaling cannot lift a fraction off zero. If the QP wants
    a non-zero share for such a column, a follow-up version will need to
    set it directly. v2 leaves these untouched.

    Returns the perturbed DataFrame and a per-column log.
    """
    if len(columns) == 0:
        return df_in.copy(), []

    # Map column -> spec (any spec object will do; we just need the simplex
    # metadata that apply_perturbations consults).
    spec_map: Dict[str, VariableSpec] = {}
    for group in plan.simplex_groups():
        for s in group.specs:
            if s.column in columns and s.column not in spec_map:
                spec_map[s.column] = s

    variable_scales: Dict[str, float] = {}
    log: List[dict] = []

    for c, f_s, f_d in zip(columns, f_solved, f_default):
        if abs(f_d) <= zero_default_eps:
            if f_s > zero_default_eps:
                warnings.warn(
                    f"Phase 2 QP — '{c}': default share is ~0 but QP wants "
                    f"{f_s:.4f}. Multiplicative scaling cannot lift a "
                    "fraction off zero; column left unchanged."
                )
            log.append({
                "column":  c, "phase": 2, "status": "skipped_zero_default",
                "default": float(f_d), "solved": float(f_s),
            })
            continue

        scale = f_s / f_d
        # Skip identity scales (saves apply_perturbations work)
        if abs(scale - 1.0) < 1e-12:
            log.append({
                "column":  c, "phase": 2, "status": "ok_identity",
                "default": float(f_d), "solved": float(f_s),
                "scale":   1.0,
            })
            continue

        variable_scales[c] = scale
        log.append({
            "column":  c, "phase": 2, "status": "ok",
            "default": float(f_d), "solved": float(f_s),
            "scale":   float(scale),
        })

    if not variable_scales:
        return df_in.copy(), log

    specs_for_apply: List[VariableSpec] = [
        spec_map[c] for c in variable_scales if c in spec_map
    ]
    df_out = apply_perturbations(
        df_in, specs_for_apply, variable_scales,
        simplex_registry=simplex_registry,
    )
    return df_out, log
