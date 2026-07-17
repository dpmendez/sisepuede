"""
sisepuede/calibration/qp_phase2_surrogate.py

Sequential Quadratic Programming (SQP) optimizer for v3's production-side
energy calibration. Approach: consumption has already been calibrated by v2
and its fractions are frozen; this optimiser tunes production knobs against
the production-side IEA observations at that fixed consumption state.

The optimizer solves:

    min_f  ‖A * ĝ(f) - b_surr‖^2            (nonlinear via surrogate)
         + gamma * ‖f - f_default‖^2

    subject to
        lb <= f <= ub                          (VariableSpec box)

where:
- ĝ(f) is the surrogate's raw per-tech prediction, shape (T_ssp,)
- A is the crosswalk aggregation matrix, shape (T_iea, T_ssp), so
  A * ĝ(f) is the IEA-comparable prediction in TJ
- b_surr is the IEA-side observation vector, same shape as A * ĝ(f)
- f is production knobs only; consumption fractions are baked into the
  surrogate's training data as a fixed background

Because ĝ is nonlinear (GBM ensemble or other), the middle residual term makes
the QP non-convex. The SQP trick is to linearize ĝ around the current iterate
f_k using its Jacobian at f_k:

    ĝ(f) ≈ ĝ(f_k) + J_k * (f - f_k)
    J_iea = A * Surrogate.jacobian(f_k)

The middle term becomes ‖J_iea * f - (b_surr - g_k + J_iea * f_k)‖^2, a
linear residual identical in form to v2's Phase-2 QP. We solve one
convex QP per outer iteration and repeat until ‖f_next - f_k‖_inf < tol
or max_iter is hit.

Key design choices
------------------
- **Row-normalisation by |b_surr|**. Raw ELECTOUT residuals are in TJ
  and span 3+ orders of magnitude (WIND ~ 800 TJ vs HYDRO ~ 570 000 TJ).
  Without normalisation the optimiser would ignore small fuels and gamma
  would have to be tuned per country. Instead each row is divided by
  max(|b_surr[i]|, eps) so the residual is per-fuel relative error;
  rows with b_surr[i] ≈ 0 are auto-dropped by the mask.

- **Fixed-radius trust region**. `trust_radius` caps ‖f - f_k‖_inf per
  step. A ratio-test / adaptive Delta is deferred to v3.1 (see
  `_surrogate.py::Surrogate.jacobian`'s eps note).

- **Envelope guard**. `surrogate.in_envelope(f_next)` is checked every
  iteration; violations are recorded in diagnostics but do NOT halt the
  loop. The final verification run inside CalibratorV3 catches genuine
  extrapolation failures.

- **Gate-driven row dropping**. Callers may pass `accepted_iea_mask` to
  skip IEA target rows the surrogate isn't trusted on; the optimiser
  drops them from both the residual and the Jacobian.

Deferred to v3.1
----------------
- Ratio-test trust region (grow on prediction agreement, shrink on
  overshoot).
- Per-column adaptive eps for the surrogate Jacobian.
- Warm-start of the QP solver across outer iterations.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sisepuede.calibration._surrogate import Surrogate


# Value below which |b_surr[i]| counts as "essentially zero" and the row
# is auto-dropped from the residual (also avoids divide-by-zero in the
# row-normalisation). Units: TJ. 1e-6 TJ = 1 J — way below any real IEA
# observation.
DEFAULT_B_SURR_ZERO_TOL = 1e-6


def solve_phase2_surrogate(
    surrogate:         Surrogate,
    A:       np.ndarray,
    b_surr:            np.ndarray,
    *,
    f_default:         Optional[np.ndarray] = None,
    accepted_iea_mask: Optional[np.ndarray] = None,
    lb:                Optional[np.ndarray] = None,
    ub:                Optional[np.ndarray] = None,
    gamma:             float = 1.0,
    trust_radius:      float = 0.05,
    max_iter:          int   = 10,
    tol:               float = 1e-4,
    b_surr_zero_tol:   float = DEFAULT_B_SURR_ZERO_TOL,
    normalize_by_b:    bool  = True,
    solver:            str   = "OSQP",
    verbose:           bool  = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve v3's production-side QP via SQP.

    Parameters
    ----------
    surrogate : Surrogate
        Trained surrogate whose targets are raw per-technology SSP fields.
    A : ndarray, shape (T_iea, T_ssp)
        Aggregation matrix from raw SSP outputs to IEA-comparable TJ
        values. Column order must match `surrogate.targets`.
    b_surr : ndarray, shape (T_iea,)
        IEA-side observations (TJ) for each of the T_iea target rows.
    f_default : ndarray, shape (n,) or None
        Starting iterate and regularisation anchor. None -> `np.ones(n)`
        (scale-factor baseline). Must have length `len(surrogate.columns)`.
    accepted_iea_mask : ndarray of bool, shape (T_iea,), or None
        Rows the caller wants the optimiser to consider. None keeps all.
        Callers typically compute this from
        `surrogate.report.accepted_targets` via the crosswalk (see
        `iea_mask_from_accepted_targets`).
    lb, ub : ndarray, shape (n,), or None
        Box constraints on `f`. None -> `surrogate.envelope[min|max]`.
    gamma : float
        Regularisation weight on `‖f - f_default‖^2`. After
        row-normalisation the surrogate residual is O(1) per accepted
        row, so gamma ~ 1.0 is a reasonable starting point (contrast v2's
        gamma=100 in share space).
    trust_radius : float
        `‖f_next - f_k‖_inf ≤ trust_radius` per outer iteration.
    max_iter : int
        Outer-loop cap.
    tol : float
        Convergence tolerance on `‖f_next - f_k‖_inf`.
    b_surr_zero_tol : float
        Rows with `|b_surr[i]| < b_surr_zero_tol` are dropped (avoids
        divide-by-zero + those rows are uninformative anyway).
    normalize_by_b : bool
        When True, row-normalise the surrogate residual by
        `max(|b_surr[i]|, b_surr_zero_tol)`. See module docstring.
    solver : str
        cvxpy solver name. Default OSQP to match v2.
    verbose : bool
        Print per-iteration diagnostics.

    Returns
    -------
    f_star : ndarray, shape (n,)
        Optimised scale-factor vector.
    diagnostics : dict
        Keys:
          `status`        : "converged" | "max_iter" | "no_targets"
          `iters`         : outer iterations run
          `history`       : list of per-iter dicts (step norm, residual
                            norms, in-envelope flag)
          `dropped_iea`   : list of IEA row indices auto-dropped
                            (mask + b_surr_zero_tol)
          `envelope_violations` : list of iters where the step landed
                            outside the LHS box
    """
    n = len(surrogate.columns)
    if f_default is None:
        f_default = np.ones(n, dtype=float)
    else:
        f_default = np.asarray(f_default, dtype=float).reshape(-1)
        if f_default.shape[0] != n:
            raise ValueError(
                f"solve_phase2_surrogate: f_default length {f_default.shape[0]} "
                f"!= n={n} (surrogate columns)."
            )

    A = np.asarray(A, dtype=float)
    b    = np.asarray(b_surr,      dtype=float).reshape(-1)
    if A.shape[1] != len(surrogate.targets):
        raise ValueError(
            f"solve_phase2_surrogate: A has {A.shape[1]} columns "
            f"but surrogate.targets has {len(surrogate.targets)}. Column "
            f"order must match."
        )
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"solve_phase2_surrogate: A has {A.shape[0]} rows "
            f"but b_surr has {b.shape[0]} entries."
        )

    # ── Assemble the effective IEA-row mask ─────────────────────────────
    # Rows are auto-dropped when:
    #   - accepted_iea_mask says the surrogate isn't trusted there, or
    #   - b_surr[i] is NaN (country doesn't report this fuel to IEA), or
    #   - |b_surr[i]| < zero_tol (essentially-zero observation; also
    #     avoids divide-by-zero in the row-normalisation).
    T_iea = A.shape[0]
    keep = np.ones(T_iea, dtype=bool)
    if accepted_iea_mask is not None:
        keep &= np.asarray(accepted_iea_mask, dtype=bool)
    nan_b  = np.isnan(b)
    zero_b = ~nan_b & (np.abs(b) < b_surr_zero_tol)
    dropped_iea = np.where(nan_b | zero_b | ~keep)[0].tolist()
    keep &= ~(nan_b | zero_b)

    if not keep.any():
        # Nothing to optimise against.
        return f_default.copy(), {
            "status":               "no_targets",
            "iters":                0,
            "history":              [],
            "dropped_iea":          dropped_iea,
            "envelope_violations":  [],
        }

    A_used = A[keep]     # (T_used, T_ssp)
    b_used = b[keep]     # (T_used,)

    # ── Row-normalise so residuals are ~O(1) ────────────────────────────
    if normalize_by_b:
        scale = np.maximum(np.abs(b_used), b_surr_zero_tol)      # (T_used,)
        A_used = A_used / scale[:, None]
        b_used    = b_used    / scale

    # ── Box constraints default to the envelope ────────────────────────
    if lb is None:
        lb = surrogate.envelope["min"].to_numpy(dtype=float)
    else:
        lb = np.asarray(lb, dtype=float).reshape(-1)
    if ub is None:
        ub = surrogate.envelope["max"].to_numpy(dtype=float)
    else:
        ub = np.asarray(ub, dtype=float).reshape(-1)

    # ── SQP outer loop ─────────────────────────────────────────────────
    f_k                  = f_default.copy()
    history:             List[Dict[str, Any]] = []
    envelope_violations: List[int]            = []

    for k in range(max_iter):
        # 1. Surrogate + Jacobian at f_k, aggregated to used IEA rows.
        y_ssp_k    = surrogate.predict(f_k[np.newaxis, :])[0]  # (T_ssp,)
        J_ssp_k    = surrogate.jacobian(f_k)                   # (T_ssp, n)
        g_iea_k    = A_used @ y_ssp_k                          # (T_used,)
        J_iea_k    = A_used @ J_ssp_k                          # (T_used, n)
        b_local    = b_used - g_iea_k + J_iea_k @ f_k          # (T_used,)

        # 2. Solve one QP.
        f_next = _solve_one_qp(
            J_iea_k, b_local,
            f_default    = f_default,
            gamma        = gamma,
            lb           = lb,
            ub           = ub,
            trust_anchor = f_k,
            trust_radius = trust_radius,
            solver       = solver,
        )

        # 3. Diagnostics.
        step_norm     = float(np.linalg.norm(f_next - f_k, np.inf))
        residual_surr = float(np.sum((A_used @ y_ssp_k - b_used) ** 2))
        in_env, mask_env = surrogate.in_envelope(f_next, return_mask=True)
        if not in_env:
            envelope_violations.append(k + 1)

        entry: Dict[str, Any] = {
            "iter":            k + 1,
            "step_inf_norm":   step_norm,
            "residual_surr":   residual_surr,
            "in_envelope":     bool(in_env),
        }
        history.append(entry)
        if verbose:
            print(
                f"  iter {k+1:>3}  step={step_norm:.4e}  "
                f"|surr|^2={residual_surr:.4e}  in_env={in_env}"
            )

        f_k = f_next
        if step_norm < tol:
            return f_k, {
                "status":               "converged",
                "iters":                k + 1,
                "history":              history,
                "dropped_iea":          dropped_iea,
                "envelope_violations":  envelope_violations,
            }

    return f_k, {
        "status":               "max_iter",
        "iters":                max_iter,
        "history":              history,
        "dropped_iea":          dropped_iea,
        "envelope_violations":  envelope_violations,
    }


##########################
#    INNER QP            #
##########################

def _solve_one_qp(
    A:              np.ndarray,
    b:              np.ndarray,
    f_default:      np.ndarray,
    *,
    gamma:          float,
    lb:             Optional[np.ndarray],
    ub:             Optional[np.ndarray],
    trust_anchor:   np.ndarray,
    trust_radius:   float,
    solver:         str,
) -> np.ndarray:
    """One convex QP: minimise ‖A f - b‖^2 + gamma ‖f - f_default‖^2 s.t.
    box + trust-region constraints.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:                          # pragma: no cover
        raise ImportError(
            "solve_phase2_surrogate requires cvxpy. Install with "
            "`pip install cvxpy`."
        ) from exc

    n = A.shape[1]
    f = cp.Variable(n)

    objective = cp.Minimize(
        cp.sum_squares(A @ f - b)
        + gamma * cp.sum_squares(f - f_default)
    )

    constraints = []
    if lb is not None:
        constraints.append(f >= lb)
    if ub is not None:
        constraints.append(f <= ub)
    constraints.append(cp.norm_inf(f - trust_anchor) <= trust_radius)

    problem = cp.Problem(objective, constraints)
    try:
        cp_solver = getattr(cp, solver)
    except AttributeError as exc:
        raise ValueError(
            f"_solve_one_qp: cvxpy has no solver named {solver!r}. "
            "Try one of: OSQP, CLARABEL, ECOS, SCS."
        ) from exc

    problem.solve(solver=cp_solver)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"_solve_one_qp: cvxpy status = {problem.status!r}. "
            "The linearised sub-problem is infeasible or the solver failed. "
            "Try a smaller trust_radius, larger gamma, or a different solver."
        )

    return np.asarray(f.value).reshape(-1)


##########################
#    UTILITIES           #
##########################

def iea_mask_from_accepted_targets(
    accepted_targets: List[str],
    ssp_columns:      List[str],
    A:      np.ndarray,
) -> np.ndarray:
    """Build a `(T_iea,)` bool mask from the gate's accepted-targets list.

    An IEA row is accepted iff every raw SSP column it aggregates is in
    `accepted_targets`. Rejected components pollute the aggregated
    prediction, so this is deliberately conservative.

    Parameters
    ----------
    accepted_targets : List[str]
        Names of raw SSP columns the accuracy gate accepted (from
        `SurrogateReport.accepted_targets` after `apply_accuracy_gate`).
    ssp_columns : List[str]
        Column order matching `A`'s columns.
    A : ndarray, shape (T_iea, T_ssp)
        Aggregation matrix from `crosswalk_ssp_columns_for_iea_targets`.

    Returns
    -------
    ndarray of bool, shape (T_iea,)
    """
    accepted_set = set(accepted_targets)
    T_iea, T_ssp = A.shape
    if T_ssp != len(ssp_columns):
        raise ValueError(
            f"iea_mask_from_accepted_targets: A has {T_ssp} "
            f"columns but ssp_columns has {len(ssp_columns)}."
        )

    mask = np.ones(T_iea, dtype=bool)
    for i in range(T_iea):
        for j in range(T_ssp):
            if A[i, j] != 0.0 and ssp_columns[j] not in accepted_set:
                mask[i] = False
                break
    return mask
