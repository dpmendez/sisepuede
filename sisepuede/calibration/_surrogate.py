"""
sisepuede/calibration/_surrogate.py

Surrogate model for v3 production-side energy calibration.

What this module provides:
- A `Surrogate` class wrapping one regressor per IEA target row, fit to
  the (X, Y) matrices produced by `_training_set.build_training_matrices`.
- `predict(f)` and `jacobian(f)` shaped to slot directly into the SQP
  outer loop in `qp_phase2_surrogate.solve_phase2_surrogate`.
- An envelope check that flags when the optimizer's iterate leaves the
  LHS sampling box (where the surrogate is interpolating; outside, it's
  extrapolating).
- Persistence (save/load via joblib) keyed on a consumption-state
  fingerprint, so v3 refuses to apply a surrogate trained at a different
  consumption-side configuration.

This module is pure ML — no SISEPUEDE imports, no model runs. The ML
lifecycle stages (train / evaluate / gate) are separable, callable
independently from the train_surrogate CLI. There is no `fit()`
convenience wrapper; callers must compose the primitives explicitly.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _SKLEARN_AVAILABLE = True
except ImportError:                          # pragma: no cover
    _SKLEARN_AVAILABLE = False

# Optional boosting backends. We import lazily inside _build_estimator so
# that v3 works on machines without XGBoost / LightGBM installed; the
# Surrogate only fails when a caller actually picks one of those backends
# via SurrogateSpec.model_kind.
try:
    import xgboost as _xgb            # noqa: F401  imported lazily later
    _XGB_AVAILABLE = True
except ImportError:                          # pragma: no cover
    _XGB_AVAILABLE = False

try:
    import lightgbm as _lgbm          # noqa: F401  imported lazily later
    _LGBM_AVAILABLE = True
except ImportError:                          # pragma: no cover
    _LGBM_AVAILABLE = False

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:                          # pragma: no cover
    _JOBLIB_AVAILABLE = False

# Supported regression backends, ordered by recommendation.
SUPPORTED_MODEL_KINDS = ("gbm", "xgb", "lgbm")


##########################
#    CONFIG DATACLASSES  #
##########################

@dataclass
class SurrogateSpec:
    """Hyperparameters and fit-time options for a Surrogate.

    Attributes
    ----------
    model_kind : str
        Currently supported: "gbm" (HistGradientBoostingRegressor). Other
        backends (random forest, linear, MLP) can be added; the rest of
        the v3 pipeline only assumes `predict()` / `jacobian()`.
    hyperparams : dict
        Forwarded to the regressor constructor. Defaults pinned in
        `_default_hyperparams_for(model_kind)`.
    holdout_frac : float
        Fraction of rows held out for evaluating R^2 / MAPE per target.
        Set to 0.0 to fit on all rows (no holdout; report values will be
        in-sample).
    seed : int
        Random seed for both holdout split and regressor RNG.
    per_target_models : bool
        Always True for v3 (one regressor per target row). Kept as an
        attribute so future multi-output backends can flip it.
    """

    model_kind:        str   = "gbm"
    hyperparams:       Dict[str, Any] = field(default_factory=dict)
    holdout_frac:      float = 0.2
    seed:              int   = 42
    per_target_models: bool  = True

    # --- Accuracy gate (step 4) ---
    target_r2_min:     float = 0.85    # accept target if holdout R^2 >= this
    target_mape_max:   float = 20.0    # accept target if holdout MAPE (%) <= this
    fail_mode:         str   = "warn"  # "warn" (log + continue) or "raise"

    # TODO(v3.1): LHS budget escalation. Add n_lhs_initial / n_lhs_step /
    # n_lhs_max and a retrospective learning-curve helper so the
    # orchestrator can adaptively extend LHS when the gate rejects too
    # many priority targets. Deferred until step 8 gives us evidence
    # about how bad the fixed-budget case really is.


@dataclass
class SurrogateReport:
    """Per-target accuracy report returned alongside a fitted Surrogate.

    Attributes
    ----------
    targets : List[Tuple[str, str]]
        Target rows the surrogate predicts, in column order.
    r2_per_target : pd.Series
        Holdout R^2 per target (or in-sample R^2 when holdout_frac=0).
        Index is target row, values are floats in [-inf, 1.0]. NaN when
        the target is constant (variance = 0) and R^2 is undefined.
    mape_per_target : pd.Series
        Holdout mean absolute percentage error per target. NaN for
        targets whose true value is identically zero.
    accepted_targets : List[Tuple[str, str]]
        Populated by the step-4 gating logic; left equal to `targets`
        here. The SQP solver uses this list to decide which rows of the
        surrogate residual to actually optimize against.
    rejected_targets : List[Tuple[str, str]]
        Likewise populated in step 4.
    n_train : int
        Number of rows used for fitting.
    n_holdout : int
        Number of rows used for evaluation. Zero when holdout_frac=0.
    """

    targets:          List[Tuple[str, str]]
    r2_per_target:    pd.Series
    mape_per_target:  pd.Series
    accepted_targets: List[Tuple[str, str]]
    rejected_targets: List[Tuple[str, str]]
    n_train:          int
    n_holdout:        int


##########################
#    SURROGATE CLASS     #
##########################

class Surrogate:
    """One regressor per IEA target row, plus a constant-output shortcut
    for targets whose training Y is identically zero (or any constant).

    The surrogate's coordinate system is:
      - input  : a length-n vector `f` of scale factors, with the same
                 column order as `Surrogate.columns` (matches the X frame
                 passed to `fit`).
      - output : a length-T vector of SSP-side predictions, with the same
                 row order as `Surrogate.targets` (matches Y's columns).

    Use `predict(f)` for a single point and `jacobian(f, eps=...)` for a
    central-finite-difference Jacobian (T x n) at `f`.

    See `Surrogate.train` for construction, `Surrogate.evaluate` /
    `apply_accuracy_gate` for the ML-lifecycle stages, and `save` /
    `load` for persistence.
    """

    # ---------------------- construction -------------------------------

    def __init__(
        self,
        columns:                 List[str],
        targets:                 List[Tuple[str, str]],
        per_target_estimators:   Dict[Tuple[str, str], Any],
        envelope:                pd.DataFrame,
        consumption_fingerprint: str,
        spec:                    SurrogateSpec,
        baseline_inputs:         Optional[np.ndarray] = None,
    ) -> None:
        self.columns                 = list(columns)
        self.targets                 = list(targets)
        self._estimators             = per_target_estimators
        self.envelope                = envelope          # DataFrame: index=columns, cols=['min','max']
        self.consumption_fingerprint = consumption_fingerprint
        self.spec                    = spec
        # Baseline (all scale factors = 1.0) is the natural starting point
        # for the SQP outer loop. Stored on the Surrogate so callers don't
        # have to recompute it.
        self.baseline_inputs = (
            np.ones(len(columns)) if baseline_inputs is None
            else np.asarray(baseline_inputs, dtype=float)
        )

    # ---------------------- public API ---------------------------------

    @classmethod
    def train(
        cls,
        X:                       pd.DataFrame,
        Y:                       pd.DataFrame,
        spec:                    SurrogateSpec,
        consumption_fingerprint: str,
        envelope_bounds:         Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "Surrogate":
        """Fit one regressor per target column of Y on the full (X, Y).

        No holdout split, no evaluation, no accuracy gate. Use
        `Surrogate.evaluate(X_eval, Y_eval)` on a separate dev/test set,
        then `apply_accuracy_gate(surrogate, report, spec)` to decide
        which targets survive.

        Parameters
        ----------
        X : DataFrame, shape (N_train, N_vars)
            Training inputs. Column order fixes the surrogate's
            coordinate system.
        Y : DataFrame, shape (N_train, N_targets)
            Training outputs per IEA target. Column labels are
            (balance, product) tuples; preserved as the surrogate's
            target order.
        spec : SurrogateSpec
        consumption_fingerprint : str
            Identifier of the consumption-side configuration the
            training data was generated under. Stored verbatim on the
            Surrogate; v3's orchestrator checks it before inference.
        envelope_bounds : (lb_array, ub_array) | None
            The LHS sampling box for `in_envelope` to check against,
            ordered to match X.columns. In v3 always pass the (lb, ub)
            arrays the SQP optimizer will use as its box constraints
            (built from VariableSpec.lb / ub). When None, falls back to
            the empirical (X.min, X.max) with a warning -- correct only
            as N_train -> infinity, and appropriate mostly for synthetic
            unit tests where no formal LHS box exists.

        Returns
        -------
        Surrogate
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "Surrogate.train requires scikit-learn. "
                "Install with `pip install scikit-learn`."
            )
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Surrogate.train: X and Y must have the same number of "
                f"rows (got X={X.shape[0]}, Y={Y.shape[0]})."
            )
        if X.shape[0] == 0:
            raise ValueError("Surrogate.train: empty training set.")

        columns = list(X.columns)
        targets = list(Y.columns)
        if not targets:
            raise ValueError(
                "Surrogate.train: Y has no columns. Pass at least one "
                "IEA target row."
            )

        X_train = X.to_numpy(dtype=float)

        per_target_estimators: Dict[Tuple[str, str], Any] = {}
        for t in targets:
            y_train = Y[t].to_numpy(dtype=float)

            # Constant-output shortcut. HistGradientBoosting would fit a
            # constant predictor anyway, but a `_ConstantRegressor` is
            # cheaper, exposes the constant to `jacobian()` cleanly (zero
            # gradient), and avoids R^2 = NaN noise in the report.
            if np.allclose(y_train.std(), 0.0):
                est = _ConstantRegressor(constant=float(y_train.mean()))
            else:
                est = _build_estimator(spec, target_label=t)
            est.fit(X_train, y_train)
            per_target_estimators[t] = est

        envelope = _build_envelope(X, columns, envelope_bounds)

        surrogate = cls(
            columns                 = columns,
            targets                 = targets,
            per_target_estimators   = per_target_estimators,
            envelope                = envelope,
            consumption_fingerprint = consumption_fingerprint,
            spec                    = spec,
        )
        # Book-keeping for downstream reports.
        surrogate._n_train_samples = int(X_train.shape[0])
        return surrogate

    def evaluate(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
    ) -> SurrogateReport:
        """Compute per-target R^2 and MAPE for (X, Y).

        This does NOT apply the accuracy gate -- call
        `apply_accuracy_gate(surrogate, report, spec)` for that. Returns
        raw metrics and leaves `accepted_targets` / `rejected_targets`
        empty in the report.

        Parameters
        ----------
        X : DataFrame, shape (N_eval, N_vars)
            Must have the same columns (and column order) as the
            training X.
        Y : DataFrame, shape (N_eval, N_targets)
            Column labels must be a subset of `self.targets`. Targets
            missing from Y appear as NaN in the report.

        Returns
        -------
        SurrogateReport
            r2_per_target and mape_per_target populated.
            accepted_targets / rejected_targets are empty (use
            apply_accuracy_gate to populate them).
            n_holdout = X.shape[0]; n_train = training set size.
        """
        if list(X.columns) != self.columns:
            raise ValueError(
                "Surrogate.evaluate: X.columns must match self.columns "
                "exactly (same names and order). Ensure the eval frame "
                "comes from the same feature build as training."
            )
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Surrogate.evaluate: X and Y must have the same number "
                f"of rows (got X={X.shape[0]}, Y={Y.shape[0]})."
            )

        Y_pred = self.predict(X.to_numpy(dtype=float))        # (N_eval, T)

        r2:   Dict[Tuple[str, str], float] = {}
        mape: Dict[Tuple[str, str], float] = {}
        for i, t in enumerate(self.targets):
            if t in Y.columns:
                y_true = Y[t].to_numpy(dtype=float)
                y_hat  = Y_pred[:, i]
                r2[t]   = _safe_r2(y_true, y_hat)
                mape[t] = _safe_mape(y_true, y_hat)
            else:
                r2[t]   = float("nan")
                mape[t] = float("nan")

        return SurrogateReport(
            targets          = list(self.targets),
            r2_per_target    = pd.Series(r2,   index=list(self.targets), dtype=float),
            mape_per_target  = pd.Series(mape, index=list(self.targets), dtype=float),
            accepted_targets = [],
            rejected_targets = [],
            n_train          = int(getattr(self, "_n_train_samples", -1)),
            n_holdout        = int(X.shape[0]),
        )

    def is_constant_target(self, t: Tuple[str, str]) -> bool:
        """True iff this target was fit by a constant regressor
        (training Y was zero-variance). Predictions are the training
        mean everywhere; gradient is zero. Used by `apply_accuracy_gate`
        to auto-accept these targets.
        """
        return isinstance(self._estimators.get(t), _ConstantRegressor)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict raw SSP-side outputs for a batch of inputs.

        Follows sklearn convention: input is always 2-D
        ``(n_samples, n_features)``, output is always 2-D
        ``(n_samples, n_targets)``. Callers evaluating at a single
        iterate reshape explicitly, e.g. ``surrogate.predict(f[None, :])[0]``.

        Parameters
        ----------
        X : ndarray, shape (k, n)
            Rows are scale-factor vectors. Column order must match
            ``self.columns``.

        Returns
        -------
        ndarray, shape (k, T)
            One row per input; columns are targets in ``self.targets``
            order.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.columns):
            raise ValueError(
                f"Surrogate.predict: expected shape (k, {len(self.columns)}), "
                f"got {X.shape}."
            )
        out = np.empty((X.shape[0], len(self.targets)), dtype=float)
        for i, t in enumerate(self.targets):
            out[:, i] = self._estimators[t].predict(X)
        return out

    def jacobian(
        self,
        f:   np.ndarray,
        eps: float = 5e-2,
    ) -> np.ndarray:
        """Central-finite-difference Jacobian at `f`.

        For each input column j, perturbs f[j] by +-eps and evaluates
        the surrogate at both points. Cost: one `predict()` batch of
        2n rows per target.

        Parameters
        ----------
        f : ndarray, shape (n,)
        eps : float
            Step size for the finite difference. For tree-based models
            (GBM, RF) the prediction surface is piecewise-constant in
            each input; a too-small `eps` lands inside a single leaf and
            returns 0, while a too-large `eps` averages over too many
            leaves and loses locality. Empirical sweep on the v3
            envelope: eps in [0.02, 0.10] recovers analytical gradients
            within ~10% on synthetic linear+sinusoidal targets.
            Default 5e-2 -- roughly a quarter of the v3 LHS box width
            ([0.9, 1.1] = 0.2 wide), matching the natural SQP step
            scale. Smooth backends (linear, MLP) can pass a much smaller
            `eps` without loss.

            TODO(v3.1): per-column eps calibration. Sweep eps in [1e-3,
            1e-1] on training data and pick the value that maximises
            gradient-recovery on a synthetic linear probe or that
            minimises Jacobian variance across the training envelope.
            Fixed 5e-2 is fine for now but likely suboptimal for some knob
            families.

        Returns
        -------
        J : ndarray, shape (T, n)
            Partial derivatives d g_t / d f_j evaluated at `f`. For
            constant-output targets, the corresponding row of J is zero.
        """
        f = np.asarray(f, dtype=float).reshape(-1)
        n = len(self.columns)
        if f.shape[0] != n:
            raise ValueError(
                f"Surrogate.jacobian: expected length-{n} input, got "
                f"{f.shape[0]}."
            )

        # Build 2n perturbed points: rows 0..n-1 are +eps, n..2n-1 are -eps.
        F = np.tile(f, (2 * n, 1))
        for j in range(n):
            F[j,     j] += eps
            F[n + j, j] -= eps

        Y_perturbed = self.predict(F)               # (2n, T)
        Y_plus  = Y_perturbed[:n]                   # (n,  T)
        Y_minus = Y_perturbed[n:]                   # (n,  T)
        # (Y_plus[j] - Y_minus[j]) / (2 eps)  is d g / d f_j  in shape (T,)
        # Stack as J[t, j] = (Y_plus[j, t] - Y_minus[j, t]) / (2 eps).
        J = ((Y_plus - Y_minus) / (2.0 * eps)).T    # (T, n)
        return J

    def in_envelope(
        self,
        f:           np.ndarray,
        slack:       float = 0.0,
        return_mask: bool = False,
    ):
        """Check whether `f` lies inside the LHS sampling box.

        Parameters
        ----------
        f : ndarray, shape (n,)
        slack : float
            Tolerance added to each side of each column's [min, max] box.
            Default 0.0 (strict).
        return_mask : bool
            When True, returns the per-column boolean mask along with the
            aggregate boolean.

        Returns
        -------
        bool or (bool, ndarray)
        """
        f = np.asarray(f, dtype=float).reshape(-1)
        if f.shape[0] != len(self.columns):
            raise ValueError(
                f"Surrogate.in_envelope: expected length-{len(self.columns)} "
                f"input, got {f.shape[0]}."
            )
        lo = self.envelope["min"].values - slack
        hi = self.envelope["max"].values + slack
        mask = (f >= lo) & (f <= hi)
        ok = bool(mask.all())
        return (ok, mask) if return_mask else ok

    # ---------------------- persistence --------------------------------

    def save(self, path: str) -> None:
        """Serialise the surrogate to disk via joblib.

        Notes
        -----
        joblib handles sklearn estimators (including numpy arrays) more
        efficiently than pickle. The saved file includes the regressors,
        envelope, baseline_inputs, spec, and consumption_fingerprint --
        enough to reproduce predictions exactly.
        """
        if not _JOBLIB_AVAILABLE:
            raise ImportError(
                "Surrogate.save requires joblib. Install with `pip install joblib`."
            )
        payload = {
            "columns":                 self.columns,
            "targets":                 self.targets,
            "estimators":              self._estimators,
            "envelope":                self.envelope,
            "consumption_fingerprint": self.consumption_fingerprint,
            "spec":                    self.spec,
            "baseline_inputs":         self.baseline_inputs,
            "n_train_samples":         getattr(self, "_n_train_samples", -1),
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "Surrogate":
        """Restore a Surrogate previously saved by `Surrogate.save`."""
        if not _JOBLIB_AVAILABLE:
            raise ImportError(
                "Surrogate.load requires joblib. Install with `pip install joblib`."
            )
        payload = joblib.load(path)
        surrogate = cls(
            columns                 = payload["columns"],
            targets                 = payload["targets"],
            per_target_estimators   = payload["estimators"],
            envelope                = payload["envelope"],
            consumption_fingerprint = payload["consumption_fingerprint"],
            spec                    = payload["spec"],
            baseline_inputs         = payload.get("baseline_inputs"),
        )
        surrogate._n_train_samples = int(payload.get("n_train_samples", -1))
        return surrogate

    def __repr__(self) -> str:
        return (
            f"Surrogate(n_inputs={len(self.columns)}, "
            f"n_targets={len(self.targets)}, "
            f"kind={self.spec.model_kind!r}, "
            f"fingerprint={self.consumption_fingerprint[:12]!r}...)"
        )


##########################
#    UTILITIES           #
##########################

def fingerprint_consumption_state(
    df_input: pd.DataFrame,
    frac_columns: Optional[List[str]] = None,
    time_period_col: str = "time_period",
) -> str:
    """SHA-256 hash of the consumption-side `frac_*` columns in a SISEPUEDE
    input DataFrame.

    Used by v3 to tag a Surrogate with the consumption configuration it
    was trained under. The orchestrator (ProductionCalibrator) recomputes this
    on the candidate calibrated DataFrame and refuses to apply the
    surrogate when the fingerprints disagree.

    Parameters
    ----------
    df_input : pd.DataFrame
        SISEPUEDE input DataFrame (typically the post-v2 consumption
        configuration).
    frac_columns : List[str] | None
        Which columns to hash. Defaults to every column whose name starts
        with "frac_" (i.e. all `frac_inen_*`, `frac_trns_*`, `frac_scoe_*`).
    time_period_col : str
        Column that orders the rows for hashing.

    Returns
    -------
    str
        Hex digest, 64 chars.
    """
    if frac_columns is None:
        frac_columns = sorted(c for c in df_input.columns if c.startswith("frac_"))
    if not frac_columns:
        # Nothing to hash. Use a sentinel so identical empty selections
        # collide rather than collide with NaN inputs.
        return "no-frac-columns"

    df = df_input[[time_period_col, *frac_columns]].sort_values(time_period_col)
    h = hashlib.sha256()
    # bytes-based hashing of float64 values + column order + time_period order.
    h.update(repr(frac_columns).encode("utf-8"))
    h.update(df[time_period_col].to_numpy().tobytes())
    h.update(df[frac_columns].to_numpy(dtype=np.float64).tobytes())
    return h.hexdigest()


##########################
#    INTERNAL HELPERS    #
##########################

def _default_hyperparams_for(model_kind: str) -> Dict[str, Any]:
    """Hyperparameter defaults per backend.

    All three boosting backends are tuned for the same regime: 200–2000
    LHS runs, 10–50 input knobs, smooth-ish nonlinear targets. They land
    within ~3% R^2 of each other on our problem; backend choice is
    governed by dependency availability and per-target empirical
    comparison in step 8.
    """
    if model_kind == "gbm":
        # min_samples_leaf=5 instead of HGB's default 20: v3 trains on
        # LHS budgets of ~500 runs, where 20 would force the model
        # to collapse to near-constant predictions. With 5 we still
        # avoid overfitting at the bottom of that range, and at the
        # top we get meaningful splits.
        return dict(
            max_iter          = 200,
            max_depth         = 4,
            learning_rate     = 0.08,
            min_samples_leaf  = 5,
            l2_regularization = 1e-2,
        )
    if model_kind == "xgb":
        # XGBoost defaults are calibrated for the same regime.
        # `tree_method="hist"` aligns its tree-building strategy with
        # HGB's so the comparison is fair.
        return dict(
            n_estimators     = 200,
            max_depth        = 4,
            learning_rate    = 0.08,
            min_child_weight = 5,
            reg_lambda       = 1e-2,
            tree_method      = "hist",
            verbosity        = 0,
        )
    if model_kind == "lgbm":
        # LightGBM grows leaf-wise (more aggressive than HGB's level-wise),
        # so we cap with num_leaves to match HGB's depth ~4. Histogram
        # bin defaults align with HGB.
        return dict(
            n_estimators       = 200,
            num_leaves         = 15,         # ~ 2^4 - 1
            learning_rate      = 0.08,
            min_data_in_leaf   = 5,
            reg_lambda         = 1e-2,
            verbosity          = -1,
        )
    raise ValueError(
        f"_default_hyperparams_for: unknown model_kind {model_kind!r}. "
        f"Supported: {SUPPORTED_MODEL_KINDS}."
    )


def _build_estimator(spec: SurrogateSpec, target_label: Tuple[str, str]) -> Any:
    """Instantiate one regressor per spec.

    Per-target estimators all share `spec.hyperparams`; per-target tuning
    can be added later by branching on `target_label`.
    """
    if spec.model_kind not in SUPPORTED_MODEL_KINDS:
        raise ValueError(
            f"_build_estimator: unknown model_kind {spec.model_kind!r}. "
            f"Supported: {SUPPORTED_MODEL_KINDS}."
        )

    hparams = {**_default_hyperparams_for(spec.model_kind), **spec.hyperparams}

    if spec.model_kind == "gbm":
        return HistGradientBoostingRegressor(random_state=spec.seed, **hparams)

    if spec.model_kind == "xgb":
        if not _XGB_AVAILABLE:
            raise ImportError(
                "model_kind='xgb' requires xgboost. Install with "
                "`pip install xgboost`."
            )
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=spec.seed, **hparams)

    if spec.model_kind == "lgbm":
        if not _LGBM_AVAILABLE:
            raise ImportError(
                "model_kind='lgbm' requires lightgbm. Install with "
                "`pip install lightgbm`."
            )
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=spec.seed, **hparams)

    # Unreachable -- model_kind already validated above.
    raise AssertionError(spec.model_kind)  # pragma: no cover


class _ConstantRegressor:
    """Predicts a single value for any input; zero Jacobian.

    Used when the training Y for a target has zero variance (e.g. fuels
    with no production in the country: NUCLEAR / GEOTHERM / TIDE for
    Peru). HistGradientBoosting would emit a warning and fit a constant
    anyway -- this avoids both.
    """

    def __init__(self, constant: float) -> None:
        self.constant = float(constant)

    def fit(self, X, y):       # pragma: no cover -- noop
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X)
        n = X.shape[0] if X.ndim > 1 else 1
        return np.full(n, self.constant, dtype=float)


def _build_envelope(
    X:               pd.DataFrame,
    columns:         List[str],
    envelope_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Construct the envelope DataFrame from either explicit bounds or the
    empirical (X.min, X.max). Warns when falling back.

    Envelope built either from explicit LHS bounds (preferred) or from
    the empirical (X.min, X.max) with a warning.
    """
    if envelope_bounds is None:
        warnings.warn(
            "Surrogate: envelope_bounds=None; falling back to the "
            "empirical (X.min, X.max). For real v3 calibration runs "
            "pass envelope_bounds=(lb, ub) so the envelope matches the "
            "LHS sampling box and the SQP optimizer's box constraints.",
            stacklevel=3,
        )
        lb_arr = X.min(axis=0).reindex(columns).values
        ub_arr = X.max(axis=0).reindex(columns).values
    else:
        lb_arr, ub_arr = envelope_bounds
        lb_arr = np.asarray(lb_arr, dtype=float).reshape(-1)
        ub_arr = np.asarray(ub_arr, dtype=float).reshape(-1)
        if lb_arr.shape[0] != len(columns) or ub_arr.shape[0] != len(columns):
            raise ValueError(
                f"envelope_bounds shape mismatch: got "
                f"lb={lb_arr.shape}, ub={ub_arr.shape}, expected length "
                f"{len(columns)} matching X.columns."
            )
        if not np.all(lb_arr <= ub_arr):
            raise ValueError(
                "envelope_bounds: every lb_arr[j] must be <= ub_arr[j]."
            )
    return pd.DataFrame({"min": lb_arr, "max": ub_arr}, index=columns)


def apply_accuracy_gate(
    surrogate: "Surrogate",
    report:    SurrogateReport,
    spec:      SurrogateSpec,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Populate report.accepted_targets and report.rejected_targets by
    applying spec.target_r2_min and spec.target_mape_max thresholds.

    Called explicitly by callers driving the train/dev/test workflow.
    Convention: run on the TEST-set report exactly once at the end of
    hyperparameter tuning.

    Decision rules per target `t`:
      1. `surrogate.is_constant_target(t)` -> auto-accept. Predictions are
         exactly the training mean at every point; nothing to reject.
      2. `report.n_holdout == 0` (in-sample eval): auto-accept. R^2 /
         MAPE on the training data are not a fair generalisation
         estimate and the gate can't defensibly reject on them.
      3. Otherwise: accept iff (R^2 >= target_r2_min) AND (MAPE is NaN
         or MAPE <= target_mape_max). MAPE=NaN happens when true value
         is identically zero for all eval rows -- the R^2 threshold
         alone decides.

    fail_mode dispatch:
      - "warn":  log the rejected list, continue.
      - "raise": ValueError if any target was rejected.

    Mutates `report` in place. Returns (accepted, rejected).
    """
    accepted: List[Tuple[str, str]] = []
    rejected: List[Tuple[str, str]] = []
    reasons:  Dict[Tuple[str, str], str] = {}

    for t in report.targets:
        # Rule 1: constant-output targets auto-accept.
        if surrogate.is_constant_target(t):
            accepted.append(t)
            continue

        # Rule 2: in-sample scoring is not a defensible reject signal.
        if report.n_holdout == 0:
            accepted.append(t)
            continue

        r2_t   = report.r2_per_target.loc[t]
        mape_t = report.mape_per_target.loc[t]
        r2_ok   = (not np.isnan(r2_t))   and (r2_t   >= spec.target_r2_min)
        mape_ok = (np.isnan(mape_t))     or  (mape_t <= spec.target_mape_max)

        if r2_ok and mape_ok:
            accepted.append(t)
        else:
            rejected.append(t)
            reasons[t] = _explain_rejection(
                r2_t, mape_t, spec.target_r2_min, spec.target_mape_max,
            )

    # Mutate the report so callers see the verdict on the same object.
    report.accepted_targets = accepted
    report.rejected_targets = rejected

    if rejected:
        bullets = "\n  - ".join(f"{t}: {reasons[t]}" for t in rejected)
        msg = (
            f"Surrogate accuracy gate rejected {len(rejected)} / "
            f"{len(report.targets)} target(s) (thresholds: "
            f"R^2 >= {spec.target_r2_min}, "
            f"MAPE <= {spec.target_mape_max}%):\n  - {bullets}"
        )
        if spec.fail_mode == "raise":
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)

    return accepted, rejected


def _explain_rejection(
    r2_t:      float,
    mape_t:    float,
    r2_min:    float,
    mape_max:  float,
) -> str:
    """One-line reason describing which threshold(s) a rejected target failed."""
    parts: List[str] = []
    if np.isnan(r2_t):
        parts.append("R^2=NaN (holdout too small or constant Y)")
    elif r2_t < r2_min:
        parts.append(f"R^2={r2_t:.3f} < {r2_min}")
    if (not np.isnan(mape_t)) and (mape_t > mape_max):
        parts.append(f"MAPE={mape_t:.1f}% > {mape_max}%")
    return "; ".join(parts) if parts else "(no reason recorded)"


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 that returns NaN (not -inf) for zero-variance targets."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in percent that returns NaN when y_true has any zeros.

    Energy outputs are non-negative, so zero values dominate the
    denominator and produce inf; better to flag with NaN and let the
    step-4 gate decide what to do.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    nz = y_true != 0.0
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100.0)
