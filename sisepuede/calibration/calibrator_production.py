"""
sisepuede/calibration/calibrator_production.py

Inference-only orchestrator for energy production calibration.

Given a pre-trained surrogate bundle and a raw SISEPUEDE input, runs:
    1. v2 phase 1 + phase 2 consumption (delegated to `Calibrator`, option=3)
    2. Fingerprint check on the post-v2 DataFrame vs the surrogate's stored
       consumption_fingerprint. Refuses to proceed on mismatch unless
       `force_fingerprint_mismatch=True`.
    3. v3 production phase via `solve_phase2_surrogate`.
    4. Applies the SQP solution (multiplies each production-knob column by
       its scale factor).
    5. Optional verification run: projects SISEPUEDE at the final input
       and reports per-target residuals against IEA.

Does NOT train the surrogate. That's what `train_surrogate.py` is for.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sisepuede.calibration.calibration_group      import CalibrationPlan
from sisepuede.calibration.calibrator             import Calibrator
from sisepuede.calibration._surrogate             import (
    Surrogate,
    fingerprint_consumption_state,
)
from sisepuede.calibration.qp_phase2_surrogate    import (
    iea_mask_from_accepted_targets,
    solve_phase2_surrogate,
)


class ProductionCalibrator:
    """v3 orchestrator: v2 phases + surrogate-driven production phase.

    Parameters
    ----------
    models : SISEPUEDEModels or EnergyConsumption
        Model instance passed to the underlying `Calibrator`. For the
        verification run, `SISEPUEDEModels` with `allow_electricity_run=True`
        is required (NemoMod is needed to score production against IEA).
    crosswalk : IEACrosswalk
    df_iea_long : pd.DataFrame
        Raw IEA data for one country. Same shape as v2's Calibrator expects.
    year_target : int
        Calendar year to calibrate to.
    surrogate_bundle : dict
        Output of `_training_pipeline.load_surrogate_bundle`. Must carry:
        `surrogate`, `iea_target_rows`, `ssp_columns`, `A`, `metadata`.
    force_fingerprint_mismatch : bool
        When True, a fingerprint mismatch degrades to a warning instead
        of raising. Use only for debugging.
    """

    def __init__(
        self,
        models,
        crosswalk,
        df_iea_long:                pd.DataFrame,
        year_target:                int,
        surrogate_bundle:           Dict[str, Any],
        force_fingerprint_mismatch: bool = False,
    ) -> None:
        self.models                     = models
        self.crosswalk                  = crosswalk
        self.df_iea_long                = df_iea_long
        self.year_target                = year_target
        self.surrogate_bundle           = surrogate_bundle
        self.force_fingerprint_mismatch = force_fingerprint_mismatch

        # Internal v2 calibrator for phase 1 + consumption phase 2.
        self.v2 = Calibrator(
            models,
            crosswalk,
            df_iea_long,
            year_target,
            include_energy_production=True,
        )

    # ------------------------------------------------------------------
    #    PUBLIC API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        df_in:                pd.DataFrame,
        plan:                 CalibrationPlan,
        v2_option:            int   = 3,
        v2_n_iter:            int   = 2,
        v2_gamma:             float = 100.0,
        # SQP knobs (production side):
        gamma:                float = 1.0,
        trust_radius:         float = 0.05,
        max_iter:             int   = 10,
        tol:                  float = 1e-4,
        verify:               bool  = True,
        verbose:              bool  = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run v2 (phase 1 + consumption phase 2) then v3 production phase.

        Returns
        -------
        df_final : pd.DataFrame
            Input DataFrame after both v2 and v3 have been applied.
        log : dict
            Keys:
              `v2`             : v2's per-iteration log list
              `fingerprint`    : {'current', 'stored', 'match'}
              `sqp`            : diagnostics from `solve_phase2_surrogate`
              `f_star`         : ndarray (n_knobs,)
              `applied_knobs`  : list of column names actually multiplied
              `verify`         : dict of per-target residuals, present iff verify=True
        """
        # ── v2 phases 1 + 2 (consumption) ───────────────────────────────
        if verbose:
            print(f"[ProductionCalibrator] running v2 (option={v2_option}, n_iter={v2_n_iter})")
        df_v2, log_v2 = self.v2.calibrate(
            df_in, plan,
            option=v2_option, n_iter=v2_n_iter, gamma=v2_gamma,
        )

        # ── Fingerprint check ───────────────────────────────────────────
        fp_current = fingerprint_consumption_state(df_v2)
        fp_stored  = self.surrogate_bundle["metadata"]["consumption_fingerprint"]
        fp_match   = (fp_current == fp_stored)
        if not fp_match:
            msg = (
                f"ProductionCalibrator: consumption fingerprint mismatch. "
                f"current post-v2 = {fp_current[:16]}..., "
                f"surrogate trained at = {fp_stored[:16]}.... "
                f"Retrain the surrogate against the current consumption "
                f"state, or pass force_fingerprint_mismatch=True to bypass "
                f"the check (unsafe: surrogate predictions will be biased)."
            )
            if not self.force_fingerprint_mismatch:
                raise ValueError(msg)
            warnings.warn(msg)
        if verbose:
            print(f"  fingerprint match : {fp_match}  "
                  f"(current={fp_current[:12]}, stored={fp_stored[:12]})")

        # ── v3 production SQP ───────────────────────────────────────────
        surrogate       = self.surrogate_bundle["surrogate"]
        A               = self.surrogate_bundle["A"]
        iea_target_rows = self.surrogate_bundle["iea_target_rows"]
        ssp_columns     = self.surrogate_bundle["ssp_columns"]

        b_iea = self._b_iea_for_targets(iea_target_rows)

        # Reconstruct the accepted-IEA mask from the persisted test-set gate.
        accepted_ssp = self.surrogate_bundle["metadata"]["test_report"]["accepted_targets"]
        accepted_mask = iea_mask_from_accepted_targets(
            accepted_ssp, ssp_columns, A,
        )
        if verbose:
            print(f"  IEA targets: {len(iea_target_rows)}, "
                  f"accepted: {int(accepted_mask.sum())}")

        f_star, sqp_diag = solve_phase2_surrogate(
            surrogate         = surrogate,
            A                 = A,
            b_surr            = b_iea,
            accepted_iea_mask = accepted_mask,
            gamma             = gamma,
            trust_radius      = trust_radius,
            max_iter          = max_iter,
            tol               = tol,
            verbose           = verbose,
        )
        if verbose:
            print(f"  SQP status: {sqp_diag['status']} "
                  f"({sqp_diag['iters']} iter, "
                  f"{len(sqp_diag['envelope_violations'])} envelope violations)")

        # ── Apply production knob solution ──────────────────────────────
        df_final, applied = self._apply_production_solution(
            df_v2, surrogate.columns, f_star,
        )
        if verbose:
            print(f"  applied f* to {len(applied)}/{len(surrogate.columns)} "
                  f"production knob columns")

        log: Dict[str, Any] = {
            "v2":             log_v2,
            "fingerprint":    {"current": fp_current, "stored": fp_stored,
                               "match": fp_match},
            "sqp":            sqp_diag,
            "f_star":         np.asarray(f_star),
            "applied_knobs":  applied,
        }

        # ── Verification (optional) ─────────────────────────────────────
        if verify:
            if verbose:
                print("[ProductionCalibrator] running verification projection...")
            log["verify"] = self._verify(df_final, iea_target_rows, A, b_iea)

        return df_final, log

    def log_summary(self, log: Any) -> pd.DataFrame:
        """Passthrough to v2's log_summary for consistency with existing callers.

        Callers can hand this the v2 sub-log directly (e.g. `log['v2']` from
        `calibrate`) or the full v3 log dict (in which case only the v2
        portion is summarised).
        """
        v2_log = log["v2"] if isinstance(log, dict) and "v2" in log else log
        return self.v2.log_summary(v2_log)

    def summarize_knobs(self, *args, **kwargs) -> pd.DataFrame:
        """Passthrough for compatibility with the v2 CLI's output builder."""
        return self.v2.summarize_knobs(*args, **kwargs)

    # ------------------------------------------------------------------
    #    INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _b_iea_for_targets(
        self,
        iea_target_rows: List[Tuple[str, str]],
    ) -> np.ndarray:
        """Look up IEA-side observations (TJ) at year_target, one per target row."""
        # v2's Calibrator already caches IEA values at year_target -- reuse.
        b = np.empty(len(iea_target_rows), dtype=float)
        for i, (bal, prod) in enumerate(iea_target_rows):
            v = self.v2._iea_tj_value(bal, prod)
            b[i] = float("nan") if v is None else float(v)
        return b

    def _apply_production_solution(
        self,
        df:      pd.DataFrame,
        columns: List[str],
        f_star:  np.ndarray,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Multiply each production knob column by its scale factor.

        Columns absent from `df` are silently skipped and reported in the
        returned `applied` list so the caller can log the mismatch.
        """
        df_out = df.copy()
        applied: List[str] = []
        for col, scale in zip(columns, f_star):
            if col in df_out.columns:
                df_out[col] = df_out[col] * float(scale)
                applied.append(col)
        return df_out, applied

    def _verify(
        self,
        df_final:        pd.DataFrame,
        iea_target_rows: List[Tuple[str, str]],
        A:               np.ndarray,
        b_iea:           np.ndarray,
    ) -> Dict[str, Any]:
        """Project SISEPUEDE at df_final and score residuals against IEA.

        Only reports residuals for targets whose SSP fields are actually
        present in the model output. Uses the same crosswalk aggregation
        the training / SQP loop uses.
        """
        # 1. Run the model.
        df_out = self.v2._run_model(df_final)
        time_period = self.v2._get_time_period(df_final)

        # 2. Extract raw per-tech values at the target time period and
        #    aggregate via A.
        row_target = df_out[df_out["time_period"] == time_period]
        if row_target.empty:
            return {"error": f"time_period={time_period} not in model output"}

        # y_ssp order matches surrogate.targets == ssp_columns
        ssp_columns = self.surrogate_bundle["ssp_columns"]
        y_ssp = np.array([
            float(row_target[c].iloc[0]) if c in row_target.columns else 0.0
            for c in ssp_columns
        ])
        y_iea = A @ y_ssp

        # 3. Per-target residuals.
        residuals = []
        for i, (bal, prod) in enumerate(iea_target_rows):
            pred = float(y_iea[i])
            obs  = float(b_iea[i])
            if np.isnan(obs):
                residuals.append({
                    "target":      (bal, prod),
                    "iea_tj":      None,
                    "verify_tj":   pred,
                    "abs_err_tj":  None,
                    "rel_err_pct": None,
                })
            else:
                residuals.append({
                    "target":      (bal, prod),
                    "iea_tj":      obs,
                    "verify_tj":   pred,
                    "abs_err_tj":  pred - obs,
                    "rel_err_pct": (pred - obs) / obs * 100.0 if obs != 0 else None,
                })

        return {
            "time_period":  time_period,
            "residuals":    residuals,
            "n_targets":    len(iea_target_rows),
            "n_reportable": sum(1 for r in residuals if r["rel_err_pct"] is not None),
        }
