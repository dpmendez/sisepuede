"""
sisepuede/calibration/calibrator.py

Calibrator
----------
Two-phase calibration engine that adjusts SISEPUEDE energy inputs to match
IEA World Energy Balance targets at a reference year.

The calibration problem
-----------------------
IEA reports energy by sector AND by fuel. Think of the targets as a matrix:

           TOTAL    COAL    OIL    NATGAS    ELECTR
INDUSTRY   500 TJ   50      100    80        150
TRANSPORT  300 TJ    —      200     —         80
RESIDENT   150 TJ   20       30    50         50

Two distinct sets of input parameters control these two dimensions:

  Phase 1 — Scalar groups (row totals)
      Parameters: consumpinit_*, scalar_inen_energy_demand_*,
                  fuelefficiency_trns_*, avgload_trns_*, occrate_trns_*
      Target:     sector-total TFC  (INDUSTRYxTOTAL, TRANSPORTxTRANSPORT, etc.)
      Method:     Run model once -> compute ratio (iea / model) -> scale inputs

  Phase 2 — Simplex groups (column shares within each row)
      Parameters: frac_inen_energy_*, frac_trns_fuelmix_*, frac_scoe_heat_energy_*
      Target:     fuel share within sector  (INDUSTRYxCOAL, TRANSPORTxOIL, etc.)
      Method:     Run model once -> compute current share vs. IEA share ->
                  scale fracs -> Aitchison renormalization (simplex constraint preserved)

Why this decomposition works
-----------------------------
Scaling consumpinit_* changes how much energy is consumed in total, but leaves
the fuel fractions (frac_* variables) unchanged -> row totals move, column
shares stay fixed.

Adjusting frac_* shifts the fuel mix, but frac_* variables sum to 1 within
each category -> column shares move, the total demand is unchanged.

This independence means the two phases do not fight each other. Iterating
2–3 times handles the small residual coupling that does exist (e.g. fuel
efficiency inputs affect both total demand and fuel mix slightly).

Efficiency
----------
Each phase runs the model ONCE to collect current outputs, computes all
corrections from that single run, then applies them together. This costs
2 model runs per iteration (1 for Phase 1, 1 for Phase 2).

Reference time period
---------------------
Calibration targets the row corresponding to `year_target` in df_input.
Only values at that time period are used to compute correction scalars; the
scalar is then applied uniformly across all time periods (preserving the shape
of any pre-existing trajectory while shifting the level).

Usage
-----
    from sisepuede.calibration.calibrator import Calibrator
    from sisepuede.calibration.build_energy_calibration_plan import (
        build_energy_calibration_plan,
    )

    plan = build_energy_calibration_plan(model_attributes)

    calibrator = Calibrator(
        models       = models,       # SISEPUEDEModels instance
        crosswalk    = xw,           # IEACrosswalk instance
        df_iea_long  = df_iea,       # IEA long frame (output of IEADataLoader.load_country)
        year_target  = 2019,         # calibrate to this historical year
    )

    # See baseline error before calibration
    df_before = calibrator.evaluate(df_input)
    print(xw.summary(df_before))

    # Run calibration
    df_calibrated, log = calibrator.calibrate(df_input, plan, n_iter=2)

    # See residual error after calibration
    df_after = calibrator.evaluate(df_calibrated)
    print(xw.summary(df_after))
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sisepuede.calibration.calibration_group import CalibrationGroup, CalibrationPlan
from sisepuede.calibration.sensitivity import VariableSpec, apply_perturbations


##########################
#    CALIBRATOR          #
##########################

class Calibrator:
    """Two-phase energy calibration engine.

    Parameters
    ----------
    models : SISEPUEDEModels
        Initialised model object exposing a .project(df) method.
    crosswalk : IEACrosswalk
        Initialised crosswalk object.
    df_iea_long : pd.DataFrame
        Raw IEA long frame for one country (output of IEADataLoader.load_country).
        Must contain columns: iea_balance_code, iea_product_code, year, value_iea_tj.
    year_target : int
        Calendar year to calibrate to (must be present in df_iea_long).
    include_energy_production : bool
        Pass True only when calibrating ENTC variables that require the Julia
        EnergyProduction back-end. Default False (consumption-only model).
    """

    def __init__(
        self,
        models,
        crosswalk,
        df_iea_long: pd.DataFrame,
        year_target: int,
        include_energy_production: bool = False,
    ) -> None:

        self.models                    = models
        self.crosswalk                 = crosswalk
        self.df_iea_long               = df_iea_long.copy()
        self.year_target               = year_target
        self.include_energy_production = include_energy_production

        # Pre-extract all IEA values for year_target once.
        # Keys: (iea_balance_code, iea_product_code) -> value in TJ.
        self._iea_tj: Dict[Tuple[str, str], float] = self._cache_iea_values()

    # ------------------------------------------------------------------
    #   IEA value lookup
    # ------------------------------------------------------------------

    def _cache_iea_values(self) -> Dict[Tuple[str, str], float]:
        """Build {(balance, product): value_tj} for year_target from df_iea_long."""
        df = self.df_iea_long[self.df_iea_long["year"] == self.year_target]

        if df.empty:
            available = sorted(self.df_iea_long["year"].unique())
            raise ValueError(
                f"No IEA data for year {self.year_target}. "
                f"Available years: {available}"
            )

        cache: Dict[Tuple[str, str], float] = {}
        for _, row in df.iterrows():
            key = (str(row["iea_balance_code"]), str(row["iea_product_code"]))
            val = row["value_iea_tj"]
            if pd.notna(val):
                cache[key] = float(val)

        return cache

    def _iea_tj_value(self, balance: str, product: str) -> Optional[float]:
        """Return IEA value in TJ for (balance, product), or None if missing."""
        return self._iea_tj.get((balance, product))

    def _iea_to_ssp_units(self, balance: str, product: str) -> Optional[float]:
        """Return the IEA target converted to SSP native units (typically PJ).

        The crosswalk stores unit_conversion_to_tj = factor such that
            ssp_value * unit_conversion_to_tj = value_in_tj
        To go the other way:
            target_ssp = iea_tj / unit_conversion_to_tj
        """
        tj = self._iea_tj_value(balance, product)
        if tj is None:
            return None

        entry = self.crosswalk.get_crosswalk_entry(balance, product)
        if entry is None:
            return None

        conv = float(entry.get("unit_conversion_to_tj", 1.0))
        if conv == 0.0:
            return None

        return tj / conv

    # ------------------------------------------------------------------
    #   Model runner helpers
    # ------------------------------------------------------------------

    def _run_model(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Run the SISEPUEDE model and return the output DataFrame."""
        from sisepuede.manager.sisepuede_models import SISEPUEDEModels

        if isinstance(self.models, SISEPUEDEModels):
            return self.models.project(
                df_in,
                include_electricity_in_energy=self.include_energy_production,
            )
        return self.models.project(df_in)

    def _get_time_period(self, df_in: pd.DataFrame) -> int:
        """Return the time_period integer that corresponds to year_target.

        Raises ValueError if year_target is not found in df_in.
        """
        if "year" not in df_in.columns:
            raise ValueError(
                "df_in must have a 'year' column mapping time_period to "
                "calendar year. Add it before calling Calibrator."
            )

        match = df_in.loc[df_in["year"] == self.year_target, "time_period"]
        if match.empty:
            raise ValueError(
                f"year_target={self.year_target} not found in df_in. "
                f"Available years: {sorted(df_in['year'].unique())}"
            )

        return int(match.iloc[0])

    def _row_at_tp(
        self,
        df_out: pd.DataFrame,
        time_period: int,
        fields: List[str],
    ) -> float:
        """Sum fields in df_out at a specific time_period row.

        Returns 0.0 if the time period is not found or no fields are present.
        """
        available = [f for f in fields if f in df_out.columns]
        if not available:
            return 0.0

        mask = df_out["time_period"] == time_period
        if not mask.any():
            return 0.0

        return float(df_out.loc[mask, available].sum(axis=1).iloc[0])

    @staticmethod
    def _is_frac_group(group: CalibrationGroup) -> bool:
        """Return True if ANY of the group's columns is a fuel-fraction variable.

        Decision: Row 2 of build_energy_calibration_plan contains scalar groups
        whose knobs are frac_* variables (fuel fractions that are simplex-
        constrained). Treating them as freely scalable would violate the
        simplex constraint (fracs must sum to 1 per category). We skip these
        in Phase 1 — their targets are reached indirectly through Phase 2.
        """
        return any(col.startswith("frac_") for col in group.columns)

    # ------------------------------------------------------------------
    #   Phase 1 — calibrate sector totals (scalar groups)
    # ------------------------------------------------------------------

    def _phase1_totals(
        self,
        df_in: pd.DataFrame,
        plan: CalibrationPlan,
        time_period: int,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """Scale consumpinit_*, scalar_*, and efficiency inputs to hit sector totals.

        Decision: run the model ONCE on the current df_in to measure all sector
        outputs simultaneously. Then compute correction scalars for every group
        and apply them in a single pass. This costs 1 model run for the whole
        phase regardless of how many groups there are.

        This is safe because scalar groups for different sectors (INEN, TRNS,
        SCOE) control different input columns and different output fields — they
        do not interact. Applying their scalars in any order (or all at once)
        gives the same result.

        Scalar formula:
            scalar = target_ssp / current_ssp

        where current_ssp is the model output summed over the SSP fields that
        correspond to the IEA target, at time_period, in SSP native units.
        """
        log: List[dict] = []

        # ── 1. One model run to measure current outputs ──────────────────────
        print("    Running model (Phase 1 baseline)...", end=" ", flush=True)
        df_out = self._run_model(df_in)
        print("done")

        available_output_cols = set(df_out.columns)

        # ── 2. Compute and apply scalars for each eligible scalar group ──────
        df = df_in.copy()

        for group in plan.scalar_groups():

            # Skip groups that use frac_* knobs (simplex variables — handled in Phase 2)
            if self._is_frac_group(group):
                log.append({"group": group.name, "phase": 1, "status": "skipped_frac_group"})
                continue

            # Calibrate to the first iea_target (primary target of this group).
            # Secondary targets (e.g. TFCxTOTAL) are informational.
            if not group.iea_targets:
                log.append({"group": group.name, "phase": 1, "status": "skipped_no_targets"})
                continue

            bal, prod = group.iea_targets[0]

            # Get IEA target in SSP units (usually PJ)
            target_ssp = self._iea_to_ssp_units(bal, prod)
            if target_ssp is None:
                warnings.warn(
                    f"Phase 1 — group '{group.name}': no IEA data for "
                    f"({bal}, {prod}) in year {self.year_target}. Skipping."
                )
                log.append({"group": group.name, "phase": 1, "status": "skipped_no_iea",
                            "target": (bal, prod)})
                continue

            # Get SSP output fields from crosswalk, filtered to available columns
            ssp_fields = [
                f for f in self.crosswalk.get_ssp_fields_for_target(bal, prod)
                if f in available_output_cols
            ]
            if not ssp_fields:
                warnings.warn(
                    f"Phase 1 — group '{group.name}': no SSP output fields found "
                    f"for ({bal}, {prod}). Skipping."
                )
                log.append({"group": group.name, "phase": 1, "status": "skipped_no_ssp_fields",
                            "target": (bal, prod)})
                continue

            # Measure current model output at the target time period
            current_ssp = self._row_at_tp(df_out, time_period, ssp_fields)

            if current_ssp == 0.0:
                warnings.warn(
                    f"Phase 1 — group '{group.name}': current model output is 0 "
                    f"for ({bal}, {prod}). Cannot compute scalar. Skipping."
                )
                log.append({"group": group.name, "phase": 1, "status": "skipped_zero_current",
                            "target": (bal, prod)})
                continue

            scalar = target_ssp / current_ssp

            # Filter group columns to those that exist in df_in
            input_cols = [c for c in group.columns if c in df.columns]
            if not input_cols:
                log.append({"group": group.name, "phase": 1, "status": "skipped_no_input_cols"})
                continue

            # Apply scalar uniformly across all time periods.
            # This preserves the shape of any pre-existing trajectory while
            # shifting the level to match the IEA target year.
            df[input_cols] = df[input_cols] * scalar

            log.append({
                "group":       group.name,
                "phase":       1,
                "target":      (bal, prod),
                "current_ssp": round(current_ssp, 4),
                "target_ssp":  round(target_ssp,  4),
                "scalar":      round(scalar, 4),
                "status":      "ok",
            })

        return df, log

    # ------------------------------------------------------------------
    #   Phase 2 — calibrate fuel mix (simplex groups)
    # ------------------------------------------------------------------

    def _phase2_fuel_mix(
        self,
        df_in: pd.DataFrame,
        plan: CalibrationPlan,
        time_period: int,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """Adjust frac_* to match per-fuel IEA targets within each sector.

        Decision: run the model ONCE to measure all current fuel shares. Then
        compute scale factors for all simplex groups and apply them together via
        Aitchison renormalization (apply_perturbations), which handles the
        simplex constraint correctly: fracs continue to sum to 1.

        Scale factor formula:
            target_share  = iea_fuel_tj / iea_sector_total_tj
            current_share = model_fuel_pj / model_sector_total_pj
            scale         = target_share / current_share

        The scale is applied to frac_* columns (not to absolute energy values),
        which is correct: scaling a fraction by target/current shifts the share
        of that fuel up or down, and the renormalization step keeps everything
        summing to 1.

        Sector total lookup convention:
            For a simplex group targeting (INDUSTRY, COAL), the sector total
            is (INDUSTRY, INDUSTRY) — i.e., the product code equals the balance
            code. This is the convention used in build_energy_calibration_plan.
            Rows 6-8 target (RESIDENTxCOAL), (COMMPUBxOIL), etc., with totals
            at (RESIDENTxRESIDENT), (COMMPUBxCOMMPUB).
        """
        log: List[dict] = []

        # ── 1. One model run to measure current fuel-share outputs ───────────
        print("    Running model (Phase 2 baseline)...", end=" ", flush=True)
        df_out = self._run_model(df_in)
        print("done")

        # ── 2. Compute scale factor for every simplex group ──────────────────
        # Accumulate into a {column: scale} dict and a flat specs list.
        # apply_perturbations will handle the Aitchison renormalization.
        variable_scales: Dict[str, float] = {}
        specs_all:       List[VariableSpec] = []

        for group in plan.simplex_groups():
            if not group.iea_targets:
                continue

            bal, prod = group.iea_targets[0]

            # ── IEA values ───────────────────────────────────────────────────
            iea_fuel_tj  = self._iea_tj_value(bal, prod)
            # Sector total: by convention (INDUSTRY, INDUSTRY), (RESIDENT, RESIDENT), etc.
            iea_total_tj = self._iea_tj_value(bal, bal)

            if iea_fuel_tj is None:
                log.append({"group": group.name, "phase": 2, "status": "skipped_no_iea_fuel",
                            "target": (bal, prod)})
                continue

            if iea_total_tj is None or iea_total_tj == 0.0:
                log.append({"group": group.name, "phase": 2, "status": "skipped_no_iea_total",
                            "target": (bal, prod)})
                continue

            target_share = iea_fuel_tj / iea_total_tj

            # ── Current model shares ─────────────────────────────────────────
            ssp_fuel_fields  = self.crosswalk.get_ssp_fields_for_target(bal, prod)
            ssp_total_fields = self.crosswalk.get_ssp_fields_for_target(bal, bal)

            if not ssp_fuel_fields or not ssp_total_fields:
                log.append({"group": group.name, "phase": 2,
                            "status": "skipped_no_crosswalk", "target": (bal, prod)})
                continue

            current_fuel_ssp  = self._row_at_tp(df_out, time_period, ssp_fuel_fields)
            current_total_ssp = self._row_at_tp(df_out, time_period, ssp_total_fields)

            if current_total_ssp == 0.0:
                log.append({"group": group.name, "phase": 2,
                            "status": "skipped_zero_total", "target": (bal, prod)})
                continue

            current_share = current_fuel_ssp / current_total_ssp

            if current_share < 1e-9:
                # Fuel is essentially absent in the model but IEA says it should
                # have a non-zero share. Scaling from zero is undefined — skip.
                warnings.warn(
                    f"Phase 2 — group '{group.name}': current share ~ 0 for "
                    f"({bal}, {prod}) but IEA target share = {target_share:.4f}. "
                    "Cannot scale from zero. Skipping."
                )
                log.append({"group": group.name, "phase": 2,
                            "status": "skipped_zero_current_share", "target": (bal, prod),
                            "target_share": round(target_share, 4)})
                continue

            scale = target_share / current_share

            # Record scale for all columns in this group.
            # When a column appears in multiple groups (e.g. same frac appears
            # in Row 2 and Row 6), the last scale wins. In practice this should
            # not occur because simplex groups partition the frac columns by
            # (sector x fuel). A warning is emitted if it does.
            for spec in group.specs:
                if spec.column in variable_scales:
                    warnings.warn(
                        f"Column '{spec.column}' appears in multiple simplex groups. "
                        f"Previous scale {variable_scales[spec.column]:.4f} overwritten "
                        f"by {scale:.4f} from group '{group.name}'."
                    )
                variable_scales[spec.column] = scale
                specs_all.append(spec)

            log.append({
                "group":          group.name,
                "phase":          2,
                "target":         (bal, prod),
                "iea_fuel_tj":    round(iea_fuel_tj,    2),
                "iea_total_tj":   round(iea_total_tj,   2),
                "target_share":   round(target_share,   4),
                "current_share":  round(current_share,  4),
                "scale":          round(scale,           4),
                "status":         "ok",
            })

        if not variable_scales:
            # Nothing to adjust — return input unchanged
            return df_in.copy(), log

        # De-duplicate specs by column (first occurrence keeps its simplex metadata)
        seen: set = set()
        specs_dedup: List[VariableSpec] = []
        for s in specs_all:
            if s.column not in seen:
                seen.add(s.column)
                specs_dedup.append(s)

        # ── 3. Apply all simplex perturbations simultaneously ────────────────
        # apply_perturbations applies Aitchison renormalization per simplex
        # constraint (all frac_* columns sharing the same category prefix are
        # renormalized together so they continue to sum to 1). This is
        # order-independent and gives a clean, constraint-preserving result.
        df_out_calibrated = apply_perturbations(df_in, specs_dedup, variable_scales)

        return df_out_calibrated, log

    # ------------------------------------------------------------------
    #   Public calibration API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        df_in: pd.DataFrame,
        plan: CalibrationPlan,
        n_iter: int = 2,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """Run n_iter rounds of Phase 1 (sector totals) + Phase 2 (fuel mix).

        Parameters
        ----------
        df_in : pd.DataFrame
            Baseline SISEPUEDE input DataFrame for one country.
            Must have a 'year' column mapping time_period to calendar year.
        plan : CalibrationPlan
            Output of build_energy_calibration_plan(model_attributes).
        n_iter : int
            Number of full Phase 1 + Phase 2 iterations. Default 2.
            - 1 iteration is usually sufficient for sector totals (Phase 1).
            - 2 iterations closes residual fuel-mix / total coupling.
            - 3+ iterations offer diminishing returns.

        Returns
        -------
        df_calibrated : pd.DataFrame
            Input DataFrame with adjusted parameter values.
        log : List[dict]
            Per-iteration, per-group record of scale factors and status codes.
            Status codes:
              "ok"                       — applied successfully
              "skipped_frac_group"       — Phase 1 skipped Row 2 groups
              "skipped_no_iea"           — IEA value missing for this target/year
              "skipped_no_ssp_fields"    — crosswalk has no mapping for this pair
              "skipped_zero_current"     — model output was 0 (can't compute ratio)
              "skipped_zero_current_share" — fuel absent in model, can't scale from 0
              "skipped_no_iea_total"     — sector total missing from IEA data
        """
        time_period = self._get_time_period(df_in)
        df          = df_in.copy()
        full_log:   List[dict] = []

        for it in range(n_iter):
            print(f"\n=== Calibration iteration {it + 1}/{n_iter} ===")

            print("  Phase 1 — sector totals (consumpinit_*, scalar_*, efficiencies):")
            df, log1 = self._phase1_totals(df, plan, time_period)
            ok1    = sum(1 for r in log1 if r.get("status") == "ok")
            skip1  = sum(1 for r in log1 if r.get("status", "").startswith("skipped"))
            print(f"    applied={ok1}  skipped={skip1}")

            print("  Phase 2 — fuel mix (frac_* simplex groups):")
            df, log2 = self._phase2_fuel_mix(df, plan, time_period)
            ok2    = sum(1 for r in log2 if r.get("status") == "ok")
            skip2  = sum(1 for r in log2 if r.get("status", "").startswith("skipped"))
            print(f"    applied={ok2}  skipped={skip2}")

            full_log.append({
                "iteration": it + 1,
                "phase1":    log1,
                "phase2":    log2,
            })

        return df, full_log

    # ------------------------------------------------------------------
    #   Diagnostics
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df_in: pd.DataFrame,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run the model on df_in and return the IEA comparison table.

        Use this to measure calibration error before and after calling
        calibrate(). The returned DataFrame is the output of
        IEACrosswalk.build_comparison() and can be passed directly to
        IEACrosswalk.summary() for a compact view.

        Parameters
        ----------
        df_in : pd.DataFrame
            Input DataFrame (baseline or calibrated). Must have a 'year' column.
        year_min, year_max : int | None
            Restrict comparison to this year range. Defaults to year_target
            on both ends if not specified (single-year comparison).

        Returns
        -------
        pd.DataFrame
            Output of IEACrosswalk.build_comparison(). Key columns:
            iea_balance_code, iea_product_code, year,
            value_iea_tj, value_sisepuede_tj,
            ratio_sisepuede_over_iea, rel_err_iea.
        """
        year_min = year_min or self.year_target
        year_max = year_max or self.year_target

        df_out  = self._run_model(df_in)

        # Attach year column (IEACrosswalk.aggregate_sisepuede needs it)
        if "year" not in df_out.columns and "year" in df_in.columns:
            year_map = df_in[["time_period", "year"]].drop_duplicates()
            df_out   = df_out.merge(year_map, on="time_period", how="left")

        df_ssp  = self.crosswalk.aggregate_sisepuede(df_out)
        return self.crosswalk.build_comparison(
            df_ssp,
            self.df_iea_long,
            year_min=year_min,
            year_max=year_max,
        )

    def log_summary(self, log: List[dict]) -> pd.DataFrame:
        """Convert the calibration log to a flat DataFrame for inspection.

        Parameters
        ----------
        log : List[dict]
            The second return value of calibrate().

        Returns
        -------
        pd.DataFrame
            One row per (iteration, group). Columns include iteration, phase,
            group name, status, scalar/scale applied, and target IEA pair.
        """
        rows = []
        for entry in log:
            it = entry["iteration"]
            for phase_key, phase_log in [("phase1", entry["phase1"]),
                                          ("phase2", entry["phase2"])]:
                for rec in phase_log:
                    rows.append({
                        "iteration": it,
                        "phase":     rec.get("phase", phase_key),
                        "group":     rec.get("group", ""),
                        "status":    rec.get("status", ""),
                        "target":    str(rec.get("target", "")),
                        "scalar":    rec.get("scalar",  rec.get("scale", float("nan"))),
                        "current":   rec.get("current_ssp", rec.get("current_share", float("nan"))),
                        "target_val":rec.get("target_ssp",  rec.get("target_share",  float("nan"))),
                    })

        return pd.DataFrame(rows)
