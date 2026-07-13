"""
sisepuede/calibration/_data_generation.py

Orchestrator for producing v3 surrogate training data.

The subroutine `generate_lhs_training_data()` here is the pure-Python entry
point used by both the `generate_surrogate_data.py` CLI and (in principle)
any notebook or batch pipeline that wants a training-data pickle without
going through the CLI. All SISEPUEDE dependencies (ModelAttributes,
IEACrosswalk, SISEPUEDEModels, IEA raw DataFrame) are injectable so the
subroutine is testable and doesn't force a particular file layout on the
caller.

Output layout (when `output_dir` is provided):

    {output_dir}/{iso3}_{year}_n{N}_seed{S}/
        result.pkl      # SensitivityResult (scale factors + all SSP outputs)
        baseline.pkl    # df_input_energy -- the calibrated post-AFOLU/IPPU frame
        metadata.json   # provenance (country/year, knob config, fingerprint,
                        # wall-clock, sisepuede/git info)

The baseline pickle is what closes the reproducibility loop: given
(result.pkl, baseline.pkl) alone, any downstream caller can reconstruct the
absolute SSP input values that produced any single run. The metadata JSON
is what CalibratorV3 reads at inference time to verify (via
consumption_fingerprint) that the training-time consumption state matches
what we're currently calibrating against.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sisepuede.calibration._surrogate import fingerprint_consumption_state
from sisepuede.calibration.build_energy_calibration_plan import (
    build_energy_calibration_plan,
)
from sisepuede.calibration.calibration_group import CalibrationGroup
from sisepuede.calibration.sensitivity import (
    SensitivityResult,
    SensitivityRunner,
    VariableSpec,
)


# Default production-side knob families endorsed by the SSP author
# Data-generation callers can override this, but the default is what
# v3.0 ships with.
DEFAULT_KNOB_PREFIX_FILTERS: List[str] = [
    "nemomod_entc_scalar_availability_factor_",
    "efficfactor_entc_technology_fuel_use_",
]

DEFAULT_KNOB_BOUNDS: Tuple[float, float] = (0.9, 1.1)


def generate_lhs_training_data(
    df_input_energy:     pd.DataFrame,
    iso_country:         str,
    target_year:         int,
    n_lhs:               int,
    *,
    seed:                int = 42,
    knob_prefix_filters: Optional[List[str]] = None,
    knob_bounds:         Tuple[float, float] = DEFAULT_KNOB_BOUNDS,
    year_min:            int = 2015,
    year_max:            int = 2022,
    # SISEPUEDE dependencies (all injectable):
    model_attributes:    Any,
    iea_crosswalk:       Any,
    df_iea_raw:          pd.DataFrame,
    models:              Any,
    # Output:
    output_dir:          Optional[str] = None,
    tag:                 str = "",
    verbose:             bool = True,
) -> Tuple[SensitivityResult, Dict[str, Any]]:
    """Run an LHS sweep on SISEPUEDE and return (SensitivityResult, metadata).

    The `df_input_energy` frame is expected to be **post-consumption
    calibration** (i.e. after v2 has been applied) plus post-AFOLU/IPPU
    merge -- what `SensitivityRunner` expects as its baseline. Callers
    who pass a pre-v2 raw input will get a working training set but the
    surrogate they train will be pinned to that raw consumption state
    (which won't match a post-v2 calibrator run downstream, tripping the
    fingerprint check in CalibratorV3).

    Parameters
    ----------
    df_input_energy : pd.DataFrame
        Calibrated post-AFOLU/IPPU SISEPUEDE input. Must include a 'year'
        column mapping time_period -> calendar year.
    iso_country : str
        ISO-3 country code. Passed to SensitivityRunner and used in
        output file naming.
    target_year : int
        Calendar year the surrogate is trained to predict at. Stored in
        metadata; not used by the LHS itself.
    n_lhs : int
        Number of LHS samples (= number of SISEPUEDE runs).
    seed : int
        LHS random seed. Reproducibility is `(inputs, spec_list, seed)`.
    knob_prefix_filters : List[str] | None
        Prefixes selecting which `electout__*` group specs get perturbed.
        None uses `DEFAULT_KNOB_PREFIX_FILTERS`.
    knob_bounds : (float, float)
        `(lb, ub)` scale-factor box applied to every selected spec.
        Default `(0.9, 1.1)`.
    year_min, year_max : int
        Time-window bounds passed to SensitivityRunner for the IEA
        comparison table.
    model_attributes : ModelAttributes
        SISEPUEDE ModelAttributes object.
    iea_crosswalk : IEACrosswalk
        Initialised crosswalk.
    df_iea_raw : pd.DataFrame
        Raw IEA data for the country.
    models : SISEPUEDEModels
        Initialised with allow_electricity_run=True so NemoMod runs.
    output_dir : str | None
        When provided, persists artifacts to
        `{output_dir}/{iso3}_{year}_n{N}_seed{S}{tag_suffix}/`.
    tag : str
        Optional suffix appended to the output subdirectory name (for
        e.g. multiple training runs at the same (country, year, N, seed)
        with different specs).
    verbose : bool

    Returns
    -------
    result : SensitivityResult
    metadata : dict
    """
    t0 = time.time()

    if knob_prefix_filters is None:
        knob_prefix_filters = list(DEFAULT_KNOB_PREFIX_FILTERS)

    # ── Build the plan, isolate ENTC groups, apply bounds, flatten to specs
    plan = build_energy_calibration_plan(model_attributes)
    prod_groups: List[CalibrationGroup] = [
        g for g in plan.groups if g.name.startswith("electout__")
    ]
    if not prod_groups:
        raise RuntimeError(
            "generate_lhs_training_data: no `electout__*` groups found in "
            "calibration plan. Check build_energy_calibration_plan."
        )

    lb, ub = knob_bounds
    for g in prod_groups:
        g.set_bounds(lb, ub)

    specs: List[VariableSpec] = []
    for g in prod_groups:
        for s in g.specs:
            if any(s.column.startswith(p) for p in knob_prefix_filters):
                specs.append(s)

    if not specs:
        raise RuntimeError(
            f"generate_lhs_training_data: no specs matched any prefix in "
            f"{knob_prefix_filters}. Check the filter or the plan."
        )

    if verbose:
        print(f"[generate_lhs_training_data] {iso_country} / {target_year}")
        print(f"  knob prefix filters : {knob_prefix_filters}")
        print(f"  knob bounds         : [{lb}, {ub}]")
        print(f"  active specs        : {len(specs)}")
        print(f"  n_lhs               : {n_lhs}  (seed={seed})")

    # ── Run the sweep
    runner = SensitivityRunner(
        models                    = models,
        df_baseline               = df_input_energy,
        iea_crosswalk             = iea_crosswalk,
        df_iea_raw                = df_iea_raw,
        iso                       = iso_country,
        include_energy_production = True,
        year_min                  = year_min,
        year_max                  = year_max,
    )
    result = runner.run_lhs(specs, n_samples=n_lhs, seed=seed)

    wall_clock = time.time() - t0

    # ── Metadata
    metadata: Dict[str, Any] = {
        "iso_country":              iso_country,
        "target_year":              target_year,
        "n_lhs":                    int(n_lhs),
        "seed":                     int(seed),
        "knob_prefix_filters":      list(knob_prefix_filters),
        "knob_bounds":              [float(lb), float(ub)],
        "n_specs":                  len(specs),
        "spec_columns":             [s.column for s in specs],
        "year_min":                 int(year_min),
        "year_max":                 int(year_max),
        "consumption_fingerprint":  fingerprint_consumption_state(df_input_energy),
        "wall_clock_seconds":       float(wall_clock),
        "generated_at":             datetime.datetime.now(datetime.timezone.utc)
                                        .isoformat(timespec="seconds"),
        "python_version":           sys.version.split()[0],
        "git_commit":               _git_commit_hash(),
        "tag":                      tag,
    }
    if verbose:
        print(f"  wall clock          : {wall_clock:.1f} s "
              f"({wall_clock/n_lhs:.2f} s/run)")
        print(f"  consumption fp      : "
              f"{metadata['consumption_fingerprint'][:16]}...")

    # ── Persist
    if output_dir is not None:
        subdir = _build_output_subdir(
            output_dir, iso_country, target_year, n_lhs, seed, tag,
        )
        _persist_artifacts(subdir, result, df_input_energy, metadata, verbose)
        metadata["output_dir"] = subdir

    return result, metadata


# ─────────────────────────────────────────────────────────────────────────
#   INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────

def _build_output_subdir(
    output_dir:  str,
    iso_country: str,
    target_year: int,
    n_lhs:       int,
    seed:        int,
    tag:         str,
) -> str:
    """Canonical subdirectory name inside `output_dir`."""
    stem = f"{iso_country}_{target_year}_n{n_lhs}_seed{seed}"
    if tag:
        stem = f"{stem}_{tag}"
    subdir = os.path.join(output_dir, stem)
    os.makedirs(subdir, exist_ok=True)
    return subdir


def _persist_artifacts(
    subdir:          str,
    result:          SensitivityResult,
    df_input_energy: pd.DataFrame,
    metadata:        Dict[str, Any],
    verbose:         bool,
) -> None:
    """Write result.pkl, baseline.pkl, metadata.json into `subdir`."""
    result_path   = os.path.join(subdir, "result.pkl")
    baseline_path = os.path.join(subdir, "baseline.pkl")
    metadata_path = os.path.join(subdir, "metadata.json")

    pd.to_pickle(result,          result_path)
    pd.to_pickle(df_input_energy, baseline_path)
    with open(metadata_path, "w") as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True)

    if verbose:
        rs = _file_size_mb(result_path)
        bs = _file_size_mb(baseline_path)
        print(f"  saved: {subdir}")
        print(f"    result.pkl   ({rs:.1f} MB)")
        print(f"    baseline.pkl ({bs:.1f} MB)")
        print(f"    metadata.json")


def _file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024.0 * 1024.0)


def _git_commit_hash() -> Optional[str]:
    """Best-effort git HEAD hash for provenance. None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8").strip()
    except Exception:                        # pragma: no cover
        return None


# ─────────────────────────────────────────────────────────────────────────
#   LOAD HELPERS (for consumers of the persisted artifacts)
# ─────────────────────────────────────────────────────────────────────────

def load_training_data(
    subdir: str,
) -> Tuple[SensitivityResult, pd.DataFrame, Dict[str, Any]]:
    """Inverse of the persistence in `generate_lhs_training_data`.

    Returns (result, df_baseline, metadata) loaded from
    result.pkl / baseline.pkl / metadata.json under `subdir`.
    """
    result   = pd.read_pickle(os.path.join(subdir, "result.pkl"))
    baseline = pd.read_pickle(os.path.join(subdir, "baseline.pkl"))
    with open(os.path.join(subdir, "metadata.json")) as fp:
        metadata = json.load(fp)
    return result, baseline, metadata
