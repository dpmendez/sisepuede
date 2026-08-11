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
                        # wall-clock, sisepuede/git info, LHS design_path/hash)

An optional `design_path` argument controls where the unit-cube LHS
design ([0,1]^D, country- and year-agnostic) is stored. When given,
the design is loaded from that path if it exists, else sampled fresh
and saved there. This lets a single design drive multiple bundles
(different countries, years, or knob bounds) with byte-identical
sample points.
`metadata.json` records the `design_path`, `design_source`, and `design_hash`
so downstream tools can prove two bundles share a design.

The baseline pickle is what closes the reproducibility loop: given
(result.pkl, baseline.pkl) alone, any downstream caller can reconstruct the
absolute SSP input values that produced any single run. The metadata JSON
is what ProductionCalibrator reads at inference time to verify (via
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
    sample_lhs_unit_cube,
)

try:                                                     # scipy is already a runtime dep via sensitivity.py
    from scipy.stats.qmc import scale as lhs_scale
except ImportError:                                      # pragma: no cover
    lhs_scale = None                                     # sensitivity.py will error first


# Default production-side knob families endorsed by the SSP author
# Data-generation callers can override this, but the default is what
# v3.0 ships with.
DEFAULT_KNOB_PREFIX_FILTERS: List[str] = [
    "nemomod_entc_scalar_availability_factor_",
    "efficfactor_entc_technology_fuel_use_",
]

DEFAULT_KNOB_BOUNDS: Tuple[float, float] = (0.8, 1.2)


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
    min_ok_frac:         float = 0.95,
    # Shared LHS-design plumbing. When `design_path` is given:
    #   - if the file exists, the unit-cube [0,1]^D design is loaded from
    #     it and rescaled to `knob_bounds`; sampling is skipped.
    #   - otherwise a fresh design is sampled from `(seed, D, n_lhs)` and
    #     written to that path so subsequent runs can reuse the same sample points.
    # When `design_path` is None the design is sampled in-process and
    # not persisted separately (preserves the pre-refactor behaviour).
    design_path:         Optional[str] = None,
    # Batching. n_batches=1 (default) runs the full N in-process,
    # n_batches > 1 with batch_index in [0, K) runs only the contiguous 
    # slice [floor(i*N/K), floor((i+1)*N/K)) of run_index values,
    # writes per-run pickle shards under {bundle}/shards/, and skips
    # result.pkl persistence -- the merge step (see `merge_bundle_shards`)
    # assembles the final artifacts from shards after all batches complete.
    n_batches:           int = 1,
    batch_index:         Optional[int] = None,
) -> Tuple[SensitivityResult, Dict[str, Any]]:
    """Run an LHS sweep on SISEPUEDE and return (SensitivityResult, metadata).

    The `df_input_energy` frame is expected to be **post-consumption
    calibration** (i.e. after v2 has been applied) plus post-AFOLU/IPPU
    merge -- what `SensitivityRunner` expects as its baseline. Callers
    who pass a pre-v2 raw input will get a working training set but the
    surrogate they train will be pinned to that raw consumption state
    (which won't match a post-v2 calibrator run downstream, tripping the
    fingerprint check in ProductionCalibrator).

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

    # ── Obtain the LHS design (load-or-sample-then-rescale)
    #
    # The unit-cube design is [0,1]^D and depends only on (seed, D, N)
    # and the ordered spec columns. Storing it separately from the run
    # outputs lets a single file drive multiple countries/years/bounds
    knob_columns = [s.column for s in specs]
    design_source = _load_or_sample_unit_design(
        design_path  = design_path,
        knob_columns = knob_columns,
        n_lhs        = n_lhs,
        seed         = seed,
        verbose      = verbose,
    )
    unit = design_source["unit"]

    # Rescale [0,1]^D -> [lb, ub]^D using scipy's canonical helper (the
    # same call sample_lhs uses internally, so numeric behaviour is
    # byte-identical to the pre-refactor path).
    lb_arr = np.array([s.lb for s in specs])
    ub_arr = np.array([s.ub for s in specs])
    samples_df = pd.DataFrame(
        lhs_scale(unit.values, lb_arr, ub_arr),
        columns=knob_columns,
    ).reset_index(drop=True)

    # ── Batching: slice samples_df to the runs this invocation owns.
    # For the default n_batches=1, `batch_slice` is the whole design
    # For n_batches > 1 we slice contiguously by run_index; the merge step
    # re-assembles.
    batch_slice = _batch_run_indices(n_lhs, n_batches, batch_index)
    is_batched  = n_batches > 1 or batch_index is not None
    if is_batched:
        samples_df = samples_df.iloc[list(batch_slice)]
        if verbose:
            print(f"  batch               : "
                  f"index={batch_index} / {n_batches}  "
                  f"runs=[{batch_slice.start}, {batch_slice.stop})  "
                  f"n={len(samples_df)}")

    # Shards directory lives inside the bundle. We need `output_dir`
    # for batched mode so shards from parallel workers land in the
    # same place -- refuse the request rather than silently discard
    # per-run data to an in-memory-only run.
    shards_dir: Optional[str] = None
    if is_batched:
        if output_dir is None:
            raise ValueError(
                "generate_lhs_training_data: batched execution "
                "(n_batches > 1 or batch_index != None) requires "
                "output_dir so per-run shards can be written to "
                "{output_dir}/{iso3}_{year}_n{N}_seed{S}/shards/."
            )
        bundle_dir = _build_output_subdir(
            output_dir, iso_country, target_year, n_lhs, seed, tag,
        )
        shards_dir = os.path.join(bundle_dir, "shards")
        os.makedirs(shards_dir, exist_ok=True)

        # Write baseline.pkl + a stub metadata.json now so the merge
        # step (which may run later, on a different machine) has
        # everything it needs. "Stub" = same shape as the final
        # metadata but without batch-crosscutting counts; the merge
        # rewrites it with final totals.
        _write_bundle_stubs(
            bundle_dir      = bundle_dir,
            df_input_energy = df_input_energy,
            iso_country     = iso_country,
            target_year     = target_year,
            n_lhs           = n_lhs,
            seed            = seed,
            lb              = lb,
            ub              = ub,
            knob_prefix_filters = knob_prefix_filters,
            spec_columns    = knob_columns,
            year_min        = year_min,
            year_max        = year_max,
            n_batches       = n_batches,
            design_source   = design_source,
            tag             = tag,
            verbose         = verbose,
        )

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
    result = runner.run_samples(samples_df, specs, shards_dir=shards_dir)

    wall_clock = time.time() - t0

    # In batched mode this invocation only produces shards. The quality
    # gate, result.pkl persistence, and full-metadata write happen at
    # merge time (`merge_bundle_shards`), when all shards are on disk.
    if is_batched:
        batch_metadata = {
            "iso_country":  iso_country,
            "target_year":  target_year,
            "batch_index":  int(batch_index),
            "n_batches":    int(n_batches),
            "run_indices":  [int(i) for i in batch_slice],
            "n_ok":         int((result.run_status["status"] == "ok").sum())
                                if result.run_status is not None else 0,
            "wall_clock_seconds": float(wall_clock),
            "generated_at": datetime.datetime.now(datetime.timezone.utc)
                                .isoformat(timespec="seconds"),
            "git_commit":   _git_commit_hash(),
            "shards_dir":   shards_dir,
        }
        if verbose:
            print(f"  batch done          : wall={wall_clock:.1f}s  "
                  f"ok={batch_metadata['n_ok']}/{len(batch_slice)}")
            print(f"  shards written to   : {shards_dir}")
            print("  (merge with: python -m sisepuede.calibration."
                  "generate_surrogate_data --merge-only ...)")
        return result, batch_metadata

    # ── Quality gate: check per-sample success rate before persisting
    status = result.run_status
    n_total  = len(status) if status is not None else 0
    n_ok     = int((status["status"] == "ok").sum()) if n_total else 0
    n_fail   = int((status["status"] == "failed").sum()) if n_total else 0
    n_missing= int((status["status"] == "missing_electricity").sum()) if n_total else 0
    ok_frac  = (n_ok / n_total) if n_total else 0.0
    if verbose:
        print(f"  runs: {n_ok}/{n_total} ok, {n_fail} failed, "
              f"{n_missing} missing electricity  (min_ok_frac={min_ok_frac:.2f})")
    if n_total and ok_frac < min_ok_frac:
        # Persist run_status so the caller can inspect what failed even
        # after we abort, but skip pickling result.pkl -- the frame is
        # partially garbage and would silently poison surrogate training.
        if output_dir is not None:
            subdir = _build_output_subdir(
                output_dir, iso_country, target_year, n_lhs, seed, tag,
            )
            status_path = os.path.join(subdir, "run_status.csv")
            status.to_csv(status_path, index=False)
            print(f"  wrote run_status.csv (failures only) -> {status_path}")
        raise RuntimeError(
            f"Data generation aborted: only {n_ok}/{n_total} samples "
            f"succeeded ({ok_frac:.1%} < min_ok_frac={min_ok_frac:.0%}). "
            f"{n_fail} raised, {n_missing} returned no electricity output. "
            f"Common causes: parallel data-gen runs sharing "
            f"fp_nemomod_temp_sqlite_db (SQLite lock), Julia/NemoMod init "
            f"failure. Inspect run_status.csv for the per-sample breakdown."
        )

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
        # LHS-design provenance. Two bundles that share `design_hash`
        # were driven by the exact same unit-cube sample points, even
        # if they were run on different countries, years, or bound
        # boxes -- useful for global-ensemble diagnostics.
        "design_path":              design_source["path"],
        "design_source":            design_source["source"],  # "loaded" | "sampled" | "in_memory"
        "design_hash":              design_source["hash"],
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
    """Write result.pkl, baseline.pkl, metadata.json, run_status.csv into `subdir`."""
    result_path   = os.path.join(subdir, "result.pkl")
    baseline_path = os.path.join(subdir, "baseline.pkl")
    metadata_path = os.path.join(subdir, "metadata.json")
    status_path   = os.path.join(subdir, "run_status.csv")

    pd.to_pickle(result,          result_path)
    pd.to_pickle(df_input_energy, baseline_path)
    with open(metadata_path, "w") as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True)
    if result.run_status is not None:
        result.run_status.to_csv(status_path, index=False)

    if verbose:
        rs = _file_size_mb(result_path)
        bs = _file_size_mb(baseline_path)
        print(f"  saved: {subdir}")
        print(f"    result.pkl   ({rs:.1f} MB)")
        print(f"    baseline.pkl ({bs:.1f} MB)")
        print(f"    metadata.json")
        if result.run_status is not None:
            print(f"    run_status.csv ({len(result.run_status)} rows)")


def _file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024.0 * 1024.0)


def _batch_run_indices(
    n_total:     int,
    n_batches:   int,
    batch_index: Optional[int],
) -> range:
    """Return the contiguous slice of run_index this batch owns.

    n_batches=1 (with batch_index None or 0) yields range(0, n_total),
    which makes the single-batch code path byte-identical to today.
    Otherwise batch_index in [0, n_batches) selects
    range(floor(i*N/K), floor((i+1)*N/K)).

    Chosen contiguous over round-robin because a bundle owner reasons
    about "runs 500-999 are batch 2" more readily than about a stride.
    Since we concatenate every shard at merge time, the training set
    is identical either way.
    """
    if n_batches < 1:
        raise ValueError(f"_batch_run_indices: n_batches must be >= 1, got {n_batches}")
    if n_batches == 1 and batch_index in (None, 0):
        return range(0, n_total)
    if batch_index is None:
        raise ValueError(
            "_batch_run_indices: batch_index must be provided when n_batches > 1."
        )
    if not (0 <= batch_index < n_batches):
        raise ValueError(
            f"_batch_run_indices: batch_index must be in [0, {n_batches}), "
            f"got {batch_index}."
        )
    start = (batch_index       * n_total) // n_batches
    stop  = ((batch_index + 1) * n_total) // n_batches
    return range(start, stop)


def _write_bundle_stubs(
    *,
    bundle_dir:      str,
    df_input_energy: pd.DataFrame,
    iso_country:     str,
    target_year:     int,
    n_lhs:           int,
    seed:            int,
    lb:              float,
    ub:              float,
    knob_prefix_filters: List[str],
    spec_columns:    List[str],
    year_min:        int,
    year_max:        int,
    n_batches:       int,
    design_source:   Dict[str, Any],
    tag:             str,
    verbose:         bool,
) -> None:
    """Write baseline.pkl + a stub metadata.json for a batched bundle.

    Idempotent: skips files that already exist so a later batch does
    not clobber whatever the first batch wrote. All batches derive
    the same stub content from their inputs, so any of them can be
    the first writer.
    """
    baseline_path = os.path.join(bundle_dir, "baseline.pkl")
    metadata_path = os.path.join(bundle_dir, "metadata.json")

    if not os.path.exists(baseline_path):
        pd.to_pickle(df_input_energy, baseline_path)
        if verbose:
            print(f"  wrote baseline.pkl (stub) -> {baseline_path}")

    if not os.path.exists(metadata_path):
        stub = {
            "iso_country":              iso_country,
            "target_year":              target_year,
            "n_lhs":                    int(n_lhs),
            "seed":                     int(seed),
            "knob_prefix_filters":      list(knob_prefix_filters),
            "knob_bounds":              [float(lb), float(ub)],
            "n_specs":                  len(spec_columns),
            "spec_columns":             list(spec_columns),
            "year_min":                 int(year_min),
            "year_max":                 int(year_max),
            "consumption_fingerprint":  fingerprint_consumption_state(df_input_energy),
            "generated_at":             datetime.datetime.now(datetime.timezone.utc)
                                            .isoformat(timespec="seconds"),
            "python_version":           sys.version.split()[0],
            "git_commit":               _git_commit_hash(),
            "tag":                      tag,
            "design_path":              design_source["path"],
            "design_source":            design_source["source"],
            "design_hash":              design_source["hash"],
            # Stub -- the merge step overwrites this with n_ok, n_fail,
            # wall_clock summed across batches, etc.
            "batches":                  {"n_batches": int(n_batches)},
        }
        with open(metadata_path, "w") as fp:
            json.dump(stub, fp, indent=2, sort_keys=True)
        if verbose:
            print(f"  wrote metadata.json (stub) -> {metadata_path}")


def merge_bundle_shards(
    bundle_dir: str,
    *,
    min_ok_frac: float = 0.95,
    verbose:     bool  = True,
) -> Tuple[SensitivityResult, Dict[str, Any]]:
    """Assemble the shards a batched sweep produced into a single
    SensitivityResult and write the final bundle artifacts.

    Reads:
      {bundle}/shards/_baseline.pkl         (from the first batch)
      {bundle}/shards/run_{run_idx:06d}.pkl (one per completed run)
      {bundle}/design.csv                   (for input_samples)
      {bundle}/metadata.json                (partial provenance from the
                                            first batch; we overwrite it
                                            with a full-bundle version)
      {bundle}/baseline.pkl                 (the post-v2 input frame,
                                            written by whichever batch
                                            ran first)

    Writes:
      {bundle}/result.pkl                   SensitivityResult
      {bundle}/run_status.csv               per-run status frame
      {bundle}/metadata.json                merged provenance including
                                            wall_clock (sum across
                                            batches), n_lhs, and the
                                            merge quality-gate outcome.

    Applies the same `min_ok_frac` quality gate as the unbatched path;
    refuses to write `result.pkl` when the gate fails (run_status.csv
    is still written for diagnostics).
    """
    import glob
    shards_dir  = os.path.join(bundle_dir, "shards")
    baseline_ck = os.path.join(shards_dir, "_baseline.pkl")
    design_path = os.path.join(bundle_dir, "design.csv")

    for p in (shards_dir, baseline_ck, design_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"merge_bundle_shards: expected {p!r} to exist. Either "
                f"no batch has run yet or the bundle layout is corrupt."
            )

    # Load the design so input_samples can be reconstructed. We rescale
    # here rather than storing scaled samples in each shard.
    unit = pd.read_csv(design_path)
    n_lhs = len(unit)

    # Read all shards, sorted by run_index.
    shard_paths = sorted(glob.glob(os.path.join(shards_dir, "run_*.pkl")))
    if verbose:
        print(f"  merging {len(shard_paths)} shards from {shards_dir}")

    all_outputs, all_comparisons, status_rows = [], [], []
    seen_indices = set()
    for sp in shard_paths:
        payload = pd.read_pickle(sp)
        ri = int(payload["run_index"])
        if ri in seen_indices:
            raise RuntimeError(f"duplicate shard for run_index {ri}: {sp}")
        seen_indices.add(ri)
        status_rows.append({
            "run_index":         ri,
            "elapsed_s":         float(payload["elapsed_s"]),
            "status":            payload["status"],
            "error_class":       payload["error_class"],
            "error_message":     payload["error_message"],
            "has_nemomod_cols":  bool(payload["has_nemomod_cols"]),
        })
        if payload["model_outputs"] is not None:
            all_outputs.append(payload["model_outputs"])
            all_comparisons.append(payload["iea_comparison"])

    if not seen_indices:
        raise RuntimeError(
            f"merge_bundle_shards: no shards found in {shards_dir}. "
            f"Run at least one batch before merging."
        )

    missing = set(range(n_lhs)) - seen_indices
    if missing:
        # Not necessarily fatal -- the caller may know some batches are
        # still running -- but the quality gate below will typically
        # reject anyway. Print prominently.
        print(f"  WARNING: {len(missing)} of {n_lhs} runs have no shard "
              f"(first few missing: {sorted(missing)[:5]})")

    # Rebuild the SensitivityResult. The specs list is not persisted
    # in shards -- it lives in bundle metadata (spec_columns + bounds).
    # We reconstruct enough of it for downstream training code (which
    # reads result.input_samples / result.model_outputs and does not
    # touch variable_specs).
    with open(os.path.join(bundle_dir, "metadata.json")) as fp:
        old_meta = json.load(fp)
    knob_columns = list(old_meta["spec_columns"])
    lb, ub = old_meta["knob_bounds"]
    specs = [
        VariableSpec(column=c, lb=float(lb), ub=float(ub))
        for c in knob_columns
    ]
    samples_df = pd.DataFrame(
        lhs_scale(unit.values, np.full(len(specs), float(lb)),
                                np.full(len(specs), float(ub))),
        columns=knob_columns,
    ).reset_index(drop=True)

    baseline_payload = pd.read_pickle(baseline_ck)

    run_status_df = pd.DataFrame(status_rows, columns=[
        "run_index", "elapsed_s", "status",
        "error_class", "error_message", "has_nemomod_cols",
    ]).sort_values("run_index").reset_index(drop=True)

    empty_out  = pd.DataFrame({"run_index": pd.Series(dtype=int)})
    empty_comp = pd.DataFrame({"run_index": pd.Series(dtype=int)})
    result = SensitivityResult(
        variable_specs          = specs,
        sampling_mode           = "lhs",
        input_samples           = samples_df,
        iea_comparison          = (pd.concat(all_comparisons, ignore_index=True)
                                   if all_comparisons else empty_comp),
        model_outputs           = (pd.concat(all_outputs, ignore_index=True)
                                   if all_outputs else empty_out),
        baseline_output         = baseline_payload["baseline_output"].copy(),
        baseline_iea_comparison = baseline_payload["baseline_iea_comparison"].copy(),
        run_status              = run_status_df,
    )

    # Quality gate at merge time.
    n_total   = len(run_status_df)
    n_ok      = int((run_status_df["status"] == "ok").sum())
    n_fail    = int((run_status_df["status"] == "failed").sum())
    n_missing = int((run_status_df["status"] == "missing_electricity").sum())
    ok_frac   = n_ok / n_total if n_total else 0.0
    if verbose:
        print(f"  merged runs         : {n_ok}/{n_total} ok, "
              f"{n_fail} failed, {n_missing} missing electricity  "
              f"(min_ok_frac={min_ok_frac:.2f})")

    # Always write run_status.csv so a failed gate leaves diagnostics behind.
    run_status_df.to_csv(os.path.join(bundle_dir, "run_status.csv"), index=False)

    if ok_frac < min_ok_frac:
        raise RuntimeError(
            f"merge_bundle_shards: gate failed. Only {n_ok}/{n_total} "
            f"samples succeeded ({ok_frac:.1%} < {min_ok_frac:.0%}). "
            f"See {bundle_dir}/run_status.csv for the per-sample breakdown."
        )

    # Refresh metadata with merge-time totals. We keep whatever the
    # first batch wrote (design_hash, consumption_fingerprint, ...) and
    # replace the batch-specific fields.
    metadata = dict(old_meta)
    metadata["n_ok"]           = n_ok
    metadata["n_fail"]         = n_fail
    metadata["n_missing"]      = n_missing
    metadata["merged_at"]      = datetime.datetime.now(datetime.timezone.utc) \
                                    .isoformat(timespec="seconds")
    metadata["merge_git_commit"] = _git_commit_hash()
    metadata["batches"] = {
        "n_batches":     old_meta.get("batches", {}).get("n_batches"),
        "shards_merged": n_total,
        "runs_missing":  sorted(missing),
    }
    metadata["output_dir"] = bundle_dir

    pd.to_pickle(result, os.path.join(bundle_dir, "result.pkl"))
    with open(os.path.join(bundle_dir, "metadata.json"), "w") as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True)

    if verbose:
        print(f"  wrote result.pkl and metadata.json under {bundle_dir}")

    return result, metadata


def _load_or_sample_unit_design(
    design_path:  Optional[str],
    knob_columns: List[str],
    n_lhs:        int,
    seed:         int,
    verbose:      bool,
) -> Dict[str, Any]:
    """Load a persisted unit-cube LHS design, or sample and (optionally)
    persist a fresh one.

    Returns a dict with keys `unit` (the DataFrame in [0,1]^D),
    `source` ("loaded" | "sampled" | "in_memory"), `path` (str | None),
    and `hash` (SHA-256 hex digest of the design's float64 bytes plus
    its ordered column names -- for provenance / equality checks
    across bundles).

    On load, both the shape and the ordered column set are validated
    against `(n_lhs, knob_columns)`; a mismatch raises loudly so a
    stale design file cannot silently drive a differently-configured
    run.
    """
    import hashlib

    if design_path is not None and os.path.exists(design_path):
        unit = pd.read_csv(design_path)
        if unit.shape != (n_lhs, len(knob_columns)):
            raise ValueError(
                f"design file {design_path!r} has shape {unit.shape}; "
                f"expected ({n_lhs}, {len(knob_columns)}). Check "
                f"--n-lhs and the active knob set."
            )
        loaded_cols = list(unit.columns)
        if loaded_cols != knob_columns:
            # Point at the ACTUAL first mismatching column so a stale
            # design file (created under a different knob set or plan
            # ordering) is easy to diagnose. The error also names any
            # columns present in one set but missing from the other so
            # "renamed knob" cases show up clearly.
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(loaded_cols, knob_columns))
                 if a != b),
                min(len(loaded_cols), len(knob_columns)),
            )
            only_expected = set(knob_columns) - set(loaded_cols)
            only_loaded   = set(loaded_cols)   - set(knob_columns)
            raise ValueError(
                f"design file {design_path!r} column set does not match "
                f"the active specs.\n"
                f"  first diff at position {first_diff}:\n"
                f"    expected: {knob_columns[first_diff] if first_diff < len(knob_columns) else '(end)'}\n"
                f"    got     : {loaded_cols[first_diff]   if first_diff < len(loaded_cols)   else '(end)'}\n"
                f"  columns in the plan but not in the file: "
                f"{sorted(only_expected) if only_expected else 'none'}\n"
                f"  columns in the file but not in the plan: "
                f"{sorted(only_loaded)   if only_loaded   else 'none'}\n"
                f"If the columns are the same but ordered differently, "
                f"delete the file and re-run so a fresh design is written "
                f"with the current plan ordering."
            )
        source = "loaded"
        if verbose:
            print(f"  loaded LHS design   : {design_path}  shape={unit.shape}")
    else:
        unit = sample_lhs_unit_cube(knob_columns, n_lhs, seed)
        source = "sampled"
        if design_path is not None:
            os.makedirs(os.path.dirname(design_path) or ".", exist_ok=True)
            unit.to_csv(design_path, index=False)
            if verbose:
                print(f"  wrote  LHS design   : {design_path}  shape={unit.shape}")
        else:
            source = "in_memory"

    # Hash: canonical bytes of the rounded float64 values + the ordered
    # column names. Rounding before hashing keeps the digest stable
    # across CSV round-trip (12 decimals matches the tolerance used by
    # fingerprint_consumption_state).
    h = hashlib.sha256()
    h.update(repr(list(unit.columns)).encode("utf-8"))
    h.update(np.round(unit.to_numpy(dtype=np.float64), decimals=12).tobytes())
    design_hash = h.hexdigest()

    return {
        "unit":   unit,
        "source": source,
        "path":   design_path,
        "hash":   design_hash,
    }


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
