"""
sisepuede/calibration/_training_pipeline.py

Orchestrator for training a v3 surrogate from a persisted training-data
directory.

The subroutine `train_surrogate_from_data()` composes the primitives from
`_surrogate.py` (Surrogate.train / evaluate, apply_accuracy_gate),
`_training_set.py` (build_training_matrices, crosswalk_ssp_columns_for_iea_targets,
split_train_dev_test), and `_data_generation.load_training_data()` into one
end-to-end call:

    load training data
      -> resolve IEA target rows + crosswalk aggregation matrix A
      -> extract raw per-tech (X, Y) at target_year
      -> split train/dev/test
      -> Surrogate.train on the train split
      -> Surrogate.evaluate on dev + test splits
      -> apply_accuracy_gate on the test report
      -> persist surrogate.joblib + aggregation.pkl + metadata.json

The output directory has the same shape as generate_surrogate_data's:

    {output_dir}/{iso3}_{year}[_{tag}]/
        surrogate.joblib      # the Surrogate model
        aggregation.pkl       # {'iea_target_rows', 'ssp_columns', 'A'}
        metadata.json         # per-target R^2/MAPE (dev+test), gate verdict,
                              # training-data provenance, spec + splits

CalibratorV3 loads all three at inference time. The consumption 
fingerprint stored in the metadata is what the calibrator checks
against the current input frame to refuse mismatched surrogates.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sisepuede.calibration._data_generation import load_training_data
from sisepuede.calibration._surrogate import (
    Surrogate,
    SurrogateReport,
    SurrogateSpec,
    apply_accuracy_gate,
)
from sisepuede.calibration._training_set import (
    build_training_matrices,
    crosswalk_ssp_columns_for_iea_targets,
    split_train_dev_test,
)


# Canonical v3 IEA target list. Every ELECTOUT (per-fuel electricity
# generation) row except TOTAL (which is a redundant sum). Country-agnostic;
# rows the crosswalk cannot map for a given country just get zeros through
# the aggregation matrix, and the surrogate learns constants for them.
DEFAULT_V3_IEA_TARGETS: List[Tuple[str, str]] = [
    ("ELECTOUT", "COAL"),
    ("ELECTOUT", "OIL"),
    ("ELECTOUT", "NATGAS"),
    ("ELECTOUT", "NUCLEAR"),
    ("ELECTOUT", "HYDRO"),
    ("ELECTOUT", "WIND"),
    ("ELECTOUT", "SOLARPV"),
    ("ELECTOUT", "SOLARTH"),
    ("ELECTOUT", "BIOFUEL"),
    ("ELECTOUT", "WASTE"),
    ("ELECTOUT", "GEOTHERM"),
    ("ELECTOUT", "TIDE"),
]


def train_surrogate_from_data(
    training_data_dir: str,
    crosswalk:         Any,
    *,
    iea_target_rows:   Optional[List[Tuple[str, str]]] = None,
    target_year:       Optional[int] = None,
    spec:              Optional[SurrogateSpec] = None,
    split_fractions:   Tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_seed:        int = 42,
    output_dir:        Optional[str] = None,
    tag:               str = "",
    verbose:           bool = True,
) -> Dict[str, Any]:
    """Train + evaluate + gate a surrogate from a persisted training-data dir.

    Parameters
    ----------
    training_data_dir : str
        Directory previously written by `generate_surrogate_data.py`.
        Must contain `result.pkl`, `baseline.pkl`, `metadata.json`.
    crosswalk : IEACrosswalk
        Provides the mapping from IEA target rows to raw SSP fields.
    iea_target_rows : List[Tuple[str, str]] | None
        Which IEA cells the aggregation matrix will fold surrogate
        predictions into. Defaults to `DEFAULT_V3_IEA_TARGETS` (all
        ELECTOUT fuels). Callers can pass a subset or extend it.
    target_year : int | None
        Calendar year at which to build (X, Y). Defaults to
        metadata['target_year'] from the training-data directory.
    spec : SurrogateSpec | None
        Backend + hyperparameters + gate thresholds. Defaults to
        `SurrogateSpec()` (GBM, holdout_frac unused here since we split
        explicitly, target_r2_min=0.85, target_mape_max=20.0).
    split_fractions : (train, dev, test)
        Fractions must sum to <= 1.0. Default (0.7, 0.15, 0.15).
    split_seed : int
        Reproducibility seed for the train/dev/test split.
    output_dir : str | None
        When provided, persists artifacts to
        `{output_dir}/{iso3}_{year}[_{tag}]/`.
    tag : str
        Optional suffix appended to the output subdirectory name.
    verbose : bool

    Returns
    -------
    dict
        Keys: `surrogate`, `dev_report`, `test_report`, `iea_target_rows`,
        `ssp_columns`, `A`, `training_data_dir`, `metadata`,
        `output_dir` (if persisted).
    """
    t0 = time.time()

    # ── Load training data ─────────────────────────────────────────────
    result, baseline, td_meta = load_training_data(training_data_dir)
    if target_year is None:
        target_year = int(td_meta["target_year"])
    if iea_target_rows is None:
        iea_target_rows = list(DEFAULT_V3_IEA_TARGETS)
    if spec is None:
        spec = SurrogateSpec()

    if verbose:
        print(f"[train_surrogate_from_data]")
        print(f"  training data : {training_data_dir}")
        print(f"  target_year   : {target_year}")
        print(f"  iea targets   : {len(iea_target_rows)}")
        print(f"  spec.model_kind = {spec.model_kind}, seed = {spec.seed}")

    # ── Resolve crosswalk aggregation ──────────────────────────────────
    ssp_columns, A = crosswalk_ssp_columns_for_iea_targets(
        iea_target_rows, crosswalk,
    )
    if not ssp_columns:
        raise RuntimeError(
            "train_surrogate_from_data: no SSP columns resolved from the "
            "requested IEA target rows. Check the crosswalk / target list."
        )
    if verbose:
        print(f"  ssp columns   : {len(ssp_columns)}")

    # ── Extract raw (X, Y) at target_year ──────────────────────────────
    X, Y = build_training_matrices(result, target_year, ssp_columns)
    if verbose:
        print(f"  X, Y shapes   : {X.shape}, {Y.shape}")

    # ── Split ──────────────────────────────────────────────────────────
    splits = split_train_dev_test(X, Y, fractions=split_fractions, seed=split_seed)
    (X_tr, Y_tr), (X_dv, Y_dv), (X_ts, Y_ts) = (
        splits["train"], splits["dev"], splits["test"],
    )
    if verbose:
        print(f"  splits        : train={X_tr.shape[0]}, dev={X_dv.shape[0]}, "
              f"test={X_ts.shape[0]}")

    # ── Envelope bounds come from the recorded knob box ────────────────
    knob_lb, knob_ub = td_meta.get("knob_bounds", [0.9, 1.1])
    n_knobs = X.shape[1]
    lb_arr = np.full(n_knobs, float(knob_lb))
    ub_arr = np.full(n_knobs, float(knob_ub))

    # ── Train ──────────────────────────────────────────────────────────
    consumption_fp = td_meta["consumption_fingerprint"]
    surrogate = Surrogate.train(
        X_tr, Y_tr, spec, consumption_fp,
        envelope_bounds=(lb_arr, ub_arr),
    )
    if verbose:
        print(f"  trained       : {surrogate}")

    # ── Evaluate on dev and test ───────────────────────────────────────
    dev_report  = surrogate.evaluate(X_dv, Y_dv)  if X_dv.shape[0] else _empty_report(surrogate)
    test_report = surrogate.evaluate(X_ts, Y_ts)  if X_ts.shape[0] else _empty_report(surrogate)

    # ── Apply the accuracy gate to the TEST report only ────────────────
    # The dev report is for hyperparameter tuning; the test report is the
    # honest generalisation estimate that governs which targets survive
    # into inference.
    apply_accuracy_gate(surrogate, test_report, spec)

    # ── Metadata ───────────────────────────────────────────────────────
    wall_clock = time.time() - t0
    metadata: Dict[str, Any] = {
        "iso_country":              td_meta["iso_country"],
        "target_year":              int(target_year),
        "training_data_dir":        os.path.abspath(training_data_dir),
        "consumption_fingerprint":  consumption_fp,
        "iea_target_rows":          [list(t) for t in iea_target_rows],
        "ssp_columns":              list(ssp_columns),
        "spec":                     asdict(spec),
        "split_fractions":          list(split_fractions),
        "split_seed":               int(split_seed),
        "n_train":                  int(X_tr.shape[0]),
        "n_dev":                    int(X_dv.shape[0]),
        "n_test":                   int(X_ts.shape[0]),
        "dev_report":               _serialize_report(dev_report),
        "test_report":              _serialize_report(test_report),
        "wall_clock_seconds":       float(wall_clock),
        "generated_at":             datetime.datetime.now(datetime.timezone.utc)
                                        .isoformat(timespec="seconds"),
        "python_version":           sys.version.split()[0],
        "git_commit":               _git_commit_hash(),
        "tag":                      tag,
    }

    if verbose:
        _print_report_summary("dev",  dev_report)
        _print_report_summary("test", test_report)
        print(f"  accepted      : {len(test_report.accepted_targets)} "
              f"({[t for t in test_report.accepted_targets]})")
        print(f"  rejected      : {len(test_report.rejected_targets)} "
              f"({[t for t in test_report.rejected_targets]})")

    # ── Persist ────────────────────────────────────────────────────────
    result_out: Dict[str, Any] = {
        "surrogate":         surrogate,
        "dev_report":        dev_report,
        "test_report":       test_report,
        "iea_target_rows":   iea_target_rows,
        "ssp_columns":       ssp_columns,
        "A":       A,
        "training_data_dir": training_data_dir,
        "metadata":          metadata,
    }
    if output_dir is not None:
        subdir = _build_output_subdir(
            output_dir, td_meta["iso_country"], target_year, tag,
        )
        _persist_artifacts(
            subdir, surrogate, iea_target_rows, ssp_columns, A,
            metadata, verbose,
        )
        metadata["output_dir"] = subdir
        result_out["output_dir"] = subdir

    return result_out


##########################
#    LOAD HELPER         #
##########################

def load_surrogate_bundle(
    subdir: str,
) -> Dict[str, Any]:
    """Inverse of train_surrogate_from_data's persistence.

    Returns dict with keys `surrogate`, `iea_target_rows`, `ssp_columns`,
    `A`, `metadata`.
    """
    surrogate  = Surrogate.load(os.path.join(subdir, "surrogate.joblib"))
    agg        = joblib.load(os.path.join(subdir, "aggregation.pkl"))
    with open(os.path.join(subdir, "metadata.json")) as fp:
        metadata = json.load(fp)
    return {
        "surrogate":       surrogate,
        "iea_target_rows": [tuple(t) for t in agg["iea_target_rows"]],
        "ssp_columns":     list(agg["ssp_columns"]),
        "A":     np.asarray(agg["A"]),
        "metadata":        metadata,
    }


##########################
#    INTERNAL HELPERS    #
##########################

def _empty_report(surrogate: Surrogate) -> SurrogateReport:
    """Placeholder report for an empty eval split. Auto-accept everything."""
    targets = list(surrogate.targets)
    nan_series = pd.Series({t: float("nan") for t in targets}, dtype=float)
    return SurrogateReport(
        targets          = targets,
        r2_per_target    = nan_series.copy(),
        mape_per_target  = nan_series.copy(),
        accepted_targets = [],
        rejected_targets = [],
        n_train          = int(getattr(surrogate, "_n_train_samples", -1)),
        n_holdout        = 0,
    )


def _serialize_report(report: SurrogateReport) -> Dict[str, Any]:
    """SurrogateReport -> JSON-safe dict for metadata.json.

    v3 surrogate targets are raw SSP column *names* (strings), stored as
    strings. Older / hybrid usage where targets are (balance, product)
    tuples gets serialised as 2-element lists.
    """
    return {
        "n_train":          int(report.n_train),
        "n_holdout":        int(report.n_holdout),
        "accepted_targets": [_serialize_target(t) for t in report.accepted_targets],
        "rejected_targets": [_serialize_target(t) for t in report.rejected_targets],
        "r2_per_target":    {
            _stringify_target(t): _safe_json_float(v)
            for t, v in report.r2_per_target.items()
        },
        "mape_per_target":  {
            _stringify_target(t): _safe_json_float(v)
            for t, v in report.mape_per_target.items()
        },
    }


def _serialize_target(t: Any) -> Any:
    """JSON-safe representation of a surrogate target.

    Tuples (from the older IEA-cell-target design) become 2-element
    lists. Strings (v3's raw per-tech SSP columns) stay strings.
    """
    if isinstance(t, tuple):
        return list(t)
    return t


def _stringify_target(t: Any) -> str:
    """Compact string form for JSON dict keys.

    Tuples -> 'balance|product'. Strings -> themselves.
    """
    if isinstance(t, tuple):
        return "|".join(str(x) for x in t)
    return str(t)


def _safe_json_float(x: Any) -> Optional[float]:
    """NaN / inf -> None so json.dump doesn't error."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _print_report_summary(label: str, report: SurrogateReport) -> None:
    if report.n_holdout == 0:
        print(f"  {label}: (empty)")
        return
    r2   = report.r2_per_target.dropna()
    mape = report.mape_per_target.dropna()
    r2_line   = f"R^2 median={r2.median():.3f}, min={r2.min():.3f}"      if len(r2)   else "R^2: n/a"
    mape_line = f"MAPE median={mape.median():.1f}%, max={mape.max():.1f}%" if len(mape) else "MAPE: n/a"
    print(f"  {label:<5}: n={report.n_holdout}, {r2_line}, {mape_line}")


def _build_output_subdir(
    output_dir:  str,
    iso_country: str,
    target_year: int,
    tag:         str,
) -> str:
    """Canonical output subdirectory name."""
    stem = f"{iso_country}_{target_year}"
    if tag:
        stem = f"{stem}_{tag}"
    subdir = os.path.join(output_dir, stem)
    os.makedirs(subdir, exist_ok=True)
    return subdir


def _persist_artifacts(
    subdir:          str,
    surrogate:       Surrogate,
    iea_target_rows: List[Tuple[str, str]],
    ssp_columns:     List[str],
    A:     np.ndarray,
    metadata:        Dict[str, Any],
    verbose:         bool,
) -> None:
    """Write surrogate.joblib, aggregation.pkl, metadata.json into `subdir`."""
    sur_path = os.path.join(subdir, "surrogate.joblib")
    agg_path = os.path.join(subdir, "aggregation.pkl")
    md_path  = os.path.join(subdir, "metadata.json")

    surrogate.save(sur_path)
    joblib.dump(
        {
            "iea_target_rows": iea_target_rows,
            "ssp_columns":     ssp_columns,
            "A":     A,
        },
        agg_path,
    )
    with open(md_path, "w") as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True)

    if verbose:
        print(f"  saved: {subdir}")
        print(f"    surrogate.joblib  ({_file_size_mb(sur_path):.2f} MB)")
        print(f"    aggregation.pkl   ({_file_size_mb(agg_path):.2f} MB)")
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
