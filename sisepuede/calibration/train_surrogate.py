"""
sisepuede/calibration/train_surrogate.py

CLI wrapper for `_training_pipeline.train_surrogate_from_data`.

Consumes a training-data directory previously produced by
`generate_surrogate_data.py`, trains a Surrogate on a train/dev/test split,
applies the accuracy gate to the test-set report, and persists the fitted
model plus its crosswalk aggregation matrix and provenance metadata.

Does NOT run SISEPUEDE. Pure ML.

Example
-------

    source ~/sisepuede-env/bin/activate
    python -m sisepuede.calibration.train_surrogate \\
        --data /path/to/sisepuede/out/training_data/PER_2018_n2000_seed42 \\
        --output-dir sisepuede/out/surrogates \\
        --model-kind gbm \\
        --r2-min 0.85 \\
        --mape-max 20.0

Outputs (under `--output-dir`):

    {output_dir}/{iso3}_{year}[_{tag}]/
        surrogate.joblib   the Surrogate model
        aggregation.pkl    {'iea_target_rows', 'ssp_columns', 'A_crosswalk'}
        metadata.json      per-target R^2/MAPE (dev + test), gate verdict,
                           training-data provenance, spec + splits
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

_REPO_ROOT = "/Users/dianamendez/feature-energy-calibration"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

warnings.filterwarnings("ignore")

from sisepuede.manager.sisepuede_file_structure import SISEPUEDEFileStructure

from sisepuede.calibration.iea_crosswalk       import IEACrosswalk
from sisepuede.calibration._surrogate          import SurrogateSpec
from sisepuede.calibration._training_pipeline import (
    DEFAULT_V3_IEA_TARGETS,
    train_surrogate_from_data,
)


# ── Default paths (override on the command line) ──────────────────────────
DEFAULT_CROSSWALK_FILE = (
    f"{_REPO_ROOT}/sisepuede/ref/data_crosswalks/sisepuede_iea_energy_crosswalk.csv"
)
DEFAULT_OUTPUT_DIR = f"{_REPO_ROOT}/sisepuede/out/surrogates"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a v3 surrogate from a training-data directory.",
    )

    # Inputs.
    p.add_argument("--data",           type=str, required=True,
                   help=("Training-data directory (output of "
                         "generate_surrogate_data.py). Must contain "
                         "result.pkl, baseline.pkl, metadata.json."))
    p.add_argument("--crosswalk-file", type=str, default=DEFAULT_CROSSWALK_FILE)

    # Optional overrides.
    p.add_argument("--target-year", type=int, default=None,
                   help=("Calendar year to train at. Default: value stored "
                         "in the training-data metadata.json."))
    p.add_argument("--iea-target",  action="append", default=None,
                   metavar="BAL,PROD",
                   help=("Comma-separated `iea_balance_code,iea_product_code` "
                         "to include as a target row. Pass multiple times to "
                         f"add several. Defaults to the canonical v3 list "
                         f"({len(DEFAULT_V3_IEA_TARGETS)} ELECTOUT rows)."))

    # SurrogateSpec knobs.
    p.add_argument("--model-kind", type=str, default="gbm",
                   choices=["gbm", "xgb", "lgbm"])
    p.add_argument("--seed",       type=int, default=42,
                   help="Regressor RNG seed.")
    p.add_argument("--r2-min",     type=float, default=0.85,
                   help="Accuracy gate: minimum test-set R^2 to accept a target.")
    p.add_argument("--mape-max",   type=float, default=20.0,
                   help="Accuracy gate: maximum test-set MAPE (%%) to accept a target.")
    p.add_argument("--fail-mode",  type=str, default="warn",
                   choices=["warn", "raise"])

    # Split knobs.
    p.add_argument("--split-train", type=float, default=0.7)
    p.add_argument("--split-dev",   type=float, default=0.15)
    p.add_argument("--split-test",  type=float, default=0.15)
    p.add_argument("--split-seed",  type=int,   default=42)

    # Output.
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--tag",        type=str, default="",
                   help="Optional suffix appended to the output subdirectory name.")

    p.add_argument("--quiet", action="store_true")
    return p


def _parse_iea_targets(raw: list) -> list:
    """Parse `--iea-target BAL,PROD` arguments into (balance, product) tuples."""
    out = []
    for entry in raw:
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"--iea-target must be `BALANCE,PRODUCT`, got {entry!r}"
            )
        out.append((parts[0], parts[1]))
    return out


def main() -> None:
    args = _build_parser().parse_args()
    verbose = not args.quiet

    iea_targets = (
        _parse_iea_targets(args.iea_target) if args.iea_target else None
    )

    if verbose:
        print("=" * 72)
        print(f" v3 surrogate training  --  data: {args.data}")
        print(f" model: {args.model_kind}  splits: "
              f"({args.split_train}, {args.split_dev}, {args.split_test})  "
              f"gate: R^2>={args.r2_min}, MAPE<={args.mape_max}%")
        print("=" * 72)

    # ── SISEPUEDE ModelAttributes + crosswalk ───────────────────────────
    file_structure   = SISEPUEDEFileStructure()
    model_attributes = file_structure.model_attributes
    xw = IEACrosswalk(model_attributes, path_crosswalk=args.crosswalk_file)

    # ── SurrogateSpec ───────────────────────────────────────────────────
    spec = SurrogateSpec(
        model_kind      = args.model_kind,
        seed            = args.seed,
        target_r2_min   = args.r2_min,
        target_mape_max = args.mape_max,
        fail_mode       = args.fail_mode,
    )

    # ── Train + evaluate + gate + persist ───────────────────────────────
    result = train_surrogate_from_data(
        training_data_dir = args.data,
        crosswalk         = xw,
        iea_target_rows   = iea_targets,
        target_year       = args.target_year,
        spec              = spec,
        split_fractions   = (args.split_train, args.split_dev, args.split_test),
        split_seed        = args.split_seed,
        output_dir        = args.output_dir,
        tag               = args.tag,
        verbose           = verbose,
    )

    if verbose:
        print(f"\nDone. Output: {result.get('output_dir')}")


if __name__ == "__main__":
    main()
