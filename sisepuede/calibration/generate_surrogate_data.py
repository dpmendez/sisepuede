"""
sisepuede/calibration/generate_surrogate_data.py

CLI wrapper for `_data_generation.generate_lhs_training_data`.

The v3 pipeline has three surrogate-specific CLIs plus the shared
`run_energy_calibration.py`:

  1. run_energy_calibration.py --cal-option 3   (v2: run first to calibrate consumption)
  2. generate_surrogate_data.py                 (this: LHS sweep -> training data pickle)
  3. train_surrogate.py                         (train + evaluate + gate -> surrogate artifact)
  4. run_energy_calibration.py --cal-option 5   (v3 inference using the surrogate)

Example
-------

    source ~/sisepuede-env/bin/activate
    python -m sisepuede.calibration.generate_surrogate_data \\
        --country PER \\
        --target-year 2018 \\
        --n-lhs 2000 \\
        --seed 42 \\
        --calibrated-input <path_to_calibrated_input>/input_data.csv \\

The input CSV **should be a post-consumption-calibration (post-v2) input**.
Passing a raw uncalibrated CSV still works but the surrogate you train will
be pinned to that raw consumption state, and CalibratorV3 will refuse to
apply it (fingerprint mismatch) at inference time.

Outputs (under `--output-dir`):

    {output_dir}/{iso3}_{year}_n{N}_seed{S}{_tag}/
        result.pkl        SensitivityResult (scale factors + all SSP outputs)
        baseline.pkl      the calibrated post-AFOLU/IPPU input frame
        metadata.json     provenance (fingerprint, spec list, seed, wall-clock, git commit)
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
from sisepuede.manager.sisepuede_models         import SISEPUEDEModels

from sisepuede.calibration.build_iea_energy_crosswalk import IEACrosswalkBuilder
from sisepuede.calibration.iea_crosswalk              import IEACrosswalk
from sisepuede.calibration.iea_data_loader            import IEADataLoader
from sisepuede.calibration.energy_calibration         import (
    _build_energy_input_frame,
    _load_inputs,
)
from sisepuede.calibration._data_generation           import (
    DEFAULT_KNOB_BOUNDS,
    DEFAULT_KNOB_PREFIX_FILTERS,
    generate_lhs_training_data,
)


# ── Default paths (override on the command line) ──────────────────────────
DEFAULT_CALIBRATED_INPUT = "/Users/dianamendez/sisepuede-data/input_data_peru_base.csv"
DEFAULT_IEA_DATA_DIR     = "/Users/dianamendez/data_collection_temporary"
DEFAULT_CROSSWALK_FILE   = (
    f"{_REPO_ROOT}/sisepuede/ref/data_crosswalks/sisepuede_iea_energy_crosswalk.csv"
)
DEFAULT_OUTPUT_DIR       = f"{_REPO_ROOT}/sisepuede/out/training_data"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate LHS training data for the v3 energy-production surrogate.",
    )

    p.add_argument("--country",     type=str, required=True,
                   help="ISO-3 country code (e.g. PER, ARG, KEN).")
    p.add_argument("--target-year", type=int, required=True,
                   help="Calendar year the surrogate is trained to predict at.")
    p.add_argument("--n-lhs",       type=int, default=2000,
                   help="Number of LHS samples (default: 2000).")
    p.add_argument("--seed",        type=int, default=42,
                   help="LHS random seed (default: 42).")

    # Time window used for the IEA comparison table (same defaults as run_energy_calibration).
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year",   type=int, default=2022)

    # Knob configuration -- defaults match the SSP-endorsed policy.
    p.add_argument(
        "--knob-prefix", action="append", default=None,
        help=(
            "SSP input prefix (e.g. nemomod_entc_scalar_availability_factor_) "
            "to include in the LHS. Pass multiple times to include several. "
            f"Defaults to the SSP-endorsed policy: {DEFAULT_KNOB_PREFIX_FILTERS}."
        ),
    )
    p.add_argument("--knob-lb", type=float, default=DEFAULT_KNOB_BOUNDS[0],
                   help=f"Scale-factor lower bound (default: {DEFAULT_KNOB_BOUNDS[0]}).")
    p.add_argument("--knob-ub", type=float, default=DEFAULT_KNOB_BOUNDS[1],
                   help=f"Scale-factor upper bound (default: {DEFAULT_KNOB_BOUNDS[1]}).")

    # Paths.
    p.add_argument("--calibrated-input", type=str, default=DEFAULT_CALIBRATED_INPUT,
                   help="Path to the post-v2 calibrated SSP input CSV.")
    p.add_argument("--iea-data-dir",     type=str, default=DEFAULT_IEA_DATA_DIR)
    p.add_argument("--crosswalk-file",   type=str, default=DEFAULT_CROSSWALK_FILE)
    p.add_argument("--output-dir",       type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Where the training-data subdirectory is written.")
    p.add_argument("--tag",              type=str, default="",
                   help="Optional suffix appended to the output subdirectory name.")

    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress prints.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    verbose = not args.quiet

    if verbose:
        print("=" * 72)
        print(f" v3 training-data generation  --  {args.country} / {args.target_year}")
        print(f" n_lhs = {args.n_lhs}  seed = {args.seed}  "
              f"bounds = [{args.knob_lb}, {args.knob_ub}]")
        print("=" * 72)

    # ── SISEPUEDE setup ─────────────────────────────────────────────────
    if verbose:
        print("\n[1/4] Initialising SISEPUEDEFileStructure + models...")
    file_structure   = SISEPUEDEFileStructure()
    model_attributes = file_structure.model_attributes
    if not file_structure.allow_electricity_run:
        raise RuntimeError(
            "SISEPUEDEFileStructure.allow_electricity_run is False. v3 "
            "requires NemoMod / Julia; check reference files."
        )

    models = SISEPUEDEModels(
        model_attributes,
        allow_electricity_run      = True,
        fp_julia                   = file_structure.dir_jl,
        fp_nemomod_reference_files = file_structure.dir_ref_nemo,
        fp_nemomod_temp_sqlite_db  = file_structure.fp_sqlite_tmp_nemomod_intermediate,
        initialize_julia           = True,
    )

    # ── Load calibrated input and pre-run AFOLU + IPPU ──────────────────
    if verbose:
        print("\n[2/4] Loading input CSV + pre-running AFOLU + IPPU...")
    df_input = _load_inputs(
        args.calibrated_input, args.start_year, args.end_year, verbose,
    )
    df_input_energy = _build_energy_input_frame(df_input, model_attributes, verbose)

    # ── Build crosswalk + load IEA data ─────────────────────────────────
    if verbose:
        print("\n[3/4] Building crosswalk + loading IEA data...")
    IEACrosswalkBuilder(model_attributes, args.crosswalk_file).build(write_csv=True)
    xw         = IEACrosswalk(model_attributes, path_crosswalk=args.crosswalk_file)
    loader     = IEADataLoader(args.iea_data_dir, model_attributes)
    df_iea_raw = loader.load_country(args.country)
    if verbose:
        print(f"      crosswalk pairs: {len(xw.df_crosswalk)}")
        print(f"      IEA rows for {args.country}: {len(df_iea_raw)}  "
              f"({df_iea_raw['year'].min()}–{df_iea_raw['year'].max()})")

    # ── LHS sweep + persist ─────────────────────────────────────────────
    if verbose:
        print(f"\n[4/4] Running LHS + persisting artifacts...")
    result, metadata = generate_lhs_training_data(
        df_input_energy     = df_input_energy,
        iso_country         = args.country,
        target_year         = args.target_year,
        n_lhs               = args.n_lhs,
        seed                = args.seed,
        knob_prefix_filters = args.knob_prefix,           # None -> use defaults
        knob_bounds         = (args.knob_lb, args.knob_ub),
        year_min            = args.start_year,
        year_max            = args.end_year,
        model_attributes    = model_attributes,
        iea_crosswalk       = xw,
        df_iea_raw          = df_iea_raw,
        models              = models,
        output_dir          = args.output_dir,
        tag                 = args.tag,
        verbose             = verbose,
    )

    if verbose:
        print(f"\nDone. Output: {metadata.get('output_dir')}")


if __name__ == "__main__":
    main()
