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
be pinned to that raw consumption state, and ProductionCalibrator will refuse to
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
from pathlib import Path

# Repo root = two levels up from this file (…/sisepuede/calibration/<this>.py).
# Derived from __file__ so the script is portable across machines.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    merge_bundle_shards,
)


# ── Default paths (override on the command line) ──────────────────────────
# External inputs come from env vars; the crosswalk ships with the repo, so
# its default is derived from the repo root. Training output falls back to
# $SISEPUEDE_TRAINING_DIR and, if that's unset, to an in-repo default.
DEFAULT_CALIBRATED_INPUT = "input_data_per_calibrated.csv"
DEFAULT_IEA_DATA_DIR     = os.environ.get("IEA_DATA_DIR")
DEFAULT_CROSSWALK_FILE   = str(
    _REPO_ROOT / "sisepuede" / "ref" / "data_crosswalks"
    / "sisepuede_iea_energy_crosswalk.csv"
)
DEFAULT_OUTPUT_DIR       = os.environ.get(
    "SISEPUEDE_TRAINING_DIR",
    str(_REPO_ROOT / "sisepuede" / "out" / "training_data"),
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate LHS training data for the v3 energy-production surrogate.",
    )

    p.add_argument("--country",     type=str, default=None,
                   help="ISO-3 country code (e.g. PER, ARG, KEN). "
                        "Required except with --design-only.")
    p.add_argument("--target-year", type=int, default=None,
                   help="Calendar year the surrogate is trained to predict at. "
                        "Required except with --design-only.")
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

    # Paths. External locations fall back to env vars ($IEA_DATA_DIR,
    # $SISEPUEDE_TRAINING_DIR); the IEA dir is required if neither the flag
    # nor the env var is set.
    p.add_argument("--calibrated-input", type=str, default=DEFAULT_CALIBRATED_INPUT,
                   help="Path to the post-v2 calibrated SSP input CSV.")
    p.add_argument("--iea-data-dir",     type=str, default=DEFAULT_IEA_DATA_DIR,
                   required=DEFAULT_IEA_DATA_DIR is None,
                   help="Directory holding the IEA energy balance files (env: IEA_DATA_DIR).")
    p.add_argument("--crosswalk-file",   type=str, default=DEFAULT_CROSSWALK_FILE,
                   help="Path to the IEA-SISEPUEDE crosswalk CSV "
                        "(default: repo copy under sisepuede/ref/data_crosswalks/).")
    p.add_argument("--output-dir",       type=str, default=DEFAULT_OUTPUT_DIR,
                   help=("Where the training-data subdirectory is written "
                         "(env: SISEPUEDE_TRAINING_DIR; default: repo "
                         "sisepuede/out/training_data/)."))
    p.add_argument("--tag",              type=str, default="",
                   help="Optional suffix appended to the output subdirectory name.")

    p.add_argument("--min-ok-frac", type=float, default=0.95,
                   help=("Minimum fraction of LHS samples that must succeed "
                         "(status='ok'). The sweep is aborted (nothing "
                         "pickled) when the observed rate is lower -- "
                         "see run_status.csv for the per-sample breakdown. "
                         "Default 0.95."))

    p.add_argument(
        "--design-path", type=str, default=None,
        help=(
            "Path to a CSV file holding the unit-cube [0,1]^D LHS design "
            "(rows = samples, columns = knob names, values in [0,1]). "
            "If the file exists, the design is loaded from it (sampling "
            "is skipped) and rescaled to --knob-lb/--knob-ub; if it "
            "does not exist, a fresh design is sampled from --seed / "
            "--n-lhs and written to that path so subsequent runs "
            "(other countries, years, or bound choices) can reuse the "
            "same sample points. When omitted, the design is sampled "
            "in-process and not persisted separately."
        ),
    )

    # Batched execution.
    p.add_argument("--n-batches", type=int, default=1,
                   help=("Split the sweep into K contiguous batches by "
                         "run_index. Each invocation must set "
                         "--batch-index i for some i in [0, K); the "
                         "worker executes runs [floor(i*N/K), floor((i+1)*N/K)) "
                         "and writes per-run pickle shards to {bundle}/shards/. "
                         "Default 1 = no batching."))
    p.add_argument("--batch-index", type=int, default=None,
                   help="0-based index of this batch. Required with --n-batches > 1.")
    p.add_argument("--merge-only", action="store_true",
                   help=("Skip sampling / execution and assemble a "
                         "SensitivityResult from shards under "
                         "{output-dir}/{iso3}_{year}_n{N}_seed{S}/shards/. "
                         "Writes result.pkl + run_status.csv + "
                         "metadata.json. Applies the same --min-ok-frac "
                         "quality gate as the unbatched path."))

    p.add_argument("--design-only", action="store_true",
                   help=("Only sample the unit-cube LHS design and write "
                         "it to --design-path, then exit. Skips SISEPUEDE / "
                         "NemoMod / Julia init entirely; needs only "
                         "--seed / --n-lhs / --design-path (and --knob-prefix "
                         "if overriding defaults). Use this to pre-generate "
                         "a shared design that will drive multiple batches, "
                         "countries, or target years -- no SISEPUEDE inputs "
                         "loaded, no country data touched."))

    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress prints.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    verbose = not args.quiet

    # Post-parse validation for the mode-dependent required args.
    if not args.design_only:
        missing = [
            n for n, v in [("--country", args.country),
                           ("--target-year", args.target_year)]
            if v is None
        ]
        if missing:
            raise SystemExit(
                f"error: the following arguments are required unless "
                f"--design-only is set: {', '.join(missing)}"
            )

    if verbose:
        print("=" * 72)
        if args.design_only:
            print(" v3 LHS design-only generation")
        else:
            print(f" v3 training-data generation  --  {args.country} / {args.target_year}")
        print(f" n_lhs = {args.n_lhs}  seed = {args.seed}  "
              f"bounds = [{args.knob_lb}, {args.knob_ub}]")
        print("=" * 72)

    # ── Design-only: no country data, no NemoMod init. Build the
    # calibration plan just enough to determine the knob column set,
    # then sample the unit-cube design and write it to --design-path.
    if args.design_only:
        if not args.design_path:
            raise SystemExit("error: --design-only requires --design-path")
        from sisepuede.calibration.build_energy_calibration_plan import (
            build_energy_calibration_plan,
        )
        from sisepuede.calibration._data_generation import (
            _load_or_sample_unit_design, DEFAULT_KNOB_PREFIX_FILTERS,
        )

        if verbose:
            print("\n[1/2] Initialising SISEPUEDEFileStructure (no NemoMod)...")
        file_structure   = SISEPUEDEFileStructure()
        model_attributes = file_structure.model_attributes

        if verbose:
            print("\n[2/2] Building calibration plan + sampling design...")
        knob_prefix_filters = args.knob_prefix or list(DEFAULT_KNOB_PREFIX_FILTERS)
        plan = build_energy_calibration_plan(model_attributes)
        prod_groups = [g for g in plan.groups if g.name.startswith("electout__")]
        specs = [
            s for g in prod_groups for s in g.specs
            if any(s.column.startswith(p) for p in knob_prefix_filters)
        ]
        if not specs:
            raise SystemExit(
                f"error: no specs matched any prefix in {knob_prefix_filters}. "
                f"Check --knob-prefix or the calibration plan."
            )
        knob_columns = [s.column for s in specs]
        if verbose:
            print(f"      knob prefix filters: {knob_prefix_filters}")
            print(f"      D (num knobs)      : {len(knob_columns)}")
            print(f"      first 3 columns    : {knob_columns[:3]}")

        _load_or_sample_unit_design(
            design_path  = args.design_path,
            knob_columns = knob_columns,
            n_lhs        = args.n_lhs,
            seed         = args.seed,
            verbose      = verbose,
        )
        if verbose:
            print(f"\nDone. Design saved to: {args.design_path}")
        return

    # ── Merge-only: no SISEPUEDE / NemoMod init needed, no country
    # data reload. Just walk the existing bundle's shards and assemble
    # result.pkl + run_status.csv + metadata.json.
    if args.merge_only:
        bundle_dir = os.path.join(
            args.output_dir,
            f"{args.country}_{args.target_year}_n{args.n_lhs}_seed{args.seed}"
            + (f"_{args.tag}" if args.tag else ""),
        )
        if verbose:
            print(f"\nmerge-only mode  --  bundle: {bundle_dir}")
        result, metadata = merge_bundle_shards(
            bundle_dir  = bundle_dir,
            min_ok_frac = args.min_ok_frac,
            verbose     = verbose,
        )
        if verbose:
            print(f"\nDone. Output: {metadata.get('output_dir')}")
        return

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
        min_ok_frac         = args.min_ok_frac,
        design_path         = args.design_path,
        n_batches           = args.n_batches,
        batch_index         = args.batch_index,
    )

    if verbose:
        print(f"\nDone. Output: {metadata.get('output_dir')}")


if __name__ == "__main__":
    main()
