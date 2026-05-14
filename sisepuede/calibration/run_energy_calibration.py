"""
sisepuede/calibration/run_energy_calibration.py

Thin CLI wrapper around ``energy_calibration.energy_calibration``.

Example
-------
    python -m sisepuede.calibration.run_energy_calibration \\
        --country PER --target-year 2018 --num-iterations 2 --cal-option 3

All paths and parameters can be overridden on the command line; defaults are
the development paths used in ``energy_calibration.ipynb``.
"""

from __future__ import annotations

import argparse
import sys

_REPO_ROOT = "/Users/dianamendez/feature-energy-calibration"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sisepuede.calibration.energy_calibration import energy_calibration


# ── Default paths (override on the command line if needed) ────────────────────
DEFAULT_SISEPUEDE_INPUT = "/Users/dianamendez/sisepuede-data/input_data_peru_base.csv"
DEFAULT_IEA_DATA_DIR    = "/Users/dianamendez/data_collection_temporary"
DEFAULT_CROSSWALK_FILE  = (
    "/Users/dianamendez/feature-energy-calibration/"
    "sisepuede/ref/data_crosswalks/sisepuede_iea_energy_crosswalk.csv"
)
DEFAULT_OUTPUT_DIR      = "/Users/dianamendez/sisepuede-data"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the SISEPUEDE energy calibration pipeline.",
    )

    # Calibration target
    p.add_argument("--country",        type=str, default="PER",
                   help="ISO-3 country code (default: PER).")
    p.add_argument("--target-year",    type=int, default=2018,
                   help="Calendar year to calibrate to (default: 2018).")
    p.add_argument("--num-iterations", type=int, default=2,
                   help="Number of Phase 1 + Phase 2 iterations (default: 2).")
    p.add_argument("--cal-option",     type=int, default=3, choices=[0, 1, 2, 3, 4],
                   help="Calibration option, 0..4. See Calibrator.calibrate (default: 3).")
    p.add_argument("--gamma",          type=float, default=100.0,
                   help="QP regularisation weight, used when cal-option in {3,4} (default: 100.0).")
    p.add_argument("--enforce-varspec-bounds", action="store_true",
                   help="Enforce VariableSpec bounds in the QP (cal-option in {3,4}).")

    # Time window
    p.add_argument("--start-year",     type=int, default=2015,
                   help="First year of the comparison window (default: 2015).")
    p.add_argument("--end-year",       type=int, default=2022,
                   help="Last year of the comparison window (default: 2022).")
    p.add_argument("--iea-year-limit", type=int, default=2022,
                   help="Drop input rows beyond this year before calibrating (default: 2022).")

    # Paths
    p.add_argument("--sisepuede-input", type=str, default=DEFAULT_SISEPUEDE_INPUT,
                   help="Path to the SISEPUEDE input CSV.")
    p.add_argument("--iea-data-dir",    type=str, default=DEFAULT_IEA_DATA_DIR,
                   help="Directory holding the IEA energy balance files.")
    p.add_argument("--crosswalk-file",  type=str, default=DEFAULT_CROSSWALK_FILE,
                   help="Path to the IEA↔SISEPUEDE crosswalk CSV.")
    p.add_argument("--output-dir",      type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Directory where outputs (plots/, tables/, CSVs) are written.")

    # Output knobs
    p.add_argument("--tag", type=str, default="",
                   help="Optional suffix appended to output filenames.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress prints (verbose=False).")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    energy_calibration(
        iso_country            = args.country,
        target_year            = args.target_year,
        num_iterations         = args.num_iterations,
        cal_option             = args.cal_option,
        start_year             = args.start_year,
        end_year               = args.end_year,
        iea_year_limit         = args.iea_year_limit,
        path_sisepuede_input   = args.sisepuede_input,
        path_iea_data_dir      = args.iea_data_dir,
        path_crosswalk_file    = args.crosswalk_file,
        path_output_dir        = args.output_dir,
        tag                    = args.tag,
        gamma                  = args.gamma,
        enforce_varspec_bounds = args.enforce_varspec_bounds,
        verbose                = not args.quiet,
    )


if __name__ == "__main__":
    main()
