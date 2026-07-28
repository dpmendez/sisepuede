"""
sisepuede/calibration/_hyperparam_search.py

Automated hyperparameter search for the v3 surrogate.

Convention
----------
Selection happens on the **dev** split only. The **test** split stays
untouched so `apply_accuracy_gate(surrogate, test_report, spec)`
downstream is still a real generalisation check on data the search
never saw.

Usage example:

    from sisepuede.calibration._hyperparam_search import (
        grid_search_hyperparams,
    )
    from sisepuede.calibration._surrogate import SurrogateSpec

    base_spec = SurrogateSpec(model_kind="gbm", seed=42)

    param_grid = {
        "max_iter":      [100, 200, 400],
        "max_depth":     [3, 4, 6],
        "learning_rate": [0.05, 0.08, 0.15],
    }

    search = grid_search_hyperparams(
        X_train         = X_tr,
        Y_train         = Y_tr,
        X_dev           = X_dv,
        Y_dev           = Y_dv,
        base_spec       = base_spec,
        param_grid      = param_grid,
        envelope_bounds = envelope_bounds,
        consumption_fingerprint = metadata["consumption_fingerprint"],
        metric          = "mean_r2",
    )
    search.results_df.head()
    best_spec = search.best_spec        # ready to retrain the winner
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sisepuede.calibration._surrogate import (
    Surrogate,
    SurrogateReport,
    SurrogateSpec,
)


SUPPORTED_METRICS = ("mean_r2", "median_r2", "mean_mape", "median_mape")


##########################
#    RESULT CONTAINER    #
##########################

@dataclass
class HyperparamSearchResult:
    """Everything a caller needs to inspect and re-run the winning config.

    Attributes
    ----------
    results_df : pd.DataFrame
        One row per combination, sorted so the best config is row 0.
        Columns: the swept hyperparameter names, plus
        `mean_r2`, `median_r2`, `mean_mape`, `median_mape`,
        `n_valid_targets`, `fit_seconds`, `failed`.
    best_params : dict
        The winning hyperparameter combination (only the swept keys).
    best_spec : SurrogateSpec
        A new SurrogateSpec identical to `base_spec` but with
        `hyperparams = {**base_spec.hyperparams, **best_params}`.
        Retrain the surrogate on train + dev with this spec before
        the final test-set gate.
    best_score : float
    metric : str
        Which column of `results_df` was used to rank.
    """

    results_df:  pd.DataFrame
    best_params: Dict[str, Any]
    best_spec:   SurrogateSpec
    best_score:  float
    metric:      str


##########################
#    PUBLIC API          #
##########################

def grid_search_hyperparams(
    X_train:                 pd.DataFrame,
    Y_train:                 pd.DataFrame,
    X_dev:                   pd.DataFrame,
    Y_dev:                   pd.DataFrame,
    *,
    base_spec:               SurrogateSpec,
    param_grid:              Mapping[str, Sequence[Any]],
    envelope_bounds:         Tuple[np.ndarray, np.ndarray],
    consumption_fingerprint: str,
    metric:                  str = "mean_r2",
    on_progress:             Optional[Callable[[int, int, Dict[str, Any], Dict[str, float]], None]] = None,
) -> HyperparamSearchResult:
    """Exhaustive Cartesian sweep over `param_grid`.

    Each combination replaces the corresponding keys in
    `base_spec.hyperparams`, trains a fresh surrogate on
    (X_train, Y_train), and scores on (X_dev, Y_dev).

    Parameters
    ----------
    X_train, Y_train, X_dev, Y_dev : pd.DataFrame
        Aligned splits produced by `split_train_dev_test`.
    base_spec : SurrogateSpec
        The `model_kind`, `seed`, holdout / gate thresholds, and any
        `hyperparams` you want held constant. Only the keys present in
        `param_grid` get overridden per combination.
    param_grid : mapping of hyperparameter name -> iterable of values
        E.g. `{"max_iter": [100, 400], "learning_rate": [0.05, 0.1]}`.
    envelope_bounds : (lb_array, ub_array)
        Same shape / meaning as `Surrogate.train(envelope_bounds=...)`.
    consumption_fingerprint : str
        Passed through to every fitted surrogate.
    metric : {"mean_r2", "median_r2", "mean_mape", "median_mape"}
        Which aggregate to rank on. R^2 is maximised, MAPE is minimised.
    on_progress : callable | None
        Optional callback `(i, n_total, params, scores)` fired after
        each combination. Useful for tqdm.

    Returns
    -------
    HyperparamSearchResult
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"grid_search_hyperparams: unknown metric {metric!r}. "
            f"Supported: {SUPPORTED_METRICS}."
        )
    if not param_grid:
        raise ValueError("grid_search_hyperparams: `param_grid` must be non-empty.")

    keys = list(param_grid.keys())
    value_lists = [list(param_grid[k]) for k in keys]
    combinations = [
        dict(zip(keys, combo)) for combo in itertools.product(*value_lists)
    ]

    return _run_search(
        combinations           = combinations,
        X_train                = X_train,
        Y_train                = Y_train,
        X_dev                  = X_dev,
        Y_dev                  = Y_dev,
        base_spec              = base_spec,
        envelope_bounds        = envelope_bounds,
        consumption_fingerprint= consumption_fingerprint,
        metric                 = metric,
        swept_keys             = keys,
        on_progress            = on_progress,
    )


def random_search_hyperparams(
    X_train:                 pd.DataFrame,
    Y_train:                 pd.DataFrame,
    X_dev:                   pd.DataFrame,
    Y_dev:                   pd.DataFrame,
    *,
    base_spec:               SurrogateSpec,
    param_distributions:     Mapping[str, Sequence[Any]],
    n_iter:                  int,
    envelope_bounds:         Tuple[np.ndarray, np.ndarray],
    consumption_fingerprint: str,
    metric:                  str = "mean_r2",
    seed:                    int = 0,
    on_progress:             Optional[Callable[[int, int, Dict[str, Any], Dict[str, float]], None]] = None,
) -> HyperparamSearchResult:
    """Sample `n_iter` combinations uniformly from `param_distributions`.

    Each value list is treated as a discrete distribution; one value is
    drawn per key per iteration. Duplicate combinations are possible
    and are evaluated only once (deduplicated by key/value tuple).

    Prefer this over `grid_search_hyperparams` when the grid is too
    large to enumerate exhaustively — for a modest computation budget
    random search covers each dimension more thoroughly.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"random_search_hyperparams: unknown metric {metric!r}. "
            f"Supported: {SUPPORTED_METRICS}."
        )
    if not param_distributions:
        raise ValueError("random_search_hyperparams: `param_distributions` must be non-empty.")
    if n_iter <= 0:
        raise ValueError("random_search_hyperparams: `n_iter` must be positive.")

    rng  = np.random.default_rng(seed)
    keys = list(param_distributions.keys())
    seen: set = set()
    combinations: List[Dict[str, Any]] = []
    for _ in range(n_iter):
        combo = {
            k: param_distributions[k][int(rng.integers(0, len(param_distributions[k])))]
            for k in keys
        }
        key_tuple = tuple(combo[k] for k in keys)
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        combinations.append(combo)

    return _run_search(
        combinations           = combinations,
        X_train                = X_train,
        Y_train                = Y_train,
        X_dev                  = X_dev,
        Y_dev                  = Y_dev,
        base_spec              = base_spec,
        envelope_bounds        = envelope_bounds,
        consumption_fingerprint= consumption_fingerprint,
        metric                 = metric,
        swept_keys             = keys,
        on_progress            = on_progress,
    )


def score_dev_report(
    surrogate: Surrogate,
    report:    SurrogateReport,
) -> Dict[str, float]:
    """Aggregate per-target R^2 / MAPE into scalar model-selection metrics.

    Constant-output targets contribute nothing to the aggregate because
    their R^2 is NaN (variance = 0) and their MAPE is often NaN too;
    those targets auto-pass `apply_accuracy_gate` regardless. NaN MAPE
    values (all-zero true vectors) are also excluded from the MAPE
    aggregate.

    Returns
    -------
    dict with keys "mean_r2", "median_r2", "mean_mape", "median_mape",
    "n_valid_targets".
    """
    r2   = report.r2_per_target
    mape = report.mape_per_target

    non_constant = pd.Index([t for t in report.targets if not surrogate.is_constant_target(t)])
    r2_valid   = r2.reindex(non_constant).dropna()
    mape_valid = mape.reindex(non_constant).dropna()

    return {
        "mean_r2":         float(r2_valid.mean())   if len(r2_valid)   else float("nan"),
        "median_r2":       float(r2_valid.median()) if len(r2_valid)   else float("nan"),
        "mean_mape":       float(mape_valid.mean())   if len(mape_valid) else float("nan"),
        "median_mape":     float(mape_valid.median()) if len(mape_valid) else float("nan"),
        "n_valid_targets": int(len(r2_valid)),
    }


##########################
#    INTERNAL HELPERS    #
##########################

def _run_search(
    *,
    combinations:            List[Dict[str, Any]],
    X_train:                 pd.DataFrame,
    Y_train:                 pd.DataFrame,
    X_dev:                   pd.DataFrame,
    Y_dev:                   pd.DataFrame,
    base_spec:               SurrogateSpec,
    envelope_bounds:         Tuple[np.ndarray, np.ndarray],
    consumption_fingerprint: str,
    metric:                  str,
    swept_keys:              List[str],
    on_progress:             Optional[Callable[[int, int, Dict[str, Any], Dict[str, float]], None]],
) -> HyperparamSearchResult:
    if X_dev.shape[0] == 0:
        raise ValueError(
            "_run_search: dev split is empty. Hyperparameter selection needs "
            "a non-empty dev set — re-split with a larger `dev` fraction."
        )

    rows: List[Dict[str, Any]] = []
    n_total = len(combinations)
    for i, params in enumerate(combinations):
        merged = {**base_spec.hyperparams, **params}
        trial_spec = replace(base_spec, hyperparams=merged)

        t0 = time.time()
        scores: Dict[str, float] = {}
        failed = False
        try:
            surrogate = Surrogate.train(
                X                       = X_train,
                Y                       = Y_train,
                spec                    = trial_spec,
                consumption_fingerprint = consumption_fingerprint,
                envelope_bounds         = envelope_bounds,
            )
            report = surrogate.evaluate(X_dev, Y_dev)
            scores = score_dev_report(surrogate, report)
        except Exception as exc:                                 # pragma: no cover
            failed = True
            scores = {
                "mean_r2":         float("nan"),
                "median_r2":       float("nan"),
                "mean_mape":       float("nan"),
                "median_mape":     float("nan"),
                "n_valid_targets": 0,
                "error":           repr(exc),
            }
        fit_seconds = time.time() - t0

        row: Dict[str, Any] = {**params, **scores,
                               "fit_seconds": fit_seconds, "failed": failed}
        rows.append(row)

        if on_progress is not None:
            on_progress(i + 1, n_total, params, scores)

    results_df = pd.DataFrame(rows)
    results_df = _rank_by_metric(results_df, metric)

    best_row = results_df.iloc[0]
    best_params = {k: best_row[k] for k in swept_keys}
    best_spec = replace(
        base_spec,
        hyperparams={**base_spec.hyperparams, **best_params},
    )
    best_score = float(best_row[metric])

    return HyperparamSearchResult(
        results_df  = results_df.reset_index(drop=True),
        best_params = best_params,
        best_spec   = best_spec,
        best_score  = best_score,
        metric      = metric,
    )


def _rank_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Sort so row 0 is the best config for the chosen metric.

    R^2 metrics are maximised (descending); MAPE metrics are minimised
    (ascending). NaN scores sink to the bottom either way.
    """
    ascending = metric.endswith("mape")
    return df.sort_values(metric, ascending=ascending, na_position="last")
