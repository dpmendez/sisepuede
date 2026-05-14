
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import *




##########################################
###                                    ###
###         CALIBRATION PLOTS          ###
###                                    ###
##########################################

# # TO DO:
# # * Generalize functions to be dataframe agnostic


# Primary calibration targets (Phase 1: sector totals)
PRIMARY_TARGETS = [
    ("TRANSPORT", "TRANSPORT"),
    ("RESIDENT",  "RESIDENT"),
    ("COMMPUB",   "COMMPUB"),
    ("AGRICULT",  "AGRICULT"),
    ("INDUSTRY",  "INDUSTRY"),
]


def _maybe_save(fig, savepath: Optional[str]) -> None:
    """Save `fig` to `savepath` (path + filename, e.g. 'plots/out.png') if
    provided. No-op when `savepath` is None or empty."""
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)


def _resolve_pairs(
    df_comparison: pd.DataFrame,
    mode: str,
    sector: Optional[str],
    pairs: Optional[List[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    """Build the list of (balance, product) pairs to plot for a given mode."""
    if pairs is not None:
        return list(pairs)
    if mode == "primary":
        return list(PRIMARY_TARGETS)
    if mode == "fuel_mix":
        if not sector:
            raise ValueError("sector is required when mode='fuel_mix'")
        mask = df_comparison["iea_balance_code"] == sector
        fuels = (
            df_comparison.loc[mask, "iea_product_code"]
            .dropna().unique().tolist()
        )
        return [(sector, f) for f in fuels if f != sector]
    raise ValueError("mode must be 'primary' or 'fuel_mix'")


def _plot_before_after_single_pair(
    df_comp_baseline: pd.DataFrame,
    df_comp_calibrated: pd.DataFrame,
    bal: str,
    prod: str,
    ax_value,
    ax_ratio = None,
    ax_rel_err = None,
    year_target: Optional[int] = None,
) -> None:
    """Render one column of the before/after time-series for a single
    (balance, product) pair. The consumption panel (`ax_value`) is always
    drawn; the ratio and relative-error panels are drawn only when their
    axes are provided.
    """
    mask_b = (
        (df_comp_baseline["iea_balance_code"] == bal)
        & (df_comp_baseline["iea_product_code"] == prod)
    )
    mask_c = (
        (df_comp_calibrated["iea_balance_code"] == bal)
        & (df_comp_calibrated["iea_product_code"] == prod)
    )

    # Top panel: consumption (IEA, Before, After)
    d_iea  = df_comp_baseline[mask_b].dropna(subset=["value_iea_tj"])
    d_base = df_comp_baseline[mask_b].dropna(subset=["value_sisepuede_tj"])
    d_cal  = df_comp_calibrated[mask_c].dropna(subset=["value_sisepuede_tj"])

    ax_value.plot(d_iea["year"],  d_iea["value_iea_tj"]        / 1e3, "b-o",  lw=2,   ms=4, label="IEA")
    ax_value.plot(d_base["year"], d_base["value_sisepuede_tj"] / 1e3, "r--s", lw=1.5, ms=4, label="Before")
    ax_value.plot(d_cal["year"],  d_cal["value_sisepuede_tj"]  / 1e3, "g--^", lw=1.5, ms=4, label="After")
    ax_value.set_title(f"{bal}x{prod}", fontsize=9)
    ax_value.set_ylabel("PJ", fontsize=8)

    # Optional diagnostic panels — same Before/After colors for visual linkage
    diag_panels = [
        (ax_ratio,   "ratio",   "ratio",   1.0),
        (ax_rel_err, "rel_err", "rel_err", 0.0),
    ]
    for ax, col, ylabel, ref in diag_panels:
        if ax is None:
            continue
        d_b = df_comp_baseline[mask_b].dropna(subset=[col])
        d_c = df_comp_calibrated[mask_c].dropna(subset=[col])
        ax.plot(d_b["year"], d_b[col], "r--s", lw=1.5, ms=4, label="Before")
        ax.plot(d_c["year"], d_c[col], "g--^", lw=1.5, ms=4, label="After")
        ax.axhline(ref, color="black", linewidth=1.0, linestyle="--")
        ax.set_ylabel(ylabel, fontsize=8)

    # x-axis decoration: vline at year_target on every panel; xlabel only on
    # the bottom-most panel
    all_axes = [a for a in (ax_value, ax_ratio, ax_rel_err) if a is not None]
    for ax in all_axes:
        if year_target is not None:
            ax.axvline(year_target, color="grey", linestyle=":", linewidth=1)
        ax.tick_params(axis="both", labelsize=7)
    all_axes[-1].set_xlabel("year", fontsize=8)


def plot_before_after_time_series(
    df_comp_baseline: pd.DataFrame,
    df_comp_calibrated: pd.DataFrame,
    year_target: int,
    mode: str = "primary",
    sector: Optional[str] = None,
    pairs: Optional[List[Tuple[str, str]]] = None,
    country: str = "",
    with_diagnostics: bool = False,
    savepath: Optional[str] = None,
) -> None:
    """
    Time-series of IEA vs SISEPUEDE (before / after calibration), one column
    per (balance, product) pair.

    Parameters
    ----------
    df_comp_baseline, df_comp_calibrated : pd.DataFrame
        Comparison tables before and after calibration.
    year_target : int
        Calibration target year. Drawn as a dotted vertical reference line.
    mode : str
        - 'primary'  : plot the primary sector totals (PRIMARY_TARGETS).
        - 'fuel_mix' : plot every fuel within `sector`.
    sector : str, optional
        Required when mode='fuel_mix'. Sector code (e.g., 'RESIDENT').
    pairs : list of (balance, product), optional
        Explicit override of the pairs to plot.
    country : str
        Country label for the figure title.
    with_diagnostics : bool
        If True, add two bottom rows showing ratio and relative-error
        before/after calibration. Ratio reference at 1.0; rel_err at 0.0.
        Diagnostic lines reuse the Before (red) / After (green) colours of
        the consumption panel.
    """
    pairs = _resolve_pairs(df_comp_baseline, mode, sector, pairs)
    if not pairs:
        print("No (balance, product) pairs to plot.")
        return

    n = len(pairs)
    nrows = 3 if with_diagnostics else 1
    fig_height = 8 if with_diagnostics else 4
    gridspec_kw = {"height_ratios": [2, 1, 1]} if with_diagnostics else None
    fig, axes = plt.subplots(
        nrows, n,
        figsize=(4 * n, fig_height),
        sharey=False,
        squeeze=False,
        gridspec_kw=gridspec_kw,
    )

    for i, (bal, prod) in enumerate(pairs):
        ax_value   = axes[0, i]
        ax_ratio   = axes[1, i] if with_diagnostics else None
        ax_rel_err = axes[2, i] if with_diagnostics else None
        _plot_before_after_single_pair(
            df_comp_baseline, df_comp_calibrated,
            bal, prod,
            ax_value, ax_ratio, ax_rel_err,
            year_target=year_target,
        )

    axes[0, 0].legend(fontsize=7)
    if with_diagnostics:
        axes[1, 0].legend(fontsize=7)

    scope = f"{sector} fuel mix" if mode == "fuel_mix" else "primary sector totals"
    fig.suptitle(
        f"{country} — {scope}: IEA vs SISEPUEDE before/after calibration\n"
        f"(dotted line = calibration target year {year_target})",
        fontsize=10,
    )
    fig.tight_layout()
    _maybe_save(fig, savepath)
    plt.show()


def plot_before_after_bar(
    df_comp_baseline: pd.DataFrame,
    df_comp_calibrated: pd.DataFrame,
    year_target: int,
    mode: str = "primary",
    sector: Optional[str] = None,
    pairs: Optional[List[Tuple[str, str]]] = None,
    country: str = "",
    savepath: Optional[str] = None,
) -> None:
    """
    Grouped bar chart of IEA vs SISEPUEDE (before / after calibration) at
    `year_target`.

    - mode='primary'  : absolute values in PJ for each primary sector total.
    - mode='fuel_mix' : fuel SHARES within `sector` (each value divided by
                        the matching sector total at year_target).

    Parameters
    ----------
    df_comp_baseline, df_comp_calibrated : pd.DataFrame
    year_target : int
    mode : str
        'primary' or 'fuel_mix'.
    sector : str, optional
        Required when mode='fuel_mix'.
    pairs : list of (balance, product), optional
        Explicit override of the pairs to plot.
    country : str
    """
    pairs = _resolve_pairs(df_comp_baseline, mode, sector, pairs)
    if not pairs:
        print("No (balance, product) pairs to plot.")
        return

    def _val(df_comp, bal, prod, col):
        mask = (
            (df_comp["iea_balance_code"] == bal)
            & (df_comp["iea_product_code"] == prod)
            & (df_comp["year"] == year_target)
        )
        row = df_comp[mask]
        return row[col].iloc[0] if not row.empty else np.nan

    iea_vals    = [_val(df_comp_baseline,   b, p, "value_iea_tj")       for b, p in pairs]
    before_vals = [_val(df_comp_baseline,   b, p, "value_sisepuede_tj") for b, p in pairs]
    after_vals  = [_val(df_comp_calibrated, b, p, "value_sisepuede_tj") for b, p in pairs]

    if mode == "fuel_mix":
        iea_total    = _val(df_comp_baseline,   sector, sector, "value_iea_tj")
        before_total = _val(df_comp_baseline,   sector, sector, "value_sisepuede_tj")
        after_total  = _val(df_comp_calibrated, sector, sector, "value_sisepuede_tj")
        iea_vals    = [v / iea_total    if iea_total    else np.nan for v in iea_vals]
        before_vals = [v / before_total if before_total else np.nan for v in before_vals]
        after_vals  = [v / after_total  if after_total  else np.nan for v in after_vals]
        ylabel = f"share of {sector} total"
        title  = f"{sector} fuel mix — {country}, {year_target}"
        labels = [p for _, p in pairs]
        rotation = 0
    else:
        # PJ for readability
        iea_vals    = [v / 1e3 if pd.notna(v) else v for v in iea_vals]
        before_vals = [v / 1e3 if pd.notna(v) else v for v in before_vals]
        after_vals  = [v / 1e3 if pd.notna(v) else v for v in after_vals]
        ylabel = "PJ"
        title  = f"Primary sector totals — {country}, {year_target}"
        labels = [f"{b}x{p}" for b, p in pairs]
        rotation = 30

    x = np.arange(len(pairs))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(pairs)), 4))
    ax.bar(x - w, iea_vals,    width=w, label="IEA",    color="#1f77b4")
    ax.bar(x,     before_vals, width=w, label="Before", color="#aec7e8")
    ax.bar(x + w, after_vals,  width=w, label="After",  color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation, ha="center" if rotation == 0 else "right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, savepath)
    plt.show()


def plot_before_after_discrepancy_bar(
    df_comp_baseline: pd.DataFrame,
    df_comp_calibrated: pd.DataFrame,
    year_target: int,
    variable: str = "ratio",
    country: str = "",
    savepath: Optional[str] = None,
) -> None:
    """
    Side-by-side bar chart comparing the SISEPUEDE-vs-IEA discrepancy at
    `year_target` before and after calibration. After-bars are coloured by
    whether calibration moved the value closer to the reference (green) or
    away from it (red).

    Parameters
    ----------
    df_comp_baseline, df_comp_calibrated : pd.DataFrame
    year_target : int
    variable : str
        - 'ratio'     -> uses 'ratio'   (reference line at 1.0)
        - 'rel_error' -> uses 'rel_err' (reference line at 0.0)
    country : str
        Country label for the title.
    """
    if variable == "ratio":
        col = "ratio"
        ref = 1.0
        ylabel = "ratio  SISEPUEDE / IEA"
    elif variable == "rel_error":
        col = "rel_err"
        ref = 0.0
        ylabel = "relative error  (SSP − IEA) / IEA"
    else:
        raise ValueError("variable must be 'ratio' or 'rel_error'")

    def _at_year(df_comp):
        return (
            df_comp[df_comp["year"] == year_target]
            .groupby(["iea_balance_code", "iea_product_code"])
            [col]
            .mean()
        )

    df_plot = pd.concat(
        [_at_year(df_comp_baseline).rename("before"),
         _at_year(df_comp_calibrated).rename("after")],
        axis=1,
    ).dropna(subset=["before", "after"])

    if df_plot.empty:
        print(f"No data to plot for year {year_target} (variable={variable}).")
        return

    df_plot["improved"] = (df_plot["after"] - ref).abs() < (df_plot["before"] - ref).abs()
    df_plot = df_plot.sort_values("before", ascending=False)
    df_plot["label"] = [f"{b}x{p}" for b, p in df_plot.index]

    x = np.arange(len(df_plot))
    w = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w / 2, df_plot["before"], width=w, label="Before",
           color="#aec7e8", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, df_plot["after"], width=w, label="After",
           color=["#2ca02c" if v else "#d62728" for v in df_plot["improved"]],
           edgecolor="white", linewidth=0.5)
    ax.axhline(ref, color="black", linewidth=1.2, linestyle="--", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{country} — calibration target year {year_target}\n"
        "After bars: green = improved, red = worse"
    )
    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, savepath)
    plt.show()


def plot_baseline_discrepancy_bar(
    df_comparison: pd.DataFrame,
    year_target: int,
    variable: str = "ratio",
    country: str = "",
    savepath: Optional[str] = None,
) -> None:
    """
    Bar chart of per-(balance, product) discrepancy between SISEPUEDE and IEA
    for a single year.

    Parameters
    ----------
    df_comparison : pd.DataFrame
        Comparison table with columns 'year', 'iea_balance_code',
        'iea_product_code', 'value_iea_tj', and the metric column.
    year_target : int
        Year to filter on.
    variable : str
        Which metric to plot:
        - 'ratio'     -> 'ratio'   (reference line at 1.0)
        - 'rel_error' -> 'rel_err' (reference line at 0.0)
    country : str
        Optional country label included in the plot title.
    """
    if variable == "ratio":
        col = "ratio"
        ref = 1.0
        ylabel = "ratio  SISEPUEDE / IEA"
        # >1.15 over-estimates (red), <0.85 under-estimates (green), else blue
        def color_of(v):
            if v > 1.15:
                return "#d62728"
            if v < 0.85:
                return "#2ca02c"
            return "#1f77b4"
        ref_label = "perfect match"
    elif variable == "rel_error":
        col = "rel_err"
        ref = 0.0
        ylabel = "relative error  (IEA − SSP) / IEA"
        # |err| > 0.15 = red, else blue
        def color_of(v):
            if abs(v) > 0.15:
                return "#d62728"
            return "#1f77b4"
        ref_label = "zero error"
    else:
        raise ValueError("variable must be 'ratio' or 'rel_error'")

    df_year = df_comparison[df_comparison["year"] == year_target].copy()
    df_year = df_year.dropna(subset=[col, "value_iea_tj"])
    df_year = df_year[df_year["value_iea_tj"] > 0]

    if df_year.empty:
        print(f"No data to plot for year {year_target} (variable={variable}).")
        return

    df_year["label"] = df_year["iea_balance_code"] + "x" + df_year["iea_product_code"]
    df_year = df_year.sort_values(col, ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [color_of(v) for v in df_year[col]]
    ax.bar(range(len(df_year)), df_year[col], color=colors)
    ax.axhline(ref, color="black", linewidth=1.2, linestyle="--", label=ref_label)
    ax.set_xticks(range(len(df_year)))
    ax.set_xticklabels(df_year["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    title = f"Baseline: SISEPUEDE vs IEA — {country}, year {year_target}" if country \
            else f"Baseline: SISEPUEDE vs IEA — year {year_target}"
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, savepath)
    plt.show()


def _plot_detailed_single(
    df_pair: pd.DataFrame,
    ax1, ax2, ax3,
    second_var: str,
    third_var: str,
    second_label: str,
    third_label: str,
):
    df = df_pair.copy()

    # Melt for seaborn
    df_melt = df.melt(id_vars=['year'], value_vars=['value_iea_tj', 'value_sisepuede_tj'],
                      var_name='source', value_name='value')
    df_melt['source'] = df_melt['source'].map({'value_iea_tj': 'IEA (observed)', 'value_sisepuede_tj': 'SISEPUEDE'})

    sns.lineplot(data=df_melt, x='year', y='value', hue='source', style='source',
                 markers=['o', 's'], ax=ax1, palette=['steelblue', 'tomato'])
    ax1.set_ylabel('TJ')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    sns.lineplot(data=df, x='year', y=second_var, ax=ax2, color='green', marker='x')
    ax2.set_ylabel(second_label)
    ax2.grid(True, alpha=0.3)

    sns.lineplot(data=df, x='year', y=third_var, ax=ax3, color='purple', marker='^')
    ax3.set_ylabel(third_label)
    ax3.set_xlabel('Year')
    ax3.grid(True, alpha=0.3)


def plot_observation_comparisons(
    df_comparison: pd.DataFrame,
    pairs: Union[Tuple[str, str], List[Tuple[str, str]]] = None,
    country: str = '',
    max_pairs: int = 12,
    with_diagnostics: bool = False,
    second_var: str = 'ratio',
    third_var: str = 'rel_err',
    second_label: str = 'Ratio SSP/IEA',
    third_label: str = 'Rel Error IEA',
    savepath: Optional[str] = None,
) -> None:
    """
    Plot SISEPUEDE vs IEA time series for one or more (balance, product) pairs.

    Layout
    ------
    - `with_diagnostics=False` (default): one consumption panel per pair.
        * Single pair    -> one tall axis (8 × 6).
        * Multiple pairs -> wrapping grid (≤ 3 columns).
    - `with_diagnostics=True`: three stacked panels per pair (consumption,
      `second_var`, `third_var`). Diagnostic curves use green / purple
      colours; they are not before/after comparisons.
        * Single pair    -> 3 rows × 1 col with height ratios (2, 1, 1).
        * Multiple pairs -> 3 rows × N cols.

    Pair identifiers can be IEA code (e.g., 'INDUSTRY') or name
    (e.g., 'Industry'). Pairs with no matching rows are skipped.

    Parameters
    ----------
    df_comparison : pd.DataFrame
    pairs : tuple, list of tuples, or None
        - tuple of length 2: a single pair.
        - list of tuples: multiple pairs.
        - None: inferred from `df_comparison` (only valid when the frame
          contains exactly one pair).
    country : str
    max_pairs : int
    with_diagnostics : bool
    second_var, third_var : str
        Column names plotted in the two diagnostic rows.
    second_label, third_label : str
        Y-axis labels for the diagnostic rows.
    """
    # ── Normalise `pairs` ────────────────────────────────────────────────
    if pairs is None:
        unique_pairs = df_comparison[['iea_balance_code', 'iea_product_code']].drop_duplicates()
        if len(unique_pairs) != 1:
            raise ValueError('pairs must be provided when df_comparison includes multiple balance/product combinations')
        pairs = [tuple(unique_pairs.iloc[0])]

    if isinstance(pairs, tuple) and len(pairs) == 2 and not isinstance(pairs[0], (list, tuple)):
        pairs = [pairs]

    if not isinstance(pairs, list):
        raise ValueError('pairs must be a tuple or a list of tuples')

    # ── Resolve metadata per pair ────────────────────────────────────────
    selected = []
    for balance_id, product_id in pairs[:max_pairs]:
        mask_balance = (df_comparison['iea_balance_code'] == balance_id) | (df_comparison['iea_balance_name'] == balance_id)
        mask_product = (df_comparison['iea_product_code'] == product_id) | (df_comparison['iea_product_name'] == product_id)
        df_pair = df_comparison[mask_balance & mask_product].sort_values('year')

        if df_pair.empty:
            print(f'No data found for pair: {balance_id}, {product_id}')
            continue

        selected.append({
            'data':         df_pair,
            'balance_code': df_pair['iea_balance_code'].iloc[0],
            'product_code': df_pair['iea_product_code'].iloc[0],
            'subsector':    df_pair['sisepuede_subsector'].iloc[0] if 'sisepuede_subsector' in df_pair.columns else 'Unknown',
        })

    if not selected:
        print("No valid pairs to plot.")
        return

    n = len(selected)

    def _title(pair, fontsize):
        return f"{pair['balance_code']} x {pair['product_code']} - {pair['subsector']}", fontsize

    # ── Render ───────────────────────────────────────────────────────────
    if with_diagnostics:
        if n == 1:
            fig, axes = plt.subplots(
                3, 1, sharex=True, figsize=(8, 10),
                gridspec_kw={'height_ratios': [2, 1, 1]},
            )
            ax_cols = [axes]
        else:
            fig, axes = plt.subplots(
                3, n, figsize=(4 * n, 8), sharex='col', squeeze=False,
                gridspec_kw={'height_ratios': [2, 1, 1]},
            )
            ax_cols = [axes[:, i] for i in range(n)]

        for col_axes, pair in zip(ax_cols, selected):
            ax1, ax2, ax3 = col_axes[0], col_axes[1], col_axes[2]
            _plot_detailed_single(
                pair['data'], ax1, ax2, ax3,
                second_var, third_var, second_label, third_label,
            )
            title, fs = _title(pair, 10 if n == 1 else 9)
            ax1.set_title(title, fontsize=fs)

        suptitle = f"Detailed Comparison{'s' if n > 1 else ''} — {country}"
        y_supt = 0.98

    else:
        if n == 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            axes_flat = [ax]
        else:
            ncols = min(3, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
            axes_flat = list(axes.flatten())

        for i, pair in enumerate(selected):
            ax = axes_flat[i]
            df_melt = pair['data'].melt(
                id_vars=['year'], value_vars=['value_iea_tj', 'value_sisepuede_tj'],
                var_name='source', value_name='value',
            )
            df_melt['source'] = df_melt['source'].map(
                {'value_iea_tj': 'IEA (observed)', 'value_sisepuede_tj': 'SISEPUEDE'}
            )
            sns.lineplot(
                data=df_melt, x='year', y='value', hue='source', style='source',
                markers=['o', 's'], ax=ax, palette=['steelblue', 'tomato'],
            )
            title, fs = _title(pair, 10 if n == 1 else 9)
            ax.set_title(title, fontsize=fs)
            ax.set_xlabel("Year")
            ax.set_ylabel("TJ")
            ax.legend(fontsize=8 if n == 1 else 7)
            ax.grid(True, alpha=0.3)

        # Hide unused axes when wrapping
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        suptitle = f"IEA vs SISEPUEDE — {country}"
        y_supt = 1.01

    fig.suptitle(suptitle, fontsize=12, y=y_supt)
    plt.tight_layout()
    _maybe_save(fig, savepath)
    plt.show()


def plot_metric_bar(
    df_comparison: pd.DataFrame,
    year: int,
    metric: str = 'difference',
    orientation: str = 'vertical',
    savepath: Optional[str] = None,
) -> None:
    """
    Create a bar plot showing a metric derived from the difference between SSP and IEA values
    for a specific year, only for (balance, product) pairs that exist in both datasets using Seaborn.

    Parameters:
    - df_comparison: DataFrame with comparison data, including 'year', 'value_sisepuede_tj', 'value_iea_tj',
                     'iea_balance_code', 'iea_product_code', etc.
    - year: The year to filter data for.
    - metric: The metric to compute ('difference', 'ratio', 'percentage').
    - orientation: 'vertical' or 'horizontal' for the bar plot.
    """
    # Filter for the specified year
    df_year = df_comparison[df_comparison['year'] == year].copy()

    # Filter for pairs with both SSP and IEA values
    df_both = df_year.dropna(subset=['value_sisepuede_tj', 'value_iea_tj'])

    if df_both.empty:
        print(f"No data for year {year} with both SSP and IEA values.")
        return

    # Compute the metric
    if metric == 'difference':
        df_both['metric_value'] = df_both['value_sisepuede_tj'] - df_both['value_iea_tj']
        ylabel = 'Difference (SSP - IEA) [TJ]'
    elif metric == 'ratio':
        df_both['metric_value'] = df_both['value_sisepuede_tj'] / df_both['value_iea_tj']
        ylabel = 'Ratio (SSP / IEA)'
    elif metric == 'error':
        df_both['metric_value'] = ((df_both['value_sisepuede_tj'] - df_both['value_iea_tj']) / df_both['value_iea_tj']) * 100
        ylabel = 'Relative Error (SSP-IEA) / IEA (%)'
    else:
        raise ValueError("Metric must be 'difference', 'ratio', or 'error'")

    # Sort by metric value in descending order
    df_both = df_both.sort_values('metric_value', ascending=False)

    # Create labels for the bars, including mapping quality when available
    if 'mapping_quality' in df_both.columns:
        df_both['mapping_quality'] = df_both['mapping_quality'].fillna('Unknown')
        df_both['label'] = (
            df_both['iea_balance_code'] + ' - ' + df_both['iea_product_code']
            + ' (' + df_both['mapping_quality'].astype(str) + ')'
        )
    else:
        df_both['label'] = df_both['iea_balance_code'] + ' - ' + df_both['iea_product_code']

    # Plot using seaborn
    fig, ax = plt.subplots(figsize=(12, 8))

    if orientation == 'vertical':
        sns.barplot(data=df_both, x='label', y='metric_value', ax=ax, palette='viridis')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Balance - Product Pair')
        plt.xticks(rotation=45, ha='right')
    else:
        sns.barplot(data=df_both, y='label', x='metric_value', ax=ax, palette='viridis', orient='h')
        ax.set_xlabel(ylabel)
        ax.set_ylabel('Balance - Product Pair')

    ax.set_title(f'{metric.capitalize()} between SSP and IEA for {year}')

    # Add value labels on the bars
    for p in ax.patches:
        if orientation == 'vertical':
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10)
        else:
            ax.annotate(f'{p.get_width():.2f}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=10)

    plt.tight_layout()
    _maybe_save(fig, savepath)
    plt.show()
