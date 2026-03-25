
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import Figure
from matplotlib.axes import Axes
from typing import *

import sisepuede.utilities._toolbox as sf




##########################################
###                                    ###
###    BEGIN PLOT FUNCTIONS LIBRARY    ###
###                                    ###
##########################################

def is_valid_figtuple(
    figtuple: Any,
) -> bool:
    """
    Check if `figtuple` is a valid figure tuple of the form (fig, ax) for use
        in specifying plots. 
    """

    is_valid = isinstance(figtuple, tuple)
    is_valid &= (len(figtuple) >= 2) if is_valid else False
    is_valid &= (
        (isinstance(figtuple[0], Figure) & isinstance(figtuple[1], Axes)) 
        if is_valid 
        else False
    )

    return is_valid



def plot_stack(
    df: pd.DataFrame,
    fields: List[str],
    dict_formatting: Union[Dict[str, Dict[str, Any]], None] = None,
    field_x: Union[str, None] = None,
    figsize: Tuple = (18, 12),
    figtuple: Union[Tuple, None] = None,
    label_x: Union[str, None] = None,
    label_y: Union[str, None] = None,
    title: Union[str, None] = None,
    **kwargs,
) -> Union['matplotlib.Plot', None]:
    """plt.plot.area() cannot handle negative trajctories. Use plt.stackplot to 
        facilitate stacked area charts for trajectories that may shift between 
        positive and negative.
    
    Function Arguments
    ------------------
    df : pd.DataFrame
        DataFrame to pull data from
    fields : List[str]
        Fields to plot

    Keyword Arguments
    ----------------- 
    dict_formatting : Union[Dict[str, Dict[str, Any]], None]
        Optional dictionary used to pass field-specific formatting; e.g., use 
        
        dict_formatting = {
            field_i: {
                "kwarg_i": val_i,
                ...
            }
        }
        
        to pass formatting keywords for fields

    field_x : Union[str, None]
        Optional field `x` in data frame to use for x axis
    figsize : Tuple
        Figure size to use. Only used if `figtuple` is not a valid 
        (Figure, Axis) pair
    figtuple : Union[Tuple, None]
        Optional tuple of form `(fig, ax)` (result of plt.subplots) to pass. 
        Allows users to predefine information about the fig, ax outside of this 
        function, then plot within those confines
    label_x : Union[str, None]
        Optional label to pass for x axis. Only used if `figtuple` is not a 
        valid (Figure, Axis) pair
    label_y : Union[str, None]
        Optional label to pass for y axis. Only used if `figtuple` is not a 
        valid (Figure, Axis) pair
    title : Union[str, None]
        Optional title to pass. Only used if `figtuple` is not a valid 
        (Figure, Axis) pair
    **kwargs
        Passed to ax.stackplot, ax.set_xlabel, ax.set_ylabel, and ax.set_title
    """
    
    # check field x
    add_x = False
    field_x = field_x if field_x in df.columns else None
    if field_x is None:
        field_x = "x"
        add_x = True
    
    # check all fields
    fields = [x for x in df.columns if x in fields and (x != field_x)]
    if len(fields) == 0:
        return None
    
    if add_x:
        # if adding a dummy, copy first, then add field x
        df_plot = df[fields].copy()
        df_plot[field_x] = range(len(df_plot))
        fields.append(field_x)

    else:
        # if field_x is in the df, add it to fields, then copy
        fields.append(field_x)
        df_plot = df[fields].copy()


    # check the color dictionary
    if not isinstance(dict_formatting, dict):
        dict_formatting = {}
    
    
    ##  BUILD PLOTTED DATA
    
    # initialize ordered plot fields
    fields_plot = [x for x in fields if x != field_x]
    
    # split into positive and negative
    df_neg = pd.concat(
        [
            df_plot[[field_x]],
            df_plot[fields_plot].clip(upper = 0, )
        ], 
        axis = 1,
    )
    df_pos = pd.concat(
        [
            df_plot[[field_x]],
            df_plot[fields_plot].clip(lower = 0, )
        ], 
        axis = 1,
    )

    # reorder fields plot
    fields_with_negative_and_positive = []
    fields_with_negative = []
    fields_with_positive = []

    for x in fields_plot:
        has_pos = df_pos[x].max() > 0
        has_neg = df_neg[x].min() < 0

        if has_pos & has_neg:
            fields_with_negative_and_positive.append(x)
        elif has_neg:
            fields_with_negative.append(x)
        else:
            fields_with_positive.append(x)

    fields_plot = sorted(fields_with_negative_and_positive)
    fields_plot += sorted(fields_with_negative)
    fields_plot += sorted(fields_with_positive)
    
    
    # if colors are specified, specify an ordered vector that matches the fields
    if isinstance(dict_formatting, dict):
        try:
            # use try to avoid checking if "x" is specified
            color = [dict_formatting.get(x).get("color") for x in fields_plot]
            color = None if None in color else color

        except:
            color = None

    

    ##  GET kwargs FOR CALLING 

    dict_kwargs = dict((k, v) for (k, v) in kwargs.items())
    dict_kwargs.update(
        {
            "data": df_pos,
            "labels": fields_plot
        }
    )
    dict_kwargs.update({"colors": color, }) if color is not None else None



    ##  SPECIFY AND FORMAT AXES
    
    # check if the figtuple specification is ok
    accept_figtuple = is_valid_figtuple(figtuple, )
    
    if accept_figtuple:
        fig, ax = figtuple

    else:
        fig, ax = plt.subplots(1, 1, figsize = figsize, )

        # check label x
        if isinstance(label_x, str):
            sf.call_with_varkwargs(
                ax.set_xlabel,
                label_x,
                dict_kwargs = dict_kwargs,
            )
        
        # check label y
        if isinstance(label_y, str):
            sf.call_with_varkwargs(
                ax.set_xlabel,
                label_y,
                dict_kwargs = dict_kwargs,
            )
        
        # check title
        if isinstance(title, str):
            sf.call_with_varkwargs(
                ax.set_title,
                title,
                dict_kwargs = dict_kwargs,
            )

            
    
    ##  PLOT THE POSITIVE AND NEGATIVE COMPONENTS
    
    sf.call_with_varkwargs(
        ax.stackplot,
        field_x, 
        *fields_plot,
        dict_kwargs = dict_kwargs,
    )
    
    # update before plotting next one
    dict_kwargs.update({"data": df_neg, "labels": (), })
    sf.call_with_varkwargs(
        ax.stackplot,
        field_x, 
        *fields_plot,
        dict_kwargs = dict_kwargs,
    )
    
    return fig, ax
    
           

##########################################
###                                    ###
###         CALIBRATION PLOTS          ###
###                                    ###
##########################################

def plot_detailed_comparison(
    df_pair: pd.DataFrame,
    second_var: str,
    third_var: str,
    second_label: str,
    third_label: str,
    country: str = "",
) -> None:
    """
    Plot a detailed comparison for a single pair with three vertically stacked subplots.
    
    Top subplot (larger): IEA vs SISEPUEDE values.
    Middle subplot: Second variable (e.g., ratio or difference).
    Bottom subplot: Third variable (e.g., percentage difference).
    
    Calculates additional columns if needed (difference_sisepuede_iea, perc_difference_sisepuede).
    """
    df = df_pair.copy()
    
    # Calculate additional columns if not present
    if 'difference_sisepuede_iea' not in df.columns:
        df['difference_sisepuede_iea'] = df['value_sisepuede_tj'] - df['value_iea_tj']
    
    if 'perc_difference_sisepuede' not in df.columns:
        df['perc_difference_sisepuede'] = (
            (df['value_sisepuede_tj'] - df['value_iea_tj']) / df['value_iea_tj'] * 100
        ).fillna(0)  # Avoid NaN in division
    
    # Create subplots with shared x-axis and height ratios
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1, 1]}
    )
    
    # Top plot: IEA vs SISEPUEDE values
    ax1.plot(df['year'], df['value_iea_tj'],
             marker='o', label='IEA (observed)', color='steelblue')
    ax1.plot(df['year'], df['value_sisepuede_tj'],
             marker='s', linestyle='--', label='SISEPUEDE', color='tomato')
    ax1.set_ylabel('TJ')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Middle plot: Second variable
    ax2.plot(df['year'], df[second_var], color='green', marker='x')
    ax2.set_ylabel(second_label)
    ax2.grid(True, alpha=0.3)
    
    # Bottom plot: Third variable
    ax3.plot(df['year'], df[third_var], color='purple', marker='^')
    ax3.set_ylabel(third_label)
    ax3.set_xlabel('Year')
    ax3.grid(True, alpha=0.3)
    
    # Title with pair info
    balance_code = df['iea_balance_code'].iloc[0]
    balance_name = df['iea_balance_name'].iloc[0]
    product_code = df['iea_product_code'].iloc[0]
    product_name = df['iea_product_name'].iloc[0]
    subsector = df['sisepuede_subsector'].iloc[0] if 'sisepuede_subsector' in df.columns else 'Unknown'
    
    title = f"{balance_code} ({balance_name}) × {product_code} ({product_name}) - {subsector}"
    ax1.set_title(title, fontsize=10)
    fig.suptitle(f"Detailed Comparison — {country}", fontsize=12, y=0.98)
    
    plt.tight_layout()
    plt.show()



def plot_selected_comparisons(
    df_comparison: pd.DataFrame,
    pairs: list,
    country: str = "",
    max_panels: int = 12,
) -> None:
    """
    Plot SISEPUEDE vs IEA time series for selected pairs of balance and product identifiers.
    
    Each pair is a tuple (balance_identifier, product_identifier), where identifier can be
    either the code (e.g., 'INDUSTRY') or the name (e.g., 'Industry').
    
    Infers sisepuede_subsector from the data and displays it in the plot title.
    """
    df = df_comparison.copy()
    
    # Filter to pairs where we have at least one IEA observation
    df = df[df["value_iea_tj"].notna()]
    
    selected_pairs = []
    for balance_id, product_id in pairs[:max_panels]:
        # Find matching rows
        mask_balance = (df["iea_balance_code"] == balance_id) | (df["iea_balance_name"] == balance_id)
        mask_product = (df["iea_product_code"] == product_id) | (df["iea_product_name"] == product_id)
        mask = mask_balance & mask_product
        
        sub = df[mask]
        if sub.empty:
            print(f"No data found for pair: {balance_id}, {product_id}")
            continue
        
        # Get unique values
        balance_code = sub["iea_balance_code"].unique()[0]
        balance_name = sub["iea_balance_name"].unique()[0]
        product_code = sub["iea_product_code"].unique()[0]
        product_name = sub["iea_product_name"].unique()[0]
        subsector = sub["sisepuede_subsector"].unique()[0] if "sisepuede_subsector" in sub.columns else "Unknown"
        
        selected_pairs.append({
            "balance_code": balance_code,
            "balance_name": balance_name,
            "product_code": product_code,
            "product_name": product_name,
            "subsector": subsector,
            "data": sub.sort_values("year")
        })
    
    if not selected_pairs:
        print("No valid pairs to plot.")
        return
    
    if len(selected_pairs) == 1:
        # Single plot
        pair = selected_pairs[0]
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(pair["data"]["year"], pair["data"]["value_iea_tj"],
                marker="o", label="IEA (observed)", color="steelblue")
        ax.plot(pair["data"]["year"], pair["data"]["value_sisepuede_tj"],
                marker="s", linestyle="--", label="SISEPUEDE", color="tomato")
        
        title = (f"{pair['balance_code']} ({pair['balance_name']}) × "
                 f"{pair['product_code']} ({pair['product_name']}) - {pair['subsector']}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("TJ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"IEA vs SISEPUEDE — {country}", fontsize=12, y=1.01)
        plt.tight_layout()
        plt.show()
    else:
        # Multiple panels
        ncols = min(3, len(selected_pairs))
        nrows = int(np.ceil(len(selected_pairs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        
        for i, pair in enumerate(selected_pairs):
            ax = axes_flat[i]
            
            ax.plot(pair["data"]["year"], pair["data"]["value_iea_tj"],
                    marker="o", label="IEA (observed)", color="steelblue")
            ax.plot(pair["data"]["year"], pair["data"]["value_sisepuede_tj"],
                    marker="s", linestyle="--", label="SISEPUEDE", color="tomato")
            
            title = (f"{pair['balance_code']} ({pair['balance_name']}) × "
                     f"{pair['product_code']} ({pair['product_name']}) - {pair['subsector']}")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Year")
            ax.set_ylabel("TJ")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        
        plt.suptitle(f"IEA vs SISEPUEDE — {country}", fontsize=12, y=1.01)
        plt.tight_layout()
        plt.show()



