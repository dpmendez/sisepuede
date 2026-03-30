"""
sisepuede/calibration/iea_crosswalk.py

IEACrosswalk
------------
Load and apply the IEA <-> SISEPUEDE energy crosswalk. Given a SISEPUEDE output
dataframe and an IEA World Energy Balances dataframe, produces a paired
comparison table (one row per balance x product x year) in TJ.

Typical usage
-------------
    from sisepuede.manager.sisepuede_file_structure import SISEPUEDEFileStructure
    from sisepuede.calibration.iea_crosswalk import IEACrosswalk

    file_struct      = SISEPUEDEFileStructure()
    model_attributes = file_struct.model_attributes

    loader  = IEADataLoader("/path/to/data_collection_temporary", model_attributes)
    xw      = IEACrosswalk(model_attributes)

    df_ssp  = xw.aggregate_sisepuede(df_out_energycon, col_year = "year")
    df_iea  = loader.load_country("LBY")
    df_comp = xw.build_comparison(df_ssp, df_iea)
    df_summ = xw.summary(df_comp)
"""

import numpy as np
import os
import pandas as pd
import sisepuede.core.support_classes as sc
from typing import *



####################
#    CONSTANTS     #
####################

_CROSSWALK_REL_PATH = os.path.join(
    "data_crosswalks",
    "sisepuede_iea_energy_crosswalk.csv",
)

_VAR_SEP = ":"          # separator between variable names inside the CSV cell
_GWH_TO_TJ  = 3.6      # 1 GWh  = 3.6   TJ
_KTOE_TO_TJ = 41.868   # 1 ktoe = 41.868 TJ



####################
#    PRIMARY CLASS #
####################

class IEACrosswalk:
    """
    Load and apply the IEA <-> SISEPUEDE energy crosswalk.

    The crosswalk CSV maps each IEA (balance, product) pair to one or more
    SISEPUEDE output column names.  Three public methods convert that mapping
    into a comparison table:

        aggregate_sisepuede()  — apply the crosswalk to a SISEPUEDE output frame
        build_comparison()     — outer-join both into a paired comparison table
        summary()              — one-row-per-(balance, product) diagnostic
    """

    def __init__(self,
        model_attributes,
        path_crosswalk: Union[str, None] = None,
    ) -> None:
        """
        Function Arguments
        ------------------
        model_attributes : ModelAttributes
            SISEPUEDE model attributes object.

        Keyword Arguments
        -----------------
        path_crosswalk : Union[str, None]
            Full path to the crosswalk CSV. If None, located automatically
            inside model_attributes.dir_ref.
        """

        self.model_attributes = model_attributes

        self._initialize_regions()
        self._initialize_crosswalk(path_crosswalk)

        return None


    ##################################
    #    INITIALISATION METHODS      #
    ##################################

    def _initialize_regions(self,
    ) -> None:
        """
        Set self.regions (SISEPUEDERegions) and cache the IEA field name
        constants defined in regions._initialize_defaults_iea().
        """

        self.regions = sc.Regions(self.model_attributes)

        self.field_balance = self.regions.field_iea_balance   # "Balance"
        self.field_country = self.regions.field_iea_country   # "Country"
        self.field_product = self.regions.field_iea_product   # "Product"
        self.field_time    = self.regions.field_iea_time      # "Time"
        self.field_unit    = self.regions.field_iea_unit      # "Unit"
        self.field_value   = self.regions.field_iea_value     # "Value"

        return None


    def _initialize_crosswalk(self,
        path_crosswalk: Union[str, None],
    ) -> None:
        """
        Resolve the crosswalk path, load the CSV, and clean it.
        Sets self.df_crosswalk and self._path_crosswalk.

        Keyword Arguments
        -----------------
        path_crosswalk : Union[str, None]
            Explicit path; if None, resolved from model_attributes.dir_ref.
        """

        self._path_crosswalk = self._resolve_crosswalk_path(path_crosswalk)
        self.df_crosswalk = self._load_crosswalk(self._path_crosswalk)

        return None
    
    ##################################
    #    INTERNAL HELPERS            #
    ##################################

    def _resolve_crosswalk_path(self,
        path_crosswalk: Union[str, None],
    ) -> str:
        """Return the full path to the crosswalk CSV.

        Function Arguments
        ------------------
        path_crosswalk : Union[str, None]
            Explicit path. If None, looks in model_attributes.dir_ref.
        """

        if path_crosswalk is not None:
            return path_crosswalk
        
        dir_ref = getattr(self.model_attributes, "dir_ref", None)
        if dir_ref is None:
            raise FileNotFoundError(
                "Could not locate dir_ref from model_attributes. "
                "Pass path_crosswalk explicitly."
            )
        
        path = os.path.join(dir_ref, _CROSSWALK_REL_PATH)
        if not os.path.isfile(path):
                        raise FileNotFoundError(
                f"Crosswalk file not found at: {path}\n"
                "Run IEACrosswalkBuilder(model_attributes).build(write_csv=True) "
                "to generate it, or pass path_crosswalk explicitly."
            )

        return path
    

    def _load_crosswalk(self,
        path: str,
    ) -> pd.DataFrame:
        """Read and clean the crosswalk CSV.

        Drops blank separator rows (section dividers) and no_match rows.
        Ensures unit_conversion_to_tj is numeric.

        Function Arguments
        ------------------
        path : str
            Path to the crosswalk CSV.
        """

        df = pd.read_csv(path)

        ## CLEAN

        # drop blank separator rows used as section dividers
        df = df.dropna(subset= ["iea_balance_code", "sisepuede_output_variables"])

        # drop rows with no SISEPUEDE counterpart
        df = df[df["mapping_quality"] != "no_match"]

        # ensure conversion factor is numeric; default to 1 if missing
        df["unit_conversion_to_tj"] = pd.to_numeric(
             df["unit_conversion_to_tj"],
             errors="coerce",
        ).fillna(1.0)

        return df.reset_index(drop=True)
    
    @staticmethod
    def _to_tj(value: float,
        unit: str,
    ) -> float:
        """Convert a single IEA value to TJ.

        Function Arguments
        ------------------
        value : float
            Raw IEA value.
        unit : str
            IEA unit string (e.g. "TJ", "ktoe").
        """

        if pd.isna(value):
             return np.nan
        
        unit_lc = str(unit).strip().lower()
        if "gwh" in unit_lc:
             return float(value) * _GWH_TO_TJ
        if "ktoe" in unit_lc:
             return float(value) * _KTOE_TO_TJ
        
        return float(value) # assume TJ
         

    ##################################
    #    PUBLIC METHODS              #
    ##################################

    def aggregate_sisepuede(self,
        df_sisepuede: pd.DataFrame,
        col_year: str = "year",
    ) -> pd.DataFrame:

        """
        Apply the crosswalk to a SISEPUEDE model-output dataframe.

        For each crosswalk row:
        1. Splits sisepuede_output_variables on ":" to get column names.
        2. Keeps only columns that exist in df_sisepuede (missing columns are
           silently skipped — common for unused technologies or fuels).
        3. Sums across the surviving columns for each year.
        4. Multiplies by unit_conversion_to_tj (PJ -> TJ for energy).

        Function Arguments
        ------------------
        df_sisepuede : pd.DataFrame
            SISEPUEDE output dataframe. Must contain col_year.

        Keyword Arguments
        -----------------
        col_year : str
            Column holding the calendar year. Default "year". Add this column
            before calling by merging the time_period -> year map from the
            model input file.
        """

        available_cols = set(df_sisepuede.columns)
        rows = []

        meta_cols = [
            "iea_balance_code", "iea_balance_name",
            "iea_product_code", "iea_product_name",
            "sisepuede_subsector", "unit_sisepuede", "mapping_quality",             
        ]

        for _, xw_row in self.df_crosswalk.iterrows():
             
            ## PARSE VARIABLE LIST

            vars_requested = [
                v.strip()
                for v in str(xw_row["sisepuede_output_variables"]).split(_VAR_SEP)
                if v.strip()
            ]
            vars_found = [v for v in vars_requested if v in available_cols]

            if not vars_found:
                continue
             
            ## AGREGATE AND CONVERT

            df_agg = df_sisepuede[[col_year] + vars_found].copy()
            df_agg["value_sisepuede"] = df_agg[vars_found].sum(axis=1)

            conv = float(xw_row["unit_conversion_to_tj"])
            df_agg["value_sisepuede_tj"] = df_agg["value_sisepuede"] * conv

            ## ATTACH METADATA

            for col in meta_cols:
                df_agg[col] = xw_row[col]
            
            df_agg["sisepuede_vars_used"] = _VAR_SEP.join(vars_found)

            rows.append(
                 df_agg[
                      [col_year, "value_sisepuede_tj"]
                      + meta_cols
                      + ["sisepuede_vars_used"]
                 ]
            )
        
        if not rows:
             return pd.DataFrame()
        
        return pd.concat(rows, ignore_index=True)
    
    def build_comparison(self,
        df_sisepuede_long: pd.DataFrame,
        df_iea_long: pd.DataFrame,
        col_year: str = "year",
    ) -> pd.DataFrame:
        """
        Outer-join SISEPUEDE and IEA long frames on
        (iea_balance_code, iea_product_code, year).

        Outer join preserves rows missing on one side, making gaps visible.
        Adds ratio_sisepuede_over_iea = value_sisepuede_tj / value_iea_tj.

        ratio == 1  -> perfect agreement
        ratio  > 1  -> SISEPUEDE over-estimates
        ratio  < 1  -> SISEPUEDE under-estimates

        Function Arguments
        ------------------
        df_sisepuede_long : pd.DataFrame
            Output of aggregate_sisepuede().
        df_iea_long : pd.DataFrame
            Output of IEADataLoader.load_country().

        Keyword Arguments
        -----------------
        col_year : str
            Shared year column name. Default "year".
        """

        join_keys = ["iea_balance_code", "iea_product_code", col_year]

        df = df_sisepuede_long.merge(
            df_iea_long,
            on = join_keys,
            how = "outer",
        )

        df["ratio_sisepuede_over_iea"] = (
            df["value_sisepuede_tj"] / df["value_iea_tj"]
        )
        
        df["diff_sisepuede_iea"] = df["value_sisepuede_tj"] - df["value_iea_tj"]
        df["rel_err_sisepuede"]  = (df["value_sisepuede_tj"] - df["value_iea_tj"]) / df["value_sisepuede_tj"]
        df["rel_err_iea"]        = (df["value_iea_tj"] - df["value_sisepuede_tj"]) / df["value_iea_tj"]
        
        return df.sort_values(join_keys).reset_index(drop = True)
    

    def summary(self,
        df_comparison: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Collapse the comparison table to one row per (balance, product).

        Shows mean_ratio, n_years_matched, and mean values on both sides.
        Sorted by mean_ratio descending — rows far from 1.0 need calibration.

        Function Arguments
        ------------------
        df_comparison : pd.DataFrame
            Output of build_comparison().
        """

        group_cols = [
            "iea_balance_code", "iea_product_code",
            "iea_balance_name", "iea_product_name",
            "mapping_quality",
        ]
        group_cols = [c for c in group_cols if c in df_comparison.columns]

        return (
            df_comparison
            .groupby(group_cols, dropna = False)
            .agg(
                n_years_matched   = ("ratio_sisepuede_over_iea", "count"),
                mean_ratio        = ("ratio_sisepuede_over_iea", "mean"),
                mean_rel_err_iea  = ("rel_err_iea",              "mean"),
                mean_iea_tj       = ("value_iea_tj",             "mean"),
                mean_sisepuede_tj = ("value_sisepuede_tj",       "mean"),
            )
            .reset_index()
            .sort_values("mean_ratio", ascending = False)
            .reset_index(drop = True)
        )